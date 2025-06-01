# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch import Tensor
from typing import List, Optional, Dict

from util.misc import NestedTensor, nested_tensor_from_tensor_list
from util import box_ops
from ..registry import MODULE_BUILD_FUNCS
from .matcher import build_matcher

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SAMBackbone(nn.Module):
    """SAM backbone wrapper that works with DINO architecture"""
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model('samvit_base_patch16.sa1b',
                                      pretrained=pretrained,
                                      num_classes=0)  # remove classifier
        self.num_channels = [256]  # Output dimension from SAM's ViT backbone
        
        # Freeze the backbone parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Patch size for SAM
        self.patch_size = 16
    
    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        
        # Get original input dimensions
        batch_size, _, H, W = x.shape
        
        # Ensure input dimensions are divisible by patch size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            mask = F.pad(mask, (0, pad_w, 0, pad_h), value=True)  # Pad with True (masked)
        
        # Extract features from SAM model
        features = self.model.forward_features(x)
        
        # Calculate feature map dimensions
        h = (H + pad_h) // self.patch_size
        w = (W + pad_w) // self.patch_size
        
        # Reshape patch tokens to spatial layout [B, C, H, W]
        patch_tokens = features.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2)
        
        # Create corresponding mask for feature map
        mask_down = F.interpolate(mask.float().unsqueeze(1), size=(h, w)).to(torch.bool).squeeze(1)
        
        # Create output
        out = {}
        out[0] = NestedTensor(patch_tokens, mask_down)
        
        return out

class SAM_DET(nn.Module):
    """SAM-based detector that makes prediction for each patch"""
    def __init__(self, backbone, num_classes, num_queries=100, aux_loss=False):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.aux_loss = aux_loss
        
        # Feature dimension from backbone
        hidden_dim = backbone.num_channels[0]
        
        # Projection from backbone features to model dimension (identity in this case)
        self.input_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        
        # Classification and box regression heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        # Initialize the bbox embedding with small weights and biases
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
    
    def forward(self, samples: NestedTensor, targets: Optional[List[Dict[str, Tensor]]] = None):
        """Forward pass of the model"""
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        # Extract features from backbone
        features = self.backbone(samples)
        
        # Process the features
        src, mask = features[0].decompose()
        src = self.input_proj(src)
        
        # Extract spatial dimensions of feature map
        batch_size, feat_dim, feat_h, feat_w = src.shape
        
        # Reshape feature map to [batch_size, h*w, feat_dim]
        src_flatten = src.flatten(2).permute(0, 2, 1)  # [batch_size, H*W, C]
        mask_flatten = mask.flatten(1)  # [batch_size, H*W]
        
        # Compute predictions for each patch
        # For each patch, predict class and bbox
        outputs_class = self.class_embed(src_flatten)  # [batch_size, H*W, num_classes]
        outputs_coord = self.bbox_embed(src_flatten).sigmoid()  # [batch_size, H*W, 4]
        
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        
        return out

class SetCriterion(nn.Module):
    """Loss computation for SAM_DET"""
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.losses = ['labels', 'boxes', 'cardinality']
    
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss using focal loss"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        
        # Compute focal loss
        from .utils import sigmoid_focal_loss
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, 
                                    alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        
        losses = {'loss_ce': loss_ce}
        return losses
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute box regression loss"""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes if num_boxes > 0 else loss_bbox.sum()
        
        # GIoU loss
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes if num_boxes > 0 else loss_giou.sum()
        
        return losses
    
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute cardinality error (predicting right number of objects)"""
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are not "no-object"
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {'cardinality_error': card_err}
    
    def _get_src_permutation_idx(self, indices):
        # Permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx(self, indices):
        # Permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'cardinality': self.loss_cardinality,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def forward(self, outputs, targets):
        """Loss computation"""
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # Retrieve the matching between outputs and targets
        indices = self.matcher(outputs_without_aux, targets)
        
        # Compute number of target boxes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        
        return losses

class PostProcess(nn.Module):
    """Post-processing to convert model outputs to the expected format"""
    def __init__(self, num_select=100):
        super().__init__()
        self.num_select = num_select
    
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Convert model outputs to the desired format, selecting top-k predictions"""
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        
        # Apply sigmoid to get probabilities
        prob = out_logits.sigmoid()
        
        # Select top-k predictions over all spatial locations and classes
        # Flatten the first two dimensions (batch and spatial)
        prob_flat = prob.view(out_logits.shape[0], -1)
        topk_values, topk_indexes = torch.topk(prob_flat, self.num_select, dim=1)
        
        # Convert flat indices to (spatial_idx, class_idx) format
        topk_spatial_idx = topk_indexes // out_logits.shape[2]
        topk_class_idx = topk_indexes % out_logits.shape[2]
        
        # Extract scores and boxes
        scores = topk_values
        
        # Gather corresponding boxes
        boxes = torch.gather(out_bbox.view(out_bbox.shape[0], -1, 4), 1, 
                           topk_spatial_idx.unsqueeze(-1).expand(-1, -1, 4))
        
        # Convert to xyxy format
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        
        # Scale boxes to image size
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        
        results = [{'scores': s, 'labels': l, 'boxes': b} 
                  for s, l, b in zip(scores, topk_class_idx, boxes)]
        
        return results

@MODULE_BUILD_FUNCS.registe_with_name(module_name='samdet')
def build_samdet(args):
    """Build the SAM-based detector"""
    # Create SAM backbone
    backbone = SAMBackbone(pretrained=True)
    
    # Create full model
    model = SAM_DET(
        backbone=backbone,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss
    )
    
    # Create matcher
    matcher = build_matcher(args)
    
    # Create criterion
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
    
    criterion = SetCriterion(
        num_classes=args.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=args.focal_alpha
    )
    
    # Create post-processor
    postprocessors = {'bbox': PostProcess(num_select=args.num_select)}
    
    return model, criterion, postprocessors