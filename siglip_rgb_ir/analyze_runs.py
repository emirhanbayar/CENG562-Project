#!/usr/bin/env python3
"""
Comprehensive Results Analysis for Grid Search Experiments
This script analyzes all completed experiments and generates summary reports.
"""

import os
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def parse_experiment_name(exp_name):
    """Parse experiment name to extract parameters"""
    try:
        # Expected format: modality_lr{lr}_bs{bs}_bbox{bbox}giou{giou}class{class}_wu{warmup}
        parts = exp_name.split('_')
        
        params = {}
        
        # Find modality (first part after prefix)
        for i, part in enumerate(parts):
            if part in ['rgb', 'ir', 'concat']:
                params['modality'] = part
                break
        
        # Parse other parameters
        for part in parts:
            if part.startswith('lr'):
                lr_str = part[2:]
                # Convert back from shortened format (e.g., '001' -> '0.001')
                if len(lr_str) == 2:
                    params['learning_rate'] = float(f"0.{lr_str}")
                elif len(lr_str) == 3:
                    params['learning_rate'] = float(f"0.{lr_str}")
                elif len(lr_str) == 4:
                    params['learning_rate'] = float(f"0.{lr_str}")
                else:
                    params['learning_rate'] = float(lr_str)
                    
            elif part.startswith('bs'):
                params['batch_size'] = int(part[2:])
                
            elif part.startswith('wu'):
                params['warmup_epochs'] = int(part[2:])
                
            elif part.startswith('loss') or (len(part) >= 3 and part[0].isdigit()):
                # Parse loss coefficients (format: loss{bbox}{giou}{class} or just {bbox}{giou}{class})
                loss_part = part[4:] if part.startswith('loss') else part
                if len(loss_part) >= 3:
                    params['bbox_coef'] = float(loss_part[0])
                    params['giou_coef'] = float(loss_part[1])
                    params['class_coef'] = float(loss_part[2])
        
        return params
        
    except Exception as e:
        print(f"Warning: Could not parse experiment name '{exp_name}': {e}")
        return {}

def load_experiment_results(experiments_dir):
    """Load results from all completed experiments"""
    results = []
    
    for exp_dir in Path(experiments_dir).iterdir():
        if not exp_dir.is_dir():
            continue
            
        exp_name = exp_dir.name
        metrics_file = exp_dir / 'metrics' / 'training_metrics.csv'
        args_file = exp_dir / 'args.json'
        
        if not metrics_file.exists():
            continue
            
        try:
            # Load metrics
            metrics_df = pd.read_csv(metrics_file)
            
            # Load arguments if available
            args = {}
            if args_file.exists():
                with open(args_file, 'r') as f:
                    args = json.load(f)
            
            # Parse experiment parameters from name as backup
            parsed_params = parse_experiment_name(exp_name)
            
            # Combine args and parsed params (args take precedence)
            params = {**parsed_params, **args}
            
            # Extract best metrics
            best_val_loss = metrics_df['val_loss'].min()
            best_train_loss = metrics_df['train_loss'].min()
            
            # Handle AP metrics (might have NaN values)
            ap_cols = ['ap', 'ap_50', 'ap_75', 'ar_100']
            ap_metrics = {}
            for col in ap_cols:
                if col in metrics_df.columns:
                    valid_values = metrics_df[col].dropna()
                    if len(valid_values) > 0:
                        ap_metrics[f'best_{col}'] = valid_values.max()
                        ap_metrics[f'final_{col}'] = valid_values.iloc[-1] if len(valid_values) > 0 else np.nan
                    else:
                        ap_metrics[f'best_{col}'] = np.nan
                        ap_metrics[f'final_{col}'] = np.nan
            
            # Training stability metrics
            val_loss_std = metrics_df['val_loss'].std()
            final_val_loss = metrics_df['val_loss'].iloc[-1]
            
            result = {
                'experiment_name': exp_name,
                'experiment_dir': str(exp_dir),
                'best_val_loss': best_val_loss,
                'final_val_loss': final_val_loss,
                'best_train_loss': best_train_loss,
                'val_loss_std': val_loss_std,
                'num_epochs_completed': len(metrics_df),
                **ap_metrics,
                **params
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {exp_name}: {e}")
            continue
    
    return pd.DataFrame(results)

def create_summary_plots(df, output_dir):
    """Create comprehensive summary plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Performance vs Learning Rate
    if 'learning_rate' in df.columns and 'modality' in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Val Loss vs LR
        for modality in df['modality'].unique():
            if pd.notna(modality):
                subset = df[df['modality'] == modality]
                axes[0,0].scatter(subset['learning_rate'], subset['best_val_loss'], 
                                label=modality, alpha=0.7, s=60)
        axes[0,0].set_xlabel('Learning Rate')
        axes[0,0].set_ylabel('Best Validation Loss')
        axes[0,0].set_xscale('log')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_title('Validation Loss vs Learning Rate by Modality')
        
        # AP vs LR
        if 'best_ap' in df.columns:
            for modality in df['modality'].unique():
                if pd.notna(modality):
                    subset = df[df['modality'] == modality]
                    valid_ap = subset.dropna(subset=['best_ap'])
                    if len(valid_ap) > 0:
                        axes[0,1].scatter(valid_ap['learning_rate'], valid_ap['best_ap'], 
                                        label=modality, alpha=0.7, s=60)
            axes[0,1].set_xlabel('Learning Rate')
            axes[0,1].set_ylabel('Best AP')
            axes[0,1].set_xscale('log')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].set_title('AP vs Learning Rate by Modality')
        
        # Batch Size Analysis
        if 'batch_size' in df.columns:
            df_batch = df.groupby(['modality', 'batch_size'])['best_val_loss'].mean().reset_index()
            sns.barplot(data=df_batch, x='batch_size', y='best_val_loss', hue='modality', ax=axes[1,0])
            axes[1,0].set_title('Average Validation Loss by Batch Size')
            axes[1,0].grid(True, alpha=0.3)
        
        # Loss Coefficient Analysis
        if all(col in df.columns for col in ['bbox_coef', 'giou_coef']):
            df['loss_config'] = df['bbox_coef'].astype(str) + '_' + df['giou_coef'].astype(str)
            df_loss = df.groupby('loss_config')['best_val_loss'].mean().reset_index()
            axes[1,1].bar(range(len(df_loss)), df_loss['best_val_loss'])
            axes[1,1].set_xticks(range(len(df_loss)))
            axes[1,1].set_xticklabels(df_loss['loss_config'], rotation=45)
            axes[1,1].set_xlabel('Loss Config (bbox_giou)')
            axes[1,1].set_ylabel('Average Validation Loss')
            axes[1,1].set_title('Performance by Loss Configuration')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Modality Comparison
    if 'modality' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot of validation loss by modality
        valid_modalities = df.dropna(subset=['modality'])
        if len(valid_modalities) > 0:
            sns.boxplot(data=valid_modalities, x='modality', y='best_val_loss', ax=axes[0])
            axes[0].set_title('Validation Loss Distribution by Modality')
            axes[0].grid(True, alpha=0.3)
        
        # AP comparison if available
        if 'best_ap' in df.columns:
            valid_ap_data = df.dropna(subset=['modality', 'best_ap'])
            if len(valid_ap_data) > 0:
                sns.boxplot(data=valid_ap_data, x='modality', y='best_ap', ax=axes[1])
                axes[1].set_title('AP Distribution by Modality')
                axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'modality_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Top Performers Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Top 10 by validation loss
    top_val = df.nsmallest(10, 'best_val_loss')
    axes[0,0].barh(range(len(top_val)), top_val['best_val_loss'])
    axes[0,0].set_yticks(range(len(top_val)))
    axes[0,0].set_yticklabels([name[:30] + '...' if len(name) > 30 else name 
                             for name in top_val['experiment_name']], fontsize=8)
    axes[0,0].set_xlabel('Best Validation Loss')
    axes[0,0].set_title('Top 10 Experiments (Lowest Val Loss)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Top 10 by AP if available
    if 'best_ap' in df.columns:
        valid_ap = df.dropna(subset=['best_ap'])
        if len(valid_ap) > 0:
            top_ap = valid_ap.nlargest(10, 'best_ap')
            axes[0,1].barh(range(len(top_ap)), top_ap['best_ap'])
            axes[0,1].set_yticks(range(len(top_ap)))
            axes[0,1].set_yticklabels([name[:30] + '...' if len(name) > 30 else name 
                                     for name in top_ap['experiment_name']], fontsize=8)
            axes[0,1].set_xlabel('Best AP')
            axes[0,1].set_title('Top 10 Experiments (Highest AP)')
            axes[0,1].grid(True, alpha=0.3)
    
    # Training stability
    df['stability_score'] = 1 / (1 + df['val_loss_std'])  # Higher is more stable
    top_stable = df.nlargest(10, 'stability_score')
    axes[1,0].barh(range(len(top_stable)), top_stable['stability_score'])
    axes[1,0].set_yticks(range(len(top_stable)))
    axes[1,0].set_yticklabels([name[:30] + '...' if len(name) > 30 else name 
                             for name in top_stable['experiment_name']], fontsize=8)
    axes[1,0].set_xlabel('Stability Score')
    axes[1,0].set_title('Most Stable Training (Low Variance)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Convergence analysis
    df['convergence_ratio'] = df['best_val_loss'] / df['final_val_loss']
    best_convergence = df.nlargest(10, 'convergence_ratio')
    axes[1,1].barh(range(len(best_convergence)), best_convergence['convergence_ratio'])
    axes[1,1].set_yticks(range(len(best_convergence)))
    axes[1,1].set_yticklabels([name[:30] + '...' if len(name) > 30 else name 
                             for name in best_convergence['experiment_name']], fontsize=8)
    axes[1,1].set_xlabel('Best/Final Loss Ratio')
    axes[1,1].set_title('Best Convergence (Didn\'t Overfit)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_performers.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(df, output_file):
    """Generate a comprehensive text summary report"""
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GRID SEARCH RESULTS SUMMARY REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Experiments Analyzed: {len(df)}\n\n")
        
        # Overall Statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best Validation Loss: {df['best_val_loss'].min():.6f}\n")
        f.write(f"Worst Validation Loss: {df['best_val_loss'].max():.6f}\n")
        f.write(f"Mean Validation Loss: {df['best_val_loss'].mean():.6f}\n")
        f.write(f"Std Validation Loss: {df['best_val_loss'].std():.6f}\n")
        
        if 'best_ap' in df.columns:
            valid_ap = df['best_ap'].dropna()
            if len(valid_ap) > 0:
                f.write(f"Best AP: {valid_ap.max():.6f}\n")
                f.write(f"Mean AP: {valid_ap.mean():.6f}\n")
        f.write("\n")
        
        # Top Performers
        f.write("TOP 5 EXPERIMENTS (BY VALIDATION LOSS)\n")
        f.write("-" * 40 + "\n")
        top_5 = df.nsmallest(5, 'best_val_loss')
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            f.write(f"{i}. {row['experiment_name']}\n")
            f.write(f"   Val Loss: {row['best_val_loss']:.6f}")
            if 'best_ap' in row and pd.notna(row['best_ap']):
                f.write(f", AP: {row['best_ap']:.6f}")
            f.write("\n")
            if 'modality' in row and pd.notna(row['modality']):
                f.write(f"   Modality: {row['modality']}")
            if 'learning_rate' in row and pd.notna(row['learning_rate']):
                f.write(f", LR: {row['learning_rate']}")
            if 'batch_size' in row and pd.notna(row['batch_size']):
                f.write(f", BS: {int(row['batch_size'])}")
            f.write("\n\n")
        
        # Modality Analysis
        if 'modality' in df.columns:
            f.write("ANALYSIS BY MODALITY\n")
            f.write("-" * 40 + "\n")
            modality_stats = df.groupby('modality')['best_val_loss'].agg(['count', 'mean', 'std', 'min']).round(6)
            f.write(modality_stats.to_string())
            f.write("\n\n")
        
        # Learning Rate Analysis
        if 'learning_rate' in df.columns:
            f.write("ANALYSIS BY LEARNING RATE\n")
            f.write("-" * 40 + "\n")
            lr_stats = df.groupby('learning_rate')['best_val_loss'].agg(['count', 'mean', 'std', 'min']).round(6)
            f.write(lr_stats.to_string())
            f.write("\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        # Best modality
        if 'modality' in df.columns:
            best_modality = df.loc[df['best_val_loss'].idxmin(), 'modality']
            modality_avg = df.groupby('modality')['best_val_loss'].mean()
            f.write(f"1. Best performing modality: {best_modality}\n")
            f.write(f"   Average performance by modality:\n")
            for mod, avg_loss in modality_avg.items():
                f.write(f"   - {mod}: {avg_loss:.6f}\n")
        
        # Best learning rate
        if 'learning_rate' in df.columns:
            best_lr = df.loc[df['best_val_loss'].idxmin(), 'learning_rate']
            f.write(f"\n2. Best learning rate: {best_lr}\n")
            
            # Learning rate range recommendation
            lr_performance = df.groupby('learning_rate')['best_val_loss'].mean().sort_values()
            top_lrs = lr_performance.head(3)
            f.write("   Top 3 learning rates by average performance:\n")
            for lr, avg_loss in top_lrs.items():
                f.write(f"   - {lr}: {avg_loss:.6f}\n")
        
        # Stability recommendations
        df_temp = df.copy()
        df_temp['stability_score'] = 1 / (1 + df_temp['val_loss_std'])
        most_stable = df_temp.loc[df_temp['stability_score'].idxmax()]
        f.write(f"\n3. Most stable training: {most_stable['experiment_name']}\n")
        f.write(f"   (Low variance in validation loss)\n")
        
        f.write("\n" + "=" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze grid search results')
    parser.add_argument('--experiments_dir', type=str, default='experiments_nms_grid_search',
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='analysis_results_nms',
                       help='Directory to save analysis results')
    parser.add_argument('--min_epochs', type=int, default=10,
                       help='Minimum epochs required for analysis')
    
    args = parser.parse_args()
    
    print("Loading experiment results...")
    df = load_experiment_results(args.experiments_dir)
    
    if len(df) == 0:
        print("No experiments found!")
        return
    
    # Filter by minimum epochs
    df = df[df['num_epochs_completed'] >= args.min_epochs]
    
    print(f"Found {len(df)} completed experiments")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(df, os.path.join(args.output_dir, 'summary_report.txt'))
    
    # Create plots
    print("Creating analysis plots...")
    create_summary_plots(df, args.output_dir)
    
    # Save detailed CSV
    df.to_csv(os.path.join(args.output_dir, 'all_results.csv'), index=False)
    
    print(f"Analysis complete! Results saved to: {args.output_dir}")
    print("\nQuick Summary:")
    print(f"- Total experiments: {len(df)}")
    print(f"- Best validation loss: {df['best_val_loss'].min():.6f}")
    if 'best_ap' in df.columns:
        valid_ap = df['best_ap'].dropna()
        if len(valid_ap) > 0:
            print(f"- Best AP: {valid_ap.max():.6f}")
    
    # Show top 3 experiments
    print("\nTop 3 experiments:")
    top_3 = df.nsmallest(3, 'best_val_loss')
    for i, (_, row) in enumerate(top_3.iterrows(), 1):
        print(f"{i}. {row['experiment_name']} (Loss: {row['best_val_loss']:.6f})")

if __name__ == "__main__":
    main()