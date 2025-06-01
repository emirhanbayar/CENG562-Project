import json
from pprint import pprint
from collections import defaultdict

def explore_json_structure(json_file_path):
    # Read and parse JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    def analyze_structure(obj, path="root"):
        structure = defaultdict(set)
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}"
                structure["type"].add(f"{current_path}: dict")
                nested_structure = analyze_structure(value, current_path)
                for k, v in nested_structure.items():
                    structure[k].update(v)
                    
        elif isinstance(obj, list):
            structure["type"].add(f"{path}: list[{len(obj)}]")
            if obj:  # Analyze first item as sample
                nested_structure = analyze_structure(obj[0], f"{path}[0]")
                for k, v in nested_structure.items():
                    structure[k].update(v)
                    
        else:
            structure["type"].add(f"{path}: {type(obj).__name__}")
            
        return structure

    # Analyze and print structure
    structure = analyze_structure(data)
    print("\nJSON Structure Analysis:")
    pprint(dict(structure))

# Usage example
explore_json_structure("/home/emirhan/datasets/object_detection/FLIR_ADAS_IR/annotations/instances_val2017.json")