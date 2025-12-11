# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
from collections import defaultdict

def process_jsonl_file_train_format(input_file, output_file, image_dir):
    """Process JSONL file and convert to train format (separate entries for each caption)"""
    
    result = []
    missing_images = 0
    image_id_mapping = {}  # Map actual image filename to sequential image_id
    current_image_id = 0
    
    print(f"Processing: {os.path.basename(input_file)} (train format)")
    
    # Read JSONL file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                text = data['text']
                image_ids = data['image_ids']
                
                # Process each image ID
                for img_id in image_ids:
                    # Check if image file exists
                    full_image_path = os.path.join(image_dir, f"{img_id}.jpg")
                    if not os.path.exists(full_image_path):
                        missing_images += 1
                        continue
                    
                    # Get or assign sequential image_id
                    image_filename = f"flickr30k-images/{img_id}.jpg"
                    if image_filename not in image_id_mapping:
                        image_id_mapping[image_filename] = current_image_id
                        current_image_id += 1
                    
                    # Create separate entry for each caption
                    item = {
                        "image": image_filename,
                        "caption": text,
                        "image_id": image_id_mapping[image_filename]
                    }
                    result.append(item)
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error (line {line_num}): {e}")
    
    # Sort by image_id for consistency
    result.sort(key=lambda x: (x['image_id'], x['caption']))
    
    # Write JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # Statistics
    num_captions = len(result)
    num_images = len(image_id_mapping)
    
    print(f"  ✓ Images: {num_images:,}, Captions: {num_captions:,}")
    if missing_images > 0:
        print(f"  ⚠ Missing images: {missing_images}")
    
    return num_images, num_captions


def process_jsonl_file_test_format(input_file, output_file, image_dir):
    """Process JSONL file and convert to test/val format (grouped captions per image)"""
    
    # Store texts grouped by image ID
    image_texts = defaultdict(list)
    
    print(f"Processing: {os.path.basename(input_file)} (test/val format)")
    
    # Read JSONL file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                text = data['text']
                image_ids = data['image_ids']
                
                # Add text description for each image ID
                for image_id in image_ids:
                    image_texts[image_id].append(text)
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error (line {line_num}): {e}")
    
    # Convert to target format
    result = []
    missing_images = 0
    
    for image_id, captions in image_texts.items():
        # Check if image file exists
        full_image_path = os.path.join(image_dir, f"{image_id}.jpg")
        if not os.path.exists(full_image_path):
            missing_images += 1
            continue
        
        # Create data item with grouped captions
        item = {
            "image": f"flickr30k-images/{image_id}.jpg",
            "caption": captions
        }
        result.append(item)
    
    # Sort by image field for consistency
    result.sort(key=lambda x: x['image'])
    
    # Write JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # Statistics
    num_images = len(result)
    num_captions = sum(len(item['caption']) for item in result)
    
    print(f"  ✓ Images: {num_images:,}, Captions: {num_captions:,}")
    if missing_images > 0:
        print(f"  ⚠ Missing images: {missing_images}")
    
    return num_images, num_captions


def convert_flickr30k_cn_to_json(base_dir=None):
    """Convert flickrcn JSONL format to flickr30k-like JSON format"""

    image_dir = os.path.join(base_dir, "flickr30k-images")
    
    # Dataset configurations with specific formats
    datasets = [
        ("train_texts.jsonl", "flickr30k_cn_train.json", "Training", "train"),
        ("valid_texts.jsonl", "flickr30k_cn_val.json", "Validation", "test"),
        ("test_texts.jsonl", "flickr30k_cn_test.json", "Test", "test")
    ]
    
    print("=== flickrcn Data Conversion ===")
    print("Note: Using different formats to match original flickr30k structure:")
    print("  - Train: Separate entries for each caption (with image_id)")
    print("  - Val/Test: Grouped captions per image\n")
    
    total_images = total_captions = 0
    results = {}
    
    # Process each dataset
    for input_name, output_name, dataset_type, format_type in datasets:
        input_file = os.path.join(base_dir, input_name)
        output_file = os.path.join(base_dir, output_name)
        
        if os.path.exists(input_file):
            if format_type == "train":
                images, captions = process_jsonl_file_train_format(input_file, output_file, image_dir)
            else:
                images, captions = process_jsonl_file_test_format(input_file, output_file, image_dir)
            
            total_images += images
            total_captions += captions
            results[dataset_type.lower()] = {"images": images, "captions": captions}
        else:
            print(f"❌ File not found: {input_name}")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Total images: {total_images:,}")
    print(f"Total captions: {total_captions:,}")
    print(f"Average captions per image: {total_captions/total_images:.1f}" if total_images > 0 else "N/A")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert flickrcn JSONL format to flickr30k-like JSON format")
    parser.add_argument("--base_dir", type=str, default="/path/to/your/flickrcn/dataset", 
                       help="Base directory containing flickrcn dataset files.")
    
    args = parser.parse_args()
    convert_flickr30k_cn_to_json(args.base_dir)