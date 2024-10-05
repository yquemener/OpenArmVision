"""
This script converts a dataset from an old format to a new format.

Features:
- Reads a .db file from the input directory to get annotations
- Copies images from 'images' folder to 'keyframes' in the output directory
- Copies images from 'candidates' folder to 'candidates' in the output directory
- Creates a new dataset.db file in the output directory using the DatabaseManager
- Converts annotations from the old format to the new format
- Creates the output directory if it doesn't exist
- Deletes and recreates the output directory if it's the default 'new_dataset/'
- Displays progress bars for image copying and annotation processing

Usage:
python old_to_new_db.py <command> [args]

Commands:
  convert <input_dir> <output_dir>
  verify_new <db_path>
  verify_old <db_path> <images_dir>

Constraints:
- Input directory must contain a .db file and 'images' and 'candidates' folders
- Script requires write permissions in the parent directory of output_dir
- Requires PIL (Pillow) library for image processing
- Requires tqdm library for progress bars

Considerations:
- Large datasets may take a while to process
- Ensure sufficient disk space in the output location
- Annotations are converted from normalized coordinates to pixel coordinates
"""

import os
import sys
import shutil
import sqlite3
from PIL import Image
from database import DatabaseManager, Image as DBImage, Annotation as DBAnnotation
from tqdm import tqdm
from datetime import datetime

def convert_annotation_type(old_type):
    type_mapping = {1: 'point', 2: 'line', 3: 'rectangle'}
    return type_mapping.get(old_type, 'unknown')

def convert_coordinates(center_x, center_y, width, height, image_width, image_height, annotation_type):
    if center_x is None or center_y is None:
        print(f"Warning: Invalid coordinates (center_x={center_x}, center_y={center_y}) for image size {image_width}x{image_height}")
        return None, None, None, None
    
    if annotation_type == 'point':
        new_x = int(center_x * image_width)
        new_y = int(center_y * image_height)
        return new_x, new_y, new_x, new_y  # For points, x2 and y2 are the same as x1 and y1
    elif annotation_type == 'rectangle':
        if width is not None and height is not None:
            new_width = int(width * image_width)
            new_height = int(height * image_height)
            new_x1 = int((center_x * image_width) - (new_width / 2))
            new_y1 = int((center_y * image_height) - (new_height / 2))
            new_x2 = new_x1 + new_width
            new_y2 = new_y1 + new_height
            return new_x1, new_y1, new_x2, new_y2
        else:
            print(f"Warning: Invalid dimensions for rectangle (width={width}, height={height})")
            return None, None, None, None
    else:
        print(f"Warning: Unknown annotation type: {annotation_type}")
        return None, None, None, None

def process_annotations(old_db_path, new_db_manager, image_path_mapping):
    conn = sqlite3.connect(old_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM annotations")
    total_annotations = cursor.fetchone()[0]
    
    cursor.execute("SELECT * FROM annotations")
    
    annotations_added = 0
    annotations_skipped = 0

    all_image_paths = {os.path.splitext(os.path.basename(path))[0]: path for path in image_path_mapping.values()}

    for annotation in tqdm(cursor, total=total_annotations, desc="Processing annotations"):
        image_id, file_id, old_type, center_x, center_y, width, height, class_id, _ = annotation
        new_type = convert_annotation_type(old_type)
        
        if file_id in all_image_paths:
            new_image_path = all_image_paths[file_id]
            image = new_db_manager.get_image_by_path(new_image_path)
            
            if image:
                new_x1, new_y1, new_x2, new_y2 = convert_coordinates(center_x, center_y, width, height, image.width, image.height, new_type)
                
                if None in (new_x1, new_y1, new_x2, new_y2):
                    print(f"Skipping invalid annotation for image {image_id}")
                    annotations_skipped += 1
                    continue
                
                try:
                    new_db_manager.add_annotation(
                        image_id=image.id,
                        annotation_type=new_type,
                        x1=new_x1, y1=new_y1,
                        x2=new_x2, y2=new_y2,
                        class_name=str(class_id)
                    )
                    annotations_added += 1
                except Exception as e:
                    print(f"Error adding annotation: {e}")
                    annotations_skipped += 1
            else:
                print(f"Warning: Image found in mapping but not in database: {new_image_path}")
                annotations_skipped += 1
        else:
            print(f"Warning: No matching image found for: {file_id}")
            annotations_skipped += 1
    
    conn.close()
    print(f"Annotations processed: {annotations_added} added, {annotations_skipped} skipped")

def convert_dataset(input_dir, output_dir):
    """
    Convert the dataset from the old format to the new format.
    Handle duplicate images between 'candidates' and 'keyframes' folders.
    """
    print(f"Converting dataset from {input_dir} to {output_dir}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'keyframes'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'candidates'), exist_ok=True)
    
    # Initialize new database
    new_db_path = os.path.join(output_dir, 'dataset.db')
    new_db_manager = DatabaseManager(new_db_path)
    
    # Copy images and create mapping
    image_path_mapping = {}
    keyframes = set()
    
    # First, process keyframes
    keyframes_dir = os.path.join(input_dir, 'images')
    for filename in os.listdir(keyframes_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            src_path = os.path.join(keyframes_dir, filename)
            dst_path = os.path.join(output_dir, 'keyframes', filename)
            shutil.copy2(src_path, dst_path)
            image_path_mapping[filename] = f'keyframes/{filename}'
            keyframes.add(filename)
    
    # Then, process candidates
    candidates_dir = os.path.join(input_dir, 'candidates')
    for filename in os.listdir(candidates_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            if filename not in keyframes:
                src_path = os.path.join(candidates_dir, filename)
                dst_path = os.path.join(output_dir, 'candidates', filename)
                shutil.copy2(src_path, dst_path)
                image_path_mapping[filename] = f'candidates/{filename}'
    
    # Process annotations
    old_db_path = next(f for f in os.listdir(input_dir) if f.endswith('.db'))
    old_db_path = os.path.join(input_dir, old_db_path)
    
    conn = sqlite3.connect(old_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM annotations")
    total_annotations = cursor.fetchone()[0]
    
    cursor.execute("SELECT * FROM annotations")
    
    annotations_added = 0
    annotations_skipped = 0
    
    for annotation in tqdm(cursor, total=total_annotations, desc="Processing annotations"):
        image_id, file_id, old_type, x, y, width, height, class_id, _ = annotation
        new_type = convert_annotation_type(old_type)
        
        filename = f"{file_id}.jpg"  # Assuming all images are .jpg
        
        if filename in image_path_mapping:
            new_image_path = image_path_mapping[filename]
            image = new_db_manager.get_image_by_path(new_image_path)
            
            if image:
                new_x1, new_y1, new_x2, new_y2 = convert_coordinates(x, y, width, height, image.width, image.height, new_type)
                
                if None not in (new_x1, new_y1, new_x2, new_y2):
                    try:
                        new_db_manager.add_annotation(
                            image_id=image.id,
                            annotation_type=new_type,
                            x1=new_x1, y1=new_y1,
                            x2=new_x2, y2=new_y2,
                            class_name=str(class_id)
                        )
                        annotations_added += 1
                    except Exception as e:
                        print(f"Error adding annotation: {e}")
                        annotations_skipped += 1
                else:
                    print(f"Skipping invalid annotation for image {image_id}")
                    annotations_skipped += 1
            else:
                print(f"Warning: Image not found in new database: {new_image_path}")
                annotations_skipped += 1
        else:
            print(f"Warning: No matching image found for: {file_id}")
            annotations_skipped += 1
    
    conn.close()
    print(f"Annotations processed: {annotations_added} added, {annotations_skipped} skipped")

    return new_db_manager

def verify_new_dataset(db_path):
    """
    Verify the integrity of a new dataset:
    1. Check that all annotations are referenced to existing images.
    2. Ensure all images in the 'keyframes' folder have at least one annotation.
    3. Check if keyframes without annotations also exist in the 'candidates' folder.
    """
    print("Verifying new dataset...")
    db_manager = DatabaseManager(db_path)
    
    with db_manager.SessionLocal() as session:
        total_annotations = session.query(DBAnnotation).count()
        valid_annotations = 0
        orphaned_annotations = []
        
        # Check annotations
        annotations = session.query(DBAnnotation, DBImage.path).join(DBImage).all()
        for annotation, image_path in tqdm(annotations, total=total_annotations, desc="Checking annotations"):
            if os.path.exists(os.path.join(os.path.dirname(db_path), image_path)):
                valid_annotations += 1
            else:
                orphaned_annotations.append((annotation.id, image_path))
        
        # Check keyframes
        keyframes = session.query(DBImage).filter(DBImage.path.like('keyframes/%')).all()
        total_keyframes = len(keyframes)
        keyframes_with_annotations = 0
        keyframes_without_annotations = []
        keyframes_also_in_candidates = []
        
        for keyframe in tqdm(keyframes, total=total_keyframes, desc="Checking keyframes"):
            annotation_count = session.query(DBAnnotation).filter(DBAnnotation.image_id == keyframe.id).count()
            if annotation_count > 0:
                keyframes_with_annotations += 1
            else:
                keyframes_without_annotations.append(keyframe.path)
                # Check if this keyframe also exists in the candidates folder
                candidate_path = keyframe.path.replace('keyframes/', 'candidates/')
                if session.query(DBImage).filter(DBImage.path == candidate_path).first():
                    keyframes_also_in_candidates.append(keyframe.path)
    
    print(f"Annotation check:")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Valid annotations: {valid_annotations}")
    print(f"  Orphaned annotations: {len(orphaned_annotations)}")
    
    if orphaned_annotations:
        print("Orphaned annotation details:")
        for ann_id, img_path in orphaned_annotations:
            print(f"  Annotation ID: {ann_id}, Referenced image: {img_path}")
    
    print(f"\nKeyframe check:")
    print(f"  Total keyframes: {total_keyframes}")
    print(f"  Keyframes with annotations: {keyframes_with_annotations}")
    print(f"  Keyframes without annotations: {len(keyframes_without_annotations)}")
    print(f"  Keyframes without annotations but present in candidates: {len(keyframes_also_in_candidates)}")
    
    if keyframes_without_annotations:
        print("\nKeyframes without annotations:")
        for keyframe_path in keyframes_without_annotations:
            print(f"  {keyframe_path}")

    if keyframes_also_in_candidates:
        print("\nKeyframes without annotations but present in candidates:")
        for keyframe_path in keyframes_also_in_candidates:
            print(f"  {keyframe_path}")
    
    return (total_annotations, valid_annotations, orphaned_annotations,
            total_keyframes, keyframes_with_annotations, keyframes_without_annotations, keyframes_also_in_candidates)

def verify_old_dataset(db_path, images_dir):
    """
    Verify the presence of annotations in the keyframes of the old dataset and check if keyframes without annotations exist in the candidates folder.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("Verifying old dataset...")
    
    # Get all images in the keyframes directory
    keyframes_dir = os.path.join(images_dir, 'images')  # Assuming 'images' is the keyframes directory
    candidates_dir = os.path.join(images_dir, 'candidates')
    
    keyframes = [f for f in os.listdir(keyframes_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_keyframes = len(keyframes)
    keyframes_with_annotations = 0
    keyframes_without_annotations = []
    keyframes_also_in_candidates = []

    for keyframe in tqdm(keyframes, desc="Checking keyframes"):
        cursor.execute("SELECT COUNT(*) FROM annotations WHERE file_id LIKE ?", (f"%{os.path.splitext(keyframe)[0]}%",))
        annotation_count = cursor.fetchone()[0]
        
        if annotation_count > 0:
            keyframes_with_annotations += 1
        else:
            keyframes_without_annotations.append(keyframe)
            # Check if this keyframe also exists in the candidates folder
            if os.path.exists(os.path.join(candidates_dir, keyframe)):
                keyframes_also_in_candidates.append(keyframe)

    print(f"\nOld dataset keyframe check:")
    print(f"  Total keyframes: {total_keyframes}")
    print(f"  Keyframes with annotations: {keyframes_with_annotations}")
    print(f"  Keyframes without annotations: {len(keyframes_without_annotations)}")
    print(f"  Keyframes without annotations but present in candidates: {len(keyframes_also_in_candidates)}")

    if keyframes_without_annotations:
        print("\nKeyframes without annotations:")
        for keyframe in keyframes_without_annotations:
            print(f"  {keyframe}")

    if keyframes_also_in_candidates:
        print("\nKeyframes without annotations but present in candidates:")
        for keyframe in keyframes_also_in_candidates:
            print(f"  {keyframe}")

    conn.close()
    return total_keyframes, keyframes_with_annotations, keyframes_without_annotations, keyframes_also_in_candidates

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python old_to_new_db.py <command> [args]")
        print("Commands:")
        print("  convert <input_dir> <output_dir>")
        print("  verify_new <db_path>")
        print("  verify_old <db_path> <images_dir>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "convert":
        if len(sys.argv) < 4:
            input_dir = '../visionUI/vhelio_holes/'
            output_dir = 'new_dataset/'
        else:
            input_dir = sys.argv[2]
            output_dir = sys.argv[3]
        new_db_manager = convert_dataset(input_dir, output_dir)
        print(f"\nDataset converted from {input_dir} to {output_dir}")
        verify_new_dataset(new_db_manager.db_file)

    elif command == "verify_new":
        if len(sys.argv) < 3:
            print("Please provide the path to the new database.")
            sys.exit(1)
        db_path = sys.argv[2]
        (total_annotations, valid_annotations, orphaned_annotations,
         total_keyframes, keyframes_with_annotations, keyframes_without_annotations,
         keyframes_also_in_candidates) = verify_new_dataset(db_path)
        
        print("\nSummary:")
        print(f"Total annotations: {total_annotations}")
        print(f"Valid annotations: {valid_annotations}")
        print(f"Orphaned annotations: {len(orphaned_annotations)}")
        print(f"Total keyframes: {total_keyframes}")
        print(f"Keyframes with annotations: {keyframes_with_annotations}")
        print(f"Keyframes without annotations: {len(keyframes_without_annotations)}")
        print(f"Keyframes without annotations but present in candidates: {len(keyframes_also_in_candidates)}")

    elif command == "verify_old":
        if len(sys.argv) < 4:
            print("Please provide the path to the old database and the images directory.")
            sys.exit(1)
        db_path = sys.argv[2]
        images_dir = sys.argv[3]
        total_keyframes, keyframes_with_annotations, keyframes_without_annotations, keyframes_also_in_candidates = verify_old_dataset(db_path, images_dir)
        
        print("\nSummary:")
        print(f"Total keyframes: {total_keyframes}")
        print(f"Keyframes with annotations: {keyframes_with_annotations}")
        print(f"Keyframes without annotations: {len(keyframes_without_annotations)}")
        print(f"Keyframes without annotations but present in candidates: {len(keyframes_also_in_candidates)}")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)