import os
import cv2
from pathlib import Path

def check_images(root_dir):
    root_path = Path(root_dir)
    print(f"Scanning directory: {root_path}")
    
    corrupted_count = 0
    total_count = 0
    
    # Recursively find all image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(root_path.rglob(ext))
        
    print(f"Found {len(image_files)} images. Checking integrity...")
    
    for file_path in image_files:
        total_count += 1
        try:
            # Try to read with OpenCV
            img = cv2.imread(str(file_path))
            
            # Additional check: decoded image must not be None and have size > 0
            if img is None:
                print(f"[CORRUPT] Cannot read: {file_path}")
                os.remove(file_path)
                corrupted_count += 1
            elif img.size == 0:
                print(f"[EMPTY] Zero size: {file_path}")
                os.remove(file_path)
                corrupted_count += 1
            else:
                # Optional: Try to decode full bytes to catch truncation
                # PIL is stricter than OpenCV usually
                with open(file_path, 'rb') as f:
                    f.seek(-2, 2)
                    if f.read() != b'\xff\xd9':
                        # JPEG should end with FFD9. 
                        # Not all valid JPEGs strictly follow this if there is metadata trailing, 
                        # but truncated ones definitely miss it.
                        # Let's rely on PIL for truncation check if cv2 passed
                        pass

        except Exception as e:
            print(f"[ERROR] Error reading {file_path}: {e}")
            os.remove(file_path)
            corrupted_count += 1
            
    # Second pass specifically for Truncated Images using PIL
    # OpenCV sometimes reads truncated images without error but returns partial data
    from PIL import Image, ImageFile
    # Ensure we catch truncated images as errors
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    
    for file_path in image_files:
        if not os.path.exists(file_path): continue
        try:
            with Image.open(file_path) as img:
                img.load() # Force load pixel data to trigger truncation error
        except (IOError, SyntaxError, OSError) as e:
            print(f"[PIL CORRUPT] Truncated/Invalid: {file_path} - {e}")
            try:
                os.remove(file_path)
                corrupted_count += 1
            except:
                pass

    print(f"\nScan complete.")
    print(f"Total processed: {total_count}")
    print(f"Corrupted/Deleted: {corrupted_count}")

if __name__ == "__main__":
    check_images("FaceShape Dataset")
