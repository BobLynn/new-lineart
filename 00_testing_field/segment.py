import os
import argparse
from datetime import datetime
from ultralytics.models.sam import SAM3SemanticPredictor

# 設定模型路徑 (指向專案根目錄下的 checkpoints 資料夾)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
model_path = os.path.join(project_root, "checkpoints", "sam3.pt")

if not os.path.exists(model_path):
    print(f"Warning: Model not found at {model_path}")
    # 如果使用者將模型放在其他地方，可以在這裡修改路徑
    # model_path = "path/to/your/custom/sam3.pt"

# Generate date-based project path
date_str = datetime.now().strftime("%Y-%m-%d")
project_path = os.path.join(project_root, "runs", "segment", date_str)

# Initialize predictor with configuration
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model=model_path,
    half=True,  # Use FP16 for faster inference
    save=True,
    project=project_path,
    name="predict",
)
predictor = SAM3SemanticPredictor(overrides=overrides)

# Set image once for multiple queries
# Parse command line arguments for image path
parser = argparse.ArgumentParser(description="SAM3 Segmentation Test")
parser.add_argument("--source", type=str, help="Path to the image file to segment")
args = parser.parse_args()

images_dir = os.path.join(current_dir, "images")

if args.source:
    image_path = args.source
    # If the user provided just a filename, check if it exists in images_dir
    if not os.path.exists(image_path) and not os.path.isabs(image_path):
         potential_path = os.path.join(images_dir, image_path)
         if os.path.exists(potential_path):
             image_path = potential_path
else:
    # Check if images directory exists
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        exit(1)

    # List image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"No image files found in {images_dir}")
        exit(1)
        
    print(f"\nAvailable images in {images_dir}:")
    for idx, filename in enumerate(image_files):
        print(f"{idx + 1}. {filename}")
    
    while True:
        try:
            selection = input("\nPlease select an image number (or enter filename): ").strip()
            if not selection:
                print("Selection cannot be empty.")
                continue
                
            # Check if user entered a number
            if selection.isdigit():
                idx = int(selection) - 1
                if 0 <= idx < len(image_files):
                    image_path = os.path.join(images_dir, image_files[idx])
                    break
                else:
                    print(f"Invalid number. Please enter a number between 1 and {len(image_files)}.")
            else:
                # Check if user entered a filename
                potential_path = os.path.join(images_dir, selection)
                if os.path.exists(potential_path) and selection in image_files:
                    image_path = potential_path
                    break
                else:
                    print("File not found or invalid selection.")
        except KeyboardInterrupt:
             print("\nOperation cancelled.")
             exit(1)

if not os.path.exists(image_path):
     print(f"Error: Image not found at {image_path}")
     exit(1)

print(f"Processing image: {image_path}")
predictor.set_image(image_path)

# Query with multiple text prompts
results = predictor(text=["person", "bus", "glasses"])

# Works with descriptive phrases
results = predictor(text=["person with red cloth", "person with blue cloth"])

# Query with a single concept
results = predictor(text=["a person"])