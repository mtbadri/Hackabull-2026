#!/usr/bin/env python3
"""
Script to help collect profile images for face recognition testing.

Since there are no profile images in the repository, this script:
1. Shows what images are needed
2. Provides instructions for collecting them
3. Can generate placeholder images for testing
"""

import json
from pathlib import Path
import cv2
import numpy as np
import os

KNOWN_FACES_DIR = Path("services/vision/known_faces")
PROFILE_IMAGES_DIR = Path("tests/vision/profile_images")

def check_existing_images():
    """Check what profile images already exist."""
    print("="*60)
    print("CHECKING EXISTING PROFILE IMAGES")
    print("="*60)
    
    # Check known_faces directory
    print(f"\nKnown faces directory: {KNOWN_FACES_DIR}")
    if KNOWN_FACES_DIR.exists():
        json_files = list(KNOWN_FACES_DIR.glob("*.json"))
        print(f"Found {len(json_files)} JSON profile files:")
        
        for json_file in json_files:
            try:
                profile = json.loads(json_file.read_text())
                name = profile.get("name", json_file.stem)
                shirt_color = profile.get("shirt_color", "unknown")
                
                # Check for corresponding image files
                image_extensions = ['.jpg', '.jpeg', '.png']
                has_image = False
                for ext in image_extensions:
                    image_file = json_file.with_suffix(ext)
                    if image_file.exists():
                        has_image = True
                        print(f"  ✓ {name}: {json_file.name} + {image_file.name}")
                        break
                
                if not has_image:
                    print(f"  ✗ {name}: {json_file.name} (NO IMAGE FOUND)")
                    print(f"     Shirt color: {shirt_color}")
            except Exception as e:
                print(f"  Error reading {json_file.name}: {e}")
    else:
        print("Known faces directory not found!")
    
    # Check test images directory
    print(f"\nTest images directory: {PROFILE_IMAGES_DIR}")
    if PROFILE_IMAGES_DIR.exists():
        image_files = list(PROFILE_IMAGES_DIR.glob("*.*"))
        print(f"Found {len(image_files)} image files in test directory")
    else:
        print("Test images directory not found!")

def create_placeholder_images():
    """Create placeholder images for testing."""
    print("\n" + "="*60)
    print("CREATING PLACEHOLDER IMAGES FOR TESTING")
    print("="*60)
    
    # Create directory if it doesn't exist
    PROFILE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load profiles
    if not KNOWN_FACES_DIR.exists():
        print("Known faces directory not found!")
        return
    
    json_files = list(KNOWN_FACES_DIR.glob("*.json"))
    if not json_files:
        print("No JSON profile files found!")
        return
    
    print(f"\nCreating placeholder images for {len(json_files)} profiles:")
    
    for json_file in json_files:
        try:
            profile = json.loads(json_file.read_text())
            name = profile.get("name", json_file.stem)
            shirt_color = profile.get("shirt_color", "gray")
            
            # Create a simple placeholder image
            # In a real system, you'd use actual face photos
            image_path = PROFILE_IMAGES_DIR / f"{name}.jpg"
            
            # Create a simple colored image with text
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Set background color based on shirt color
            color_map = {
                "green": (0, 255, 0),
                "yellow": (0, 255, 255),
                "black": (0, 0, 0),
                "gray": (128, 128, 128),
                "blue": (255, 0, 0),
                "red": (0, 0, 255),
            }
            
            bg_color = color_map.get(shirt_color.lower(), (128, 128, 128))
            img[:] = bg_color
            
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Placeholder: {name}"
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = (640 - text_size[0]) // 2
            text_y = (480 + text_size[1]) // 2
            
            cv2.putText(img, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
            cv2.putText(img, f"Shirt: {shirt_color}", (text_x, text_y + 40), font, 0.7, (255, 255, 255), 2)
            
            # Save image
            cv2.imwrite(str(image_path), img)
            print(f"  Created: {image_path.name}")
            
        except Exception as e:
            print(f"  Error creating image for {json_file.name}: {e}")
    
    print(f"\nPlaceholder images saved to: {PROFILE_IMAGES_DIR}")

def generate_test_instructions():
    """Generate instructions for collecting real profile images."""
    print("\n" + "="*60)
    print("INSTRUCTIONS FOR COLLECTING REAL PROFILE IMAGES")
    print("="*60)
    
    print("\nFor proper face recognition testing, you need:")
    print("1. Clear face photos of each person")
    print("2. Multiple angles/expressions for better accuracy")
    print("3. Good lighting conditions")
    print("4. Consistent image format (JPEG recommended)")
    
    print("\n" + "-"*40)
    print("STEP 1: Collect Face Images")
    print("-"*40)
    print("For each person in known_faces/*.json:")
    print("  • Take 3-5 clear photos of their face")
    print("  • Include different angles (front, left, right)")
    print("  • Use good lighting")
    print("  • Save as: services/vision/known_faces/{name}.jpg")
    
    print("\n" + "-"*40)
    print("STEP 2: Update Test Scripts")
    print("-"*40)
    print("After collecting images, update test scripts to use them:")
    print("""
# In test_face_algorithms.py, update load_known_faces():
def load_known_faces(self, faces_dir: Path) -> None:
    for profile_file in sorted(faces_dir.glob("*.json")):
        profile = json.loads(profile_file.read_text())
        name = profile.get("name", profile_file.stem)
        
        # Look for corresponding image
        image_file = profile_file.with_suffix(".jpg")
        if image_file.exists():
            # Encode actual face image
            frame = cv2.imread(str(image_file))
            embedding = self.encode_face(frame)
            if embedding is not None:
                self.known_embeddings[name] = embedding
                self.known_names.append(name)
""")
    
    print("\n" + "-"*40)
    print("STEP 3: Run Tests")
    print("-"*40)
    print("After adding real images, run:")
    print("  python tests/vision/real_benchmark.py")
    print("  python run_face_benchmark.py --algorithm all")

def update_benchmark_for_real_images():
    """Show how to update benchmark to use real images."""
    print("\n" + "="*60)
    print("UPDATING BENCHMARK FOR REAL IMAGES")
    print("="*60)
    
    print("\nCurrent benchmark uses random embeddings.")
    print("To use real face images, update the load_known_faces method:")
    
    update_code = '''
# OLD: Using random embeddings
def load_known_faces(self, faces_dir: Path) -> None:
    for profile_file in sorted(faces_dir.glob("*.json")):
        profile = json.loads(profile_file.read_text())
        name = profile.get("name", profile_file.stem)
        self.known_names.append(name)
        # Random embedding (not realistic)
        self.known_embeddings[name] = np.random.randn(512).astype(np.float32)

# NEW: Using actual face images
def load_known_faces(self, faces_dir: Path) -> None:
    for profile_file in sorted(faces_dir.glob("*.json")):
        profile = json.loads(profile_file.read_text())
        name = profile.get("name", profile_file.stem)
        
        # Look for face image
        image_file = profile_file.with_suffix(".jpg")
        if not image_file.exists():
            # Try other extensions
            for ext in [".jpeg", ".png"]:
                alt_file = profile_file.with_suffix(ext)
                if alt_file.exists():
                    image_file = alt_file
                    break
        
        if image_file.exists():
            # Load and encode actual face
            frame = cv2.imread(str(image_file))
            if frame is not None:
                embedding = self.encode_face(frame)
                if embedding is not None:
                    self.known_embeddings[name] = embedding
                    self.known_names.append(name)
                    print(f"Loaded real face: {name}")
                else:
                    print(f"Failed to encode face: {name}")
            else:
                print(f"Failed to load image: {image_file}")
        else:
            print(f"No image found for: {name}")
            # Fall back to random embedding for testing
            self.known_names.append(name)
            self.known_embeddings[name] = np.random.randn(512).astype(np.float32)
'''
    
    print(update_code)

def main():
    """Main function."""
    print("="*80)
    print("PROFILE IMAGES COLLECTION TOOL")
    print("="*80)
    print("\nThis tool helps you collect profile images for face recognition testing.")
    
    while True:
        print("\n" + "="*40)
        print("MAIN MENU")
        print("="*40)
        print("1. Check existing images")
        print("2. Create placeholder images (for testing)")
        print("3. Show instructions for collecting real images")
        print("4. Show code updates for real images")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            check_existing_images()
        elif choice == "2":
            create_placeholder_images()
        elif choice == "3":
            generate_test_instructions()
        elif choice == "4":
            update_benchmark_for_real_images()
        elif choice == "5":
            print("\nExiting.")
            break
        else:
            print("\nInvalid choice. Please try again.")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nNext steps:")
    print("1. Collect actual face images for each profile")
    print("2. Save them as services/vision/known_faces/{name}.jpg")
    print("3. Update benchmark scripts to use real images")
    print("4. Run tests with: python run_face_benchmark.py")
    print("\nWithout real face images, benchmarks will use random embeddings")
    print("which doesn't reflect real-world accuracy.")
    print("="*80)

if __name__ == "__main__":
    main()
