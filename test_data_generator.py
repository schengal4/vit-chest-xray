"""
Test data generator for Health Universe FastAPI application.
Creates realistic test images and scenarios for comprehensive testing.
"""

import io
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
import json
from typing import Dict, List, Tuple, Optional
import tempfile
import os

class ChestXRayTestDataGenerator:
    """Generate realistic test data for chest X-ray classification testing."""
    
    def __init__(self):
        self.image_size = (224, 224)
        self.test_cases = self._define_test_cases()
    
    def _define_test_cases(self) -> List[Dict]:
        """Define comprehensive test cases for validation."""
        return [
            {
                "name": "normal_chest_xray",
                "description": "Normal chest X-ray simulation",
                "expected_class": "No Finding",
                "image_params": {"lung_opacity": 0.3, "heart_size": 0.4, "noise_level": 0.1}
            },
            {
                "name": "cardiomegaly_simulation",
                "description": "Enlarged heart simulation",
                "expected_class": "Cardiomegaly",
                "image_params": {"lung_opacity": 0.3, "heart_size": 0.7, "noise_level": 0.1}
            },
            {
                "name": "edema_pattern",
                "description": "Pulmonary edema pattern",
                "expected_class": "Edema",
                "image_params": {"lung_opacity": 0.6, "heart_size": 0.4, "noise_level": 0.2}
            },
            {
                "name": "consolidation_area",
                "description": "Lung consolidation pattern",
                "expected_class": "Consolidation",
                "image_params": {"lung_opacity": 0.8, "heart_size": 0.4, "noise_level": 0.15}
            },
            {
                "name": "pneumonia_infiltrate",
                "description": "Pneumonia infiltrate pattern",
                "expected_class": "Pneumonia",
                "image_params": {"lung_opacity": 0.7, "heart_size": 0.4, "noise_level": 0.25}
            }
        ]
    
    def create_synthetic_chest_xray(self, 
                                   lung_opacity: float = 0.3,
                                   heart_size: float = 0.4,
                                   noise_level: float = 0.1) -> Image.Image:
        """
        Create a synthetic chest X-ray image for testing.
        
        Args:
            lung_opacity: Opacity of lung fields (0.0-1.0)
            heart_size: Relative size of heart (0.0-1.0)
            noise_level: Amount of noise to add (0.0-1.0)
        
        Returns:
            PIL Image object
        """
        # Create base image (dark background like X-ray)
        img = Image.new('RGB', self.image_size, color=(20, 20, 20))
        draw = ImageDraw.Draw(img)
        
        width, height = self.image_size
        
        # Draw ribcage outline
        self._draw_ribcage(draw, width, height)
        
        # Draw lung fields
        self._draw_lungs(draw, width, height, opacity=lung_opacity)
        
        # Draw heart
        self._draw_heart(draw, width, height, size=heart_size)
        
        # Add spine
        self._draw_spine(draw, width, height)
        
        # Add noise
        if noise_level > 0:
            img = self._add_noise(img, noise_level)
        
        # Apply X-ray-like filters
        img = self._apply_xray_filters(img)
        
        return img
    
    def _draw_ribcage(self, draw: ImageDraw.Draw, width: int, height: int):
        """Draw realistic ribcage structure."""
        # Draw ribs as arched lines
        for i in range(6):
            y_pos = height * 0.2 + i * (height * 0.1)
            # Left ribs
            draw.arc([width*0.1, y_pos, width*0.5, y_pos + height*0.15], 
                    start=0, end=180, fill=(80, 80, 80), width=2)
            # Right ribs
            draw.arc([width*0.5, y_pos, width*0.9, y_pos + height*0.15], 
                    start=0, end=180, fill=(80, 80, 80), width=2)
    
    def _draw_lungs(self, draw: ImageDraw.Draw, width: int, height: int, opacity: float):
        """Draw lung fields with variable opacity."""
        lung_color = int(40 + opacity * 100)
        
        # Left lung
        left_lung_points = [
            (width*0.1, height*0.2),
            (width*0.45, height*0.15),
            (width*0.45, height*0.8),
            (width*0.15, height*0.85)
        ]
        draw.polygon(left_lung_points, fill=(lung_color, lung_color, lung_color))
        
        # Right lung
        right_lung_points = [
            (width*0.55, height*0.15),
            (width*0.9, height*0.2),
            (width*0.85, height*0.85),
            (width*0.55, height*0.8)
        ]
        draw.polygon(right_lung_points, fill=(lung_color, lung_color, lung_color))
    
    def _draw_heart(self, draw: ImageDraw.Draw, width: int, height: int, size: float):
        """Draw heart silhouette with variable size."""
        heart_size = size * 0.3  # Scale factor
        
        # Heart shape (simplified)
        heart_points = [
            (width*0.4, height*0.3),
            (width*(0.4 + heart_size), height*0.25),
            (width*(0.6 + heart_size), height*0.35),
            (width*(0.6 + heart_size), height*0.65),
            (width*0.5, height*0.75),
            (width*(0.4 - heart_size*0.3), height*0.65),
            (width*(0.4 - heart_size*0.3), height*0.35)
        ]
        
        draw.polygon(heart_points, fill=(100, 100, 100))
    
    def _draw_spine(self, draw: ImageDraw.Draw, width: int, height: int):
        """Draw spine structure."""
        spine_x = width // 2
        draw.line([(spine_x, height*0.1), (spine_x, height*0.9)], 
                 fill=(120, 120, 120), width=3)
        
        # Vertebrae
        for i in range(8):
            y_pos = height*0.2 + i * (height*0.08)
            draw.ellipse([spine_x-5, y_pos-3, spine_x+5, y_pos+3], 
                        fill=(130, 130, 130))
    
    def _add_noise(self, img: Image.Image, noise_level: float) -> Image.Image:
        """Add realistic noise to the image."""
        # Convert to numpy array
        img_array = np.array(img)
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level * 50, img_array.shape)
        noisy_img = img_array + noise
        
        # Clip values to valid range
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_img)
    
    def _apply_xray_filters(self, img: Image.Image) -> Image.Image:
        """Apply filters to make image look more like an X-ray."""
        # Convert to grayscale and back to RGB
        gray = img.convert('L')
        img = gray.convert('RGB')
        
        # Apply slight blur (X-ray effect)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Enhance contrast
        img = img.point(lambda x: x * 1.2 if x < 128 else x * 0.8)
        
        return img
    
    def generate_test_image(self, test_case: str = "normal_chest_xray", 
                           format: str = "JPEG") -> io.BytesIO:
        """
        Generate a test image for a specific test case.
        
        Args:
            test_case: Name of the test case
            format: Image format (JPEG, PNG, etc.)
        
        Returns:
            BytesIO buffer containing the image
        """
        # Find test case parameters
        case_params = None
        for case in self.test_cases:
            if case["name"] == test_case:
                case_params = case
                break
        
        if not case_params:
            case_params = self.test_cases[0]  # Default to normal
        
        # Generate image
        img = self.create_synthetic_chest_xray(**case_params["image_params"])
        
        # Save to buffer
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        buffer.seek(0)
        
        return buffer
    
    def generate_all_test_cases(self) -> Dict[str, io.BytesIO]:
        """Generate images for all test cases."""
        test_images = {}
        
        for case in self.test_cases:
            test_images[case["name"]] = self.generate_test_image(case["name"])
        
        return test_images
    
    def create_invalid_test_files(self) -> Dict[str, io.BytesIO]:
        """Create invalid files for security testing."""
        invalid_files = {}
        
        # Text file disguised as image
        text_file = io.BytesIO(b"This is not an image file")
        invalid_files["fake_image.jpg"] = text_file
        
        # Binary garbage file
        garbage_file = io.BytesIO(os.urandom(1024))
        invalid_files["garbage.png"] = garbage_file
        
        # Empty file
        empty_file = io.BytesIO(b"")
        invalid_files["empty.jpeg"] = empty_file
        
        # Extremely large file (simulated)
        large_file_content = b"x" * (11 * 1024 * 1024)  # 11MB
        large_file = io.BytesIO(large_file_content)
        invalid_files["large_file.jpg"] = large_file
        
        return invalid_files
    
    def create_edge_case_images(self) -> Dict[str, io.BytesIO]:
        """Create edge case images for robustness testing."""
        edge_cases = {}
        
        # Very small image
        small_img = Image.new('RGB', (1, 1), color=(0, 0, 0))
        buffer = io.BytesIO()
        small_img.save(buffer, format="JPEG")
        buffer.seek(0)
        edge_cases["tiny_image.jpg"] = buffer
        
        # Very large image
        large_img = Image.new('RGB', (2048, 2048), color=(128, 128, 128))
        buffer = io.BytesIO()
        large_img.save(buffer, format="JPEG")
        buffer.seek(0)
        edge_cases["large_image.jpg"] = buffer
        
        # Corrupted image (truncated)
        normal_buffer = self.generate_test_image()
        normal_data = normal_buffer.read()
        truncated_data = normal_data[:len(normal_data)//2]  # Cut in half
        corrupted_buffer = io.BytesIO(truncated_data)
        edge_cases["corrupted_image.jpg"] = corrupted_buffer
        
        # Different color modes
        for mode in ['L', 'P', 'RGBA']:
            img = Image.new(mode, (224, 224), color=128)
            buffer = io.BytesIO()
            # Convert to RGB for JPEG compatibility
            if mode != 'RGB':
                img = img.convert('RGB')
            img.save(buffer, format="JPEG")
            buffer.seek(0)
            edge_cases[f"mode_{mode}.jpg"] = buffer
        
        return edge_cases
    
    def save_test_dataset(self, output_dir: str = "test_images"):
        """Save a complete test dataset to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save normal test cases
        for case in self.test_cases:
            img_buffer = self.generate_test_image(case["name"])
            file_path = os.path.join(output_dir, f"{case['name']}.jpg")
            
            with open(file_path, 'wb') as f:
                f.write(img_buffer.read())
            
            print(f"Saved: {file_path}")
        
        # Save edge cases
        edge_cases = self.create_edge_case_images()
        edge_dir = os.path.join(output_dir, "edge_cases")
        os.makedirs(edge_dir, exist_ok=True)
        
        for name, buffer in edge_cases.items():
            file_path = os.path.join(edge_dir, name)
            with open(file_path, 'wb') as f:
                f.write(buffer.read())
            print(f"Saved edge case: {file_path}")
        
        # Save test metadata
        metadata = {
            "test_cases": self.test_cases,
            "image_size": self.image_size,
            "description": "Synthetic chest X-ray test dataset for Health Universe FastAPI testing"
        }
        
        metadata_path = os.path.join(output_dir, "test_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved metadata: {metadata_path}")
        print(f"Test dataset created in: {output_dir}")


class PerformanceTestDataGenerator:
    """Generate data for performance and load testing."""
    
    @staticmethod
    def generate_concurrent_test_images(count: int = 10) -> List[io.BytesIO]:
        """Generate multiple test images for concurrent testing."""
        generator = ChestXRayTestDataGenerator()
        images = []
        
        for i in range(count):
            # Vary the test case
            case_name = generator.test_cases[i % len(generator.test_cases)]["name"]
            img_buffer = generator.generate_test_image(case_name)
            images.append(img_buffer)
        
        return images
    
    @staticmethod
    def generate_load_test_scenario() -> Dict:
        """Generate a complete load test scenario."""
        return {
            "concurrent_users": [1, 5, 10, 20],
            "requests_per_user": [10, 20, 50],
            "test_duration_seconds": [30, 60, 120],
            "image_sizes": [(224, 224), (512, 512), (1024, 1024)],
            "file_formats": ["JPEG", "PNG"]
        }


def main():
    """Generate test dataset for manual inspection."""
    print("🧪 Generating Health Universe Test Dataset")
    print("=" * 50)
    
    generator = ChestXRayTestDataGenerator()
    
    # Generate and save test dataset
    generator.save_test_dataset()
    
    print("\n✅ Test dataset generation complete!")
    print("\n📋 Generated test cases:")
    for case in generator.test_cases:
        print(f"   - {case['name']}: {case['description']}")
    
    print("\n🔍 Use these images to manually verify the prediction endpoint")
    print("📁 Check the 'test_images' directory for generated files")


if __name__ == "__main__":
    main()