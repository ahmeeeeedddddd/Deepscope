# backend/preprocessing.py

"""
Image Preprocessing for Colorectal Cancer Diagnosis System

This module handles preprocessing of histopathology images for inference with
stain normalization to address cross-institutional generalization challenges.

CRITICAL DESIGN DECISION:
- Training data (NCT-CRC-HE-100K) is already pre-normalized from a single institution
- User-uploaded images will come from DIFFERENT hospitals/labs with varying stain protocols
- Stain normalization at inference ensures uploaded images match training distribution
- This addresses the major barrier to clinical adoption: cross-institutional generalization

Reference Issue:
"AI models trained on data from one institution often show dramatic performance 
degradation when deployed elsewhere" - This preprocessing pipeline solves that.
"""

import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from pathlib import Path
import logging
from typing import Union, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# STAIN NORMALIZATION (ADDRESSES CROSS-INSTITUTIONAL VARIATION)
# ═══════════════════════════════════════════════════════════

class SimpleStainNormalizer:
    """
    Simplified stain normalization for H&E histopathology images.
    
    Purpose: Normalize stain variations across different laboratories/institutions
    to match the distribution of the training dataset (NCT-CRC-HE-100K).
    
    Method: LAB color space histogram matching and color transfer
    """
    
    def __init__(self, target_image_path: Optional[str] = None):
        """
        Initialize normalizer.
        
        Args:
            target_image_path: Path to reference image from training set
        """
        self.target_stats = None
        if target_image_path:
            self.fit(Image.open(target_image_path))
    
    def fit(self, target_image: Union[Image.Image, np.ndarray]):
        """
        Compute target statistics from reference image.
        
        Args:
            target_image: Reference image (PIL Image or numpy array)
        """
        if isinstance(target_image, Image.Image):
            target_image = np.array(target_image)
        
        # Convert to LAB color space
        target_lab = cv2.cvtColor(target_image, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Store mean and std for each channel
        self.target_stats = {
            'mean': target_lab.mean(axis=(0, 1)),
            'std': target_lab.std(axis=(0, 1))
        }
        
        logger.info("✓ Stain normalizer fitted with reference image")
    
    def __call__(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Apply stain normalization to an image.
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            Normalized PIL Image
        """
        if self.target_stats is None:
            logger.warning("Normalizer not fitted - returning original image")
            return image if isinstance(image, Image.Image) else Image.fromarray(image)
        
        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
        
        try:
            # Convert to LAB
            img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
            
            # Get source statistics
            src_mean = img_lab.mean(axis=(0, 1))
            src_std = img_lab.std(axis=(0, 1))
            
            # Normalize to zero mean and unit variance
            img_lab = (img_lab - src_mean) / (src_std + 1e-6)
            
            # Apply target statistics
            img_lab = img_lab * self.target_stats['std'] + self.target_stats['mean']
            
            # Clip and convert back to RGB
            img_lab = np.clip(img_lab, 0, 255).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
            
            return Image.fromarray(img_rgb)
        
        except Exception as e:
            logger.error(f"Stain normalization failed: {e}")
            logger.warning("Returning original image")
            return image if isinstance(image, Image.Image) else Image.fromarray(image)


# ═══════════════════════════════════════════════════════════
# MAIN PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════

class InferencePreprocessor:
    """
    Complete preprocessing pipeline for inference.
    
    Pipeline:
    1. Load image
    2. Stain normalization (critical for cross-institutional generalization)
    3. Resize to 224x224
    4. Convert to tensor
    5. Normalize with ImageNet stats (same as training)
    """
    
    def __init__(
        self,
        image_size: int = 224,
        mean: list = [0.485, 0.456, 0.406],  # ImageNet stats
        std: list = [0.229, 0.224, 0.225],
        use_stain_norm: bool = True,
        reference_image_path: Optional[str] = None
    ):
        """
        Initialize preprocessing pipeline.
        
        Args:
            image_size: Target image size (default: 224)
            mean: Normalization mean (ImageNet stats)
            std: Normalization std (ImageNet stats)
            use_stain_norm: Whether to apply stain normalization
            reference_image_path: Path to reference image from training set
        """
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.use_stain_norm = use_stain_norm
        
        # Initialize stain normalizer
        if use_stain_norm:
            self.stain_normalizer = SimpleStainNormalizer(reference_image_path)
            logger.info("Stain normalization: ENABLED")
        else:
            self.stain_normalizer = None
            logger.info("Stain normalization: DISABLED")
        
        # Create PyTorch transforms (same as validation in training)
        self.torch_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def fit_normalizer(self, reference_image: Union[str, Image.Image, np.ndarray]):
        """
        Fit stain normalizer with a reference image from the training set.
        
        IMPORTANT: This should be called ONCE during model initialization
        with a representative image from NCT-CRC-HE-100K dataset.
        
        Args:
            reference_image: Path to reference image, PIL Image, or numpy array
        """
        if not self.use_stain_norm or self.stain_normalizer is None:
            logger.warning("Stain normalization is disabled")
            return
        
        if isinstance(reference_image, str):
            reference_image = Image.open(reference_image).convert('RGB')
        
        self.stain_normalizer.fit(reference_image)
        logger.info("✓ Stain normalizer fitted with reference image")
    
    def __call__(
        self, 
        image_path: Union[str, Path, Image.Image, np.ndarray]
    ) -> torch.Tensor:
        """
        Preprocess an image for inference.
        
        Args:
            image_path: Path to image file, PIL Image, or numpy array
            
        Returns:
            Preprocessed tensor ready for model inference (shape: [1, 3, 224, 224])
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert('RGB')
            logger.info(f"Loaded image: {Path(image_path).name}")
        elif isinstance(image_path, Image.Image):
            image = image_path.convert('RGB')
        elif isinstance(image_path, np.ndarray):
            image = Image.fromarray(image_path).convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image_path)}")
        
        # Apply stain normalization (CRITICAL for cross-institutional generalization)
        if self.use_stain_norm and self.stain_normalizer is not None:
            image = self.stain_normalizer(image)
            logger.debug("Applied stain normalization")
        
        # Apply PyTorch transforms
        tensor = self.torch_transforms(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)  # [3, 224, 224] → [1, 3, 224, 224]
        
        return tensor


# ═══════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION FOR FLASK APP
# ═══════════════════════════════════════════════════════════

# Global preprocessor instance (will be initialized in model_handler.py)
_preprocessor = None


def initialize_preprocessor(
    reference_image_path: Optional[str] = None,
    use_stain_norm: bool = True
) -> InferencePreprocessor:
    """
    Initialize the global preprocessor instance.
    
    CALL THIS ONCE when the Flask app starts (in model_handler.py).
    
    Args:
        reference_image_path: Path to a representative image from NCT-CRC-HE-100K
        use_stain_norm: Enable/disable stain normalization
        
    Returns:
        Initialized preprocessor instance
    """
    global _preprocessor
    
    _preprocessor = InferencePreprocessor(
        image_size=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        use_stain_norm=use_stain_norm,
        reference_image_path=reference_image_path
    )
    
    logger.info("✓ Preprocessor initialized")
    
    return _preprocessor


def preprocess_image(image_path: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
    """
    Preprocess an image for inference (convenience function for Flask routes).
    
    Args:
        image_path: Path to image file, PIL Image, or numpy array
        
    Returns:
        Preprocessed tensor [1, 3, 224, 224]
        
    Raises:
        RuntimeError: If preprocessor not initialized
    """
    global _preprocessor
    
    if _preprocessor is None:
        raise RuntimeError(
            "Preprocessor not initialized! "
            "Call initialize_preprocessor() first in model_handler.py"
        )
    
    return _preprocessor(image_path)