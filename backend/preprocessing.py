
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from pathlib import Path
import logging
from typing import Union, Optional
from tensorflow import keras
from keras.utils import Sequence
from keras.layers import *
from keras.models import Model

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

# ═══════════════════════════════════════════════════════════
# GLAND SEGMENTATION
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
# GLAND SEGMENTATION
# ═══════════════════════════════════════════════════════════

def load_segmentation_model(model_path: str):
    """
    Load the U-Net gland segmentation model.
    
    Args:
        model_path: Path to the saved H5 model file
        
    Returns:
        Loaded Keras model
    """
    
    try:
        # Try loading the complete model first
        model = keras.models.load_model(model_path, compile=False)
        logger.info(f"✓ Segmentation model loaded from {model_path}")
        return model
        
    except Exception as e:
        logger.warning(f"Direct load failed: {e}")
        logger.info("Rebuilding U-Net architecture and loading weights...")
        
        try:
            # Build the EXACT architecture from your training code
            def conv_block(inputs, num_filters, dropout_prob=0.2):
                """Convolutional block with two conv layers"""
                conv = Conv2D(num_filters, 3, activation='relu', padding='same',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
                conv = BatchNormalization()(conv)
                conv = Dropout(dropout_prob)(conv)
                
                conv = Conv2D(num_filters, 3, activation='relu', padding='same',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(conv)
                conv = BatchNormalization()(conv)
                
                return conv

            def encoder_block(inputs, num_filters, dropout_prob=0.2):
                """Encoder block with conv block and max pooling"""
                conv = conv_block(inputs, num_filters, dropout_prob)
                pool = MaxPooling2D(pool_size=(2, 2))(conv)
                return conv, pool

            def decoder_block(inputs, skip_features, num_filters, dropout_prob=0.2):
                """Decoder block with upsampling and concatenation"""
                upsample = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
                concat = Concatenate()([upsample, skip_features])
                conv = conv_block(concat, num_filters, dropout_prob)
                return conv

            # Build U-Net with YOUR exact architecture
            input_shape = (256, 256, 3)
            inputs = Input(shape=input_shape)

            # Encoder - YOUR EXACT FILTERS
            s1, p1 = encoder_block(inputs, 32, 0.2)
            s2, p2 = encoder_block(p1, 64, 0.2)
            s3, p3 = encoder_block(p2, 128, 0.3)
            s4, p4 = encoder_block(p3, 256, 0.3)

            # Bottleneck
            bottleneck = conv_block(p4, 512, 0.4)

            # Decoder
            d1 = decoder_block(bottleneck, s4, 256, 0.3)
            d2 = decoder_block(d1, s3, 128, 0.3)
            d3 = decoder_block(d2, s2, 64, 0.2)
            d4 = decoder_block(d3, s1, 32, 0.2)

            # Output
            outputs = Conv2D(1, 1, activation='sigmoid', padding='same')(d4)

            model = Model(inputs=inputs, outputs=outputs, name='U-Net')
            
            # Load weights from the H5 file
            model.load_weights(model_path)
            
            logger.info(f"✓ Segmentation model loaded (weights only) from {model_path}")
            return model
            
        except Exception as e2:
            logger.error(f"Failed to load model or weights: {e2}")
            raise Exception(f"Could not load segmentation model: {e2}")


def segment_glands(
    image: Union[str, Path, Image.Image, np.ndarray],
    model,
    overlay_alpha: float = 0.4
) -> Image.Image:
    """
    Segment glands in histopathology image and create overlay.
    
    Args:
        image: Input image (path, PIL Image, or numpy array)
        model: Loaded U-Net segmentation model
        overlay_alpha: Transparency of segmentation overlay (0-1)
        
    Returns:
        PIL Image with segmentation overlay
    """
    # Load and prepare image
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        img = image.convert('RGB')
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert('RGB')
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Resize to model input size (256x256 - YOUR training size)
    img_resized = img.resize((256, 256))
    img_array = np.array(img_resized) / 255.0  # Normalize to [0, 1]
    
    # Add batch dimension: (256, 256, 3) -> (1, 256, 256, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Run segmentation prediction
    pred_mask = model.predict(img_batch, verbose=0)[0]
    
    # Threshold prediction (binary segmentation at 0.5 - same as training)
    pred_mask_binary = (pred_mask > 0.5).astype(np.uint8) * 255
    
    # Handle output shape
    if len(pred_mask_binary.shape) == 3 and pred_mask_binary.shape[-1] == 1:
        pred_mask_binary = pred_mask_binary[:, :, 0]
    
    # Resize back to 224x224 for display consistency with other images
    pred_mask_binary = cv2.resize(pred_mask_binary, (224, 224), interpolation=cv2.INTER_NEAREST)
    img_array_224 = cv2.resize(img_array, (224, 224))
    
    # Create colored overlay (green for glands)
    overlay = np.zeros((224, 224, 3), dtype=np.uint8)
    overlay[pred_mask_binary > 127] = [0, 255, 0]  # Green for detected glands
    
    # Blend with original image
    img_array_uint8 = (img_array_224 * 255).astype(np.uint8)
    blended = cv2.addWeighted(img_array_uint8, 1 - overlay_alpha, overlay, overlay_alpha, 0)
    
    return Image.fromarray(blended)