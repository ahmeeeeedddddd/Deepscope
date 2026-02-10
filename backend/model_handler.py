# backend/model_handler.py

"""
Model Handler for Colorectal Cancer Diagnosis System

This module handles:
1. Loading the trained Lunit DINO ViT model
2. Running inference on preprocessed images
3. Post-processing predictions (softmax, class mapping, confidence)
4. Generating Grad-CAM visualizations for explainability
5. Returning structured results to Flask API

Model: Lunit DINO (ViT-Small/8) fine-tuned on NCT-CRC-HE-100K
Task: 9-class tissue classification
Classes: ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple
import logging
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime

# Import preprocessing
from preprocessing import initialize_preprocessor, preprocess_image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# CUSTOM CLASSIFICATION HEAD
# ═══════════════════════════════════════════════════════════

class ViTClassificationHead(nn.Module):
    """
    Custom classification head for Lunit DINO ViT.
    
    Architecture (matches training):
    - BatchNorm1d for normalization
    - Dropout for regularization
    - Linear layer to num_classes
    """
    
    def __init__(
        self, 
        num_features: int, 
        num_classes: int, 
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Initialize classification head.
        
        Args:
            num_features: Feature dimension from ViT backbone (384 for ViT-Small)
            num_classes: Number of output classes (9 for multiclass)
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        layers = []
        
        # Optional batch normalization
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(num_features))
        
        # Dropout for regularization
        layers.append(nn.Dropout(dropout_rate))
        
        # Final classification layer
        layers.append(nn.Linear(num_features, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize linear layer weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through classification head."""
        return self.classifier(x)


class LunitDINOClassifier(nn.Module):
    """
    Complete model: Lunit DINO backbone + Classification head
    Matches the ColorectalCancerViT architecture from training.
    """
    
    def __init__(
        self,
        model_name: str = "hf_hub:1aurent/vit_small_patch8_224.lunit_dino",
        num_classes: int = 9,
        num_features: int = 384,
        dropout_rate: float = 0.1
    ):
        """
        Initialize the complete model.
        
        Args:
            model_name: Pretrained model identifier
            num_classes: Number of output classes
            num_features: Feature dimension from backbone
            dropout_rate: Dropout rate for classification head
        """
        super().__init__()
        
        # Load pretrained Lunit DINO backbone (without classification head)
        # Enable attention output for explainability
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0  # Remove default head
        )

        # Enable attention output in all blocks
        for block in self.backbone.blocks:
            if hasattr(block, 'attn'):
                # Force attention to return attention weights
                block.attn.fused_attn = False
        
        # Add custom classification head (matches training architecture)
        self.head = ViTClassificationHead(
            num_features=num_features,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_batch_norm=True  # Match training config
        )
        
        # Store for Grad-CAM
        self.features = None
        self.gradients = None
        
        logger.info(f"✓ Initialized LunitDINOClassifier")
        logger.info(f"  Backbone: {model_name}")
        logger.info(f"  Num classes: {num_classes}")
        logger.info(f"  Feature dim: {num_features}")
    
    def forward(self, x):
        """Forward pass through backbone and head."""
        features = self.backbone(x)  # [B, 384]
        logits = self.head(features)  # [B, num_classes]
        return logits
    
    def forward_with_features(self, x):
        """
        Forward pass that also returns intermediate features for Grad-CAM.
        
        Returns:
            logits: Model predictions
            features: Features before classification head
        """
        features = self.backbone(x)  # [B, 384]
        self.features = features  # Store for Grad-CAM
        logits = self.head(features)  # [B, num_classes]
        return logits, features


# ═══════════════════════════════════════════════════════════
# GRAD-CAM IMPLEMENTATION FOR VISION TRANSFORMER
# ═══════════════════════════════════════════════════════════

class ViTAttentionRollout:
    """
    Attention Rollout for Vision Transformers.
    
    This extracts native attention weights from ViT layers,
    which is more appropriate than Grad-CAM for transformer architectures.
    """
    
    def __init__(self, model: nn.Module, head_fusion: str = "mean", discard_ratio: float = 0.9):
        """
        Initialize Attention Rollout.
        
        Args:
            model: The ViT model
            head_fusion: How to combine attention heads ('mean', 'max', 'min')
            discard_ratio: Ratio of lowest attention values to discard
        """
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.model.eval()
    
    def get_attention_maps(self, input_tensor: torch.Tensor) -> list:
        """
        Extract attention maps from all transformer blocks.
        
        Args:
            input_tensor: Input image tensor [1, 3, 224, 224]
            
        Returns:
            List of attention matrices from each layer
        """
        attention_maps = []
        
        def hook_fn(module, input, output):
            # Store attention weights
            # For timm ViT, we need to access the attention directly
            if hasattr(module, 'get_attention_map'):
                attn = module.get_attention_map()
                if attn is not None:
                    attention_maps.append(attn)
        
        # Custom hook to capture attention during forward pass
        def attention_hook(module, input, output):
            # Input to attention: [B, N, C]
            # We need to manually compute attention to capture it
            B, N, C = input[0].shape
            
            # Get qkv from the module
            qkv = module.qkv(input[0]).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Compute attention
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
            
            attention_maps.append(attn.detach())
        
        # Register hooks on attention modules
        hooks = []
        for block in self.model.backbone.blocks:
            if hasattr(block, 'attn'):
                hook = block.attn.register_forward_hook(attention_hook)
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_maps
    
    def generate_rollout(
        self,
        input_tensor: torch.Tensor,
        target_class: int = None,
        start_layer: int = 0
    ) -> np.ndarray:
        """
        Generate attention rollout heatmap.
        
        Args:
            input_tensor: Input image tensor [1, 3, 224, 224]
            target_class: Not used for attention rollout (kept for API compatibility)
            start_layer: Layer to start rollout from (default: 0)
            
        Returns:
            Heatmap as numpy array [224, 224]
        """
        
        # Get attention maps from all layers
        attention_maps = self.get_attention_maps(input_tensor)
        
        if len(attention_maps) == 0:
            raise ValueError("No attention maps extracted. Check model architecture.")
        
        # Process attention maps
        result = None
        
        for i, attn in enumerate(attention_maps[start_layer:]):
            # attn shape: [batch, num_heads, num_patches+1, num_patches+1]
            attn = attn.squeeze(0)  # Remove batch dimension: [num_heads, seq_len, seq_len]
            
            # Fuse attention heads
            if self.head_fusion == "mean":
                attn = attn.mean(dim=0)  # [seq_len, seq_len]
            elif self.head_fusion == "max":
                attn = attn.max(dim=0)[0]
            elif self.head_fusion == "min":
                attn = attn.min(dim=0)[0]
            
            # Discard lowest attention values
            flat = attn.view(-1)
            _, indices = flat.topk(int(flat.size(0) * self.discard_ratio))
            flat[indices] = 0
            
            # Add identity matrix (residual connection)
            I = torch.eye(attn.size(0), device=attn.device)
            attn = (attn + I) / 2
            
            # Normalize
            attn = attn / attn.sum(dim=-1, keepdim=True)
            
            # Accumulate attention
            if result is None:
                result = attn
            else:
                result = torch.matmul(attn, result)
        
        # Get attention for CLS token to all patches
        mask = result[0, 1:]  # [num_patches] - CLS attention to all patches
        
        # Reshape to spatial dimensions
        patch_size = 8
        num_patches_side = 224 // patch_size  # 28
        mask = mask.reshape(num_patches_side, num_patches_side)
        
        # Convert to numpy and normalize
        mask = mask.cpu().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        
        # Resize to original image size
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        return mask

# ═══════════════════════════════════════════════════════════
# MODEL HANDLER CLASS
# ═══════════════════════════════════════════════════════════

class ModelHandler:
    """
    Handles model loading, preprocessing, inference, post-processing, and Grad-CAM.
    
    This is the main interface between Flask API and the ML model.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = 'cuda',
        reference_image_path: Optional[str] = None,
        use_stain_norm: bool = True
    ):
        """
        Initialize model handler.
        
        Args:
            model_path: Path to saved model checkpoint (.pth file)
            config_path: Path to classification config JSON
            device: 'cuda' or 'cpu'
            reference_image_path: Path to reference image for stain normalization
            use_stain_norm: Whether to use stain normalization
        """
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize model
        self.model = self._load_model(model_path)
        
        # Initialize Grad-CAM
        # Initialize Attention Rollout (better for ViT)
        self.gradcam = ViTAttentionRollout(self.model, head_fusion="mean", discard_ratio=0.7)
        
        # Initialize preprocessor
        self._initialize_preprocessor(reference_image_path, use_stain_norm)
        
        logger.info("✓ ModelHandler initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load classification configuration."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"✓ Loaded config from: {config_path}")
        logger.info(f"  Classification mode: {config['classification_mode']}")
        logger.info(f"  Number of classes: {config['num_classes']}")
        
        return config
    
    def _load_model(self, model_path: str) -> nn.Module:
        """
        Load the trained model from checkpoint.
        
        Args:
            model_path: Path to .pth checkpoint file
            
        Returns:
            Loaded model in evaluation mode
        """
        
        logger.info(f"Loading model from: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model architecture
        model = LunitDINOClassifier(
            model_name=self.config['model_name'],
            num_classes=self.config['num_classes'],
            num_features=self.config['feature_dim'],
            dropout_rate=self.config['dropout_rate']
        )
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()
        
        # Log checkpoint info
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        logger.info(f"  Val accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
        logger.info(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        
        return model
    
    def _initialize_preprocessor(
        self,
        reference_image_path: Optional[str],
        use_stain_norm: bool
    ):
        """
        Initialize the image preprocessor with stain normalization.
        
        Args:
            reference_image_path: Path to reference image from training set
            use_stain_norm: Whether to enable stain normalization
        """
        
        # Initialize preprocessor
        self.preprocessor = initialize_preprocessor(
            reference_image_path=reference_image_path,
            use_stain_norm=use_stain_norm
        )
        
        logger.info(f"✓ Preprocessor initialized")
        logger.info(f"  Stain normalization: {'ENABLED' if use_stain_norm else 'DISABLED'}")
        
        if use_stain_norm and reference_image_path:
            logger.info(f"  Reference image: {Path(reference_image_path).name}")
    
    def generate_gradcam_heatmap(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
        target_class: int = None,
        alpha: float = 0.5
    ) -> str:
        """
        Generate and save Grad-CAM heatmap overlay.
        
        Args:
            image_path: Path to input image
            output_path: Path to save heatmap overlay
            target_class: Target class for Grad-CAM (if None, uses predicted class)
            alpha: Transparency for overlay (0=only heatmap, 1=only image)
            
        Returns:
            Path to saved heatmap image
        """
        
        try:
            # Load and preprocess image
            input_tensor = preprocess_image(image_path)
            input_tensor = input_tensor.to(self.device)
            input_tensor.requires_grad = True
            
            # Generate Attention Rollout
            logger.info("Generating Attention Rollout heatmap...")
            cam = self.gradcam.generate_rollout(input_tensor, target_class)
            
            # Load original image for overlay
            original_image = Image.open(image_path).convert('RGB')
            original_image = original_image.resize((224, 224))
            original_np = np.array(original_image)
            
            # Create heatmap
            heatmap = cm.jet(cam)[:, :, :3]  # Remove alpha channel
            heatmap = (heatmap * 255).astype(np.uint8)
            
            # Create overlay
            overlay = (alpha * original_np + (1 - alpha) * heatmap).astype(np.uint8)
            
            # Save overlay
            overlay_img = Image.fromarray(overlay)
            overlay_img.save(output_path)
            
            logger.info(f"✓ Grad-CAM heatmap saved to: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to generate Grad-CAM: {str(e)}")
            raise
    
    def predict(
        self, 
        image_path: Union[str, Path],
        generate_heatmap: bool = False,
        heatmap_output_dir: str = None
    ) -> Dict[str, Any]:
        """
        Run inference on an image and return structured predictions.
        
        Args:
            image_path: Path to the image file
            generate_heatmap: Whether to generate Grad-CAM heatmap
            heatmap_output_dir: Directory to save heatmap (if None, uses same dir as image)
            
        Returns:
            Dictionary containing:
            {
                'diagnosis': str,              # Primary diagnosis
                'confidence': float,           # Confidence score (0-1)
                'tissue_type': str,           # Tissue class name
                'is_malignant': bool,         # Whether tissue is malignant
                'class_probabilities': dict,  # All class probabilities
                'top_3_predictions': list,    # Top 3 predictions with confidence
                'clinical_group': str,        # Clinical grouping
                'heatmap_path': str (optional) # Path to Grad-CAM heatmap if generated
            }
        """
        
        try:
            # Preprocess image
            input_tensor = preprocess_image(image_path)
            input_tensor = input_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)
            
            # Convert to numpy for easier processing
            probs = probabilities.cpu().numpy()[0]  # Remove batch dimension
            
            # Get prediction
            predicted_class_idx = int(np.argmax(probs))
            confidence = float(probs[predicted_class_idx])
            
            # Map to class name
            tissue_type = self.config['idx_to_label'][str(predicted_class_idx)]
            
            # Determine if malignant
            is_malignant = tissue_type in self.config['clinical_groups']['malignant']
            
            # Create diagnosis text
            diagnosis = self._create_diagnosis(tissue_type, confidence, is_malignant)
            
            # Get all class probabilities
            class_probs = {
                self.config['idx_to_label'][str(i)]: float(probs[i])
                for i in range(len(probs))
            }
            
            # Get top 3 predictions
            top_3_indices = np.argsort(probs)[-3:][::-1]
            top_3_predictions = [
                {
                    'class': self.config['idx_to_label'][str(i)],
                    'confidence': float(probs[i]),
                    'description': self.config['class_descriptions'][
                        self.config['idx_to_label'][str(i)]
                    ]
                }
                for i in top_3_indices
            ]
            
            # Determine clinical group
            clinical_group = self._get_clinical_group(tissue_type)
            
            # Compile results
            result = {
                'diagnosis': diagnosis,
                'confidence': confidence,
                'tissue_type': tissue_type,
                'tissue_description': self.config['class_descriptions'][tissue_type],
                'is_malignant': is_malignant,
                'class_probabilities': class_probs,
                'top_3_predictions': top_3_predictions,
                'clinical_group': clinical_group,
                'num_classes': self.config['num_classes']
            }
            
            # Generate Grad-CAM heatmap if requested
            if generate_heatmap:
                # Determine output directory
                if heatmap_output_dir is None:
                    heatmap_output_dir = Path(image_path).parent
                
                # Create heatmap filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                heatmap_filename = f"heatmap_{timestamp}.png"
                heatmap_path = Path(heatmap_output_dir) / heatmap_filename
                
                # Generate and save heatmap
                heatmap_path = self.generate_gradcam_heatmap(
                    image_path=image_path,
                    output_path=heatmap_path,
                    target_class=predicted_class_idx,
                    alpha=0.5  # 50% transparency
                )
                
                result['heatmap_path'] = str(heatmap_path)
                logger.info(f"Grad-CAM heatmap saved: {heatmap_path}")
            
            logger.info(f"Prediction: {tissue_type} (confidence: {confidence:.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def _create_diagnosis(self, tissue_type: str, confidence: float, is_malignant: bool) -> str:
        """
        Create a human-readable diagnosis string.
        
        Args:
            tissue_type: Predicted tissue class
            confidence: Prediction confidence
            is_malignant: Whether tissue is malignant
            
        Returns:
            Diagnosis string
        """
        
        # Get full description
        description = self.config['class_descriptions'][tissue_type]
        
        # Create diagnosis based on tissue type
        if is_malignant:
            diagnosis = f"Malignant tissue detected: {description}"
        elif tissue_type == 'NORM':
            diagnosis = f"Normal tissue: {description}"
        elif tissue_type in ['STR', 'LYM']:
            diagnosis = f"Reactive tissue: {description}"
        else:
            diagnosis = f"Benign tissue: {description}"
        
        return diagnosis
    
    def _get_clinical_group(self, tissue_type: str) -> str:
        """
        Determine which clinical group the tissue belongs to.
        
        Args:
            tissue_type: Tissue class name
            
        Returns:
            Clinical group name
        """
        
        for group_name, tissues in self.config['clinical_groups'].items():
            if tissue_type in tissues:
                return group_name
        
        return "unknown"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        
        return {
            'model_name': self.config['model_name'],
            'num_classes': self.config['num_classes'],
            'class_names': self.config['class_names'],
            'input_size': self.config['input_size'],
            'feature_dim': self.config['feature_dim'],
            'device': str(self.device),
            'classification_mode': self.config['classification_mode']
        }
    
    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        if hasattr(self, 'gradcam'):
            self.gradcam.remove_hooks()


# ═══════════════════════════════════════════════════════════
# TESTING AND VALIDATION
# ═══════════════════════════════════════════════════════════

def test_model_handler(
    model_path: str = '../models/vit_best.pth',
    config_path: str = '../data/classification_config.json',
    test_image_path: str = None,
    reference_image_path: str = None
):
    """
    Test the model handler with a sample image.
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to config JSON
        test_image_path: Path to test image
        reference_image_path: Path to reference image for stain normalization
    """
    
    print("\n" + "="*80)
    print("TESTING MODEL HANDLER WITH GRAD-CAM")
    print("="*80)
    
    # Initialize handler
    try:
        handler = ModelHandler(
            model_path=model_path,
            config_path=config_path,
            device='cuda',
            reference_image_path=reference_image_path,
            use_stain_norm=True
        )
        
        print("\n✓ Model handler initialized successfully")
        
        # Get model info
        info = handler.get_model_info()
        print("\nModel Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test prediction if image provided
        if test_image_path and Path(test_image_path).exists():
            print(f"\n{'='*80}")
            print("RUNNING PREDICTION TEST WITH GRAD-CAM")
            print("="*80)
            print(f"Test image: {Path(test_image_path).name}")
            
            # Run prediction with Grad-CAM
            result = handler.predict(
                test_image_path,
                generate_heatmap=True,
                heatmap_output_dir='../uploads'
            )
            
            print("\nPrediction Results:")
            print("-" * 80)
            print(f"Diagnosis: {result['diagnosis']}")
            print(f"Tissue Type: {result['tissue_type']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Is Malignant: {result['is_malignant']}")
            print(f"Clinical Group: {result['clinical_group']}")
            
            if 'heatmap_path' in result:
                print(f"\n✓ Grad-CAM Heatmap: {result['heatmap_path']}")
            
            print("\nTop 3 Predictions:")
            for i, pred in enumerate(result['top_3_predictions'], 1):
                print(f"  {i}. {pred['class']}: {pred['confidence']:.2%}")
                print(f"     {pred['description']}")
            
            print("\nAll Class Probabilities:")
            for class_name, prob in sorted(
                result['class_probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"  {class_name}: {prob:.2%}")
        
        print("\n" + "="*80)
        print("✓ Model handler test complete!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Example usage and testing.
    
    To test this module:
    1. Update the paths below
    2. Run: python model_handler.py
    """
    
    # Example paths (UPDATE THESE)
    MODEL_PATH = r"D:\Gpu-shit\Coloncancer\models\vit_best.pth"
    CONFIG_PATH = r"D:\Gpu-shit\Coloncancer\data\classification_config.json"
    REFERENCE_IMAGE = r"D:\NCT-CRC-HE-100K\TUM\TUM-EDPETKWQ.tif"
    TEST_IMAGE = r"D:\NCT-CRC-HE-100K\TUM\TUM-EDPETKWQ.tif"
    
    # Run test
    test_model_handler(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        test_image_path=TEST_IMAGE if Path(TEST_IMAGE).exists() else None,
        reference_image_path=REFERENCE_IMAGE if Path(REFERENCE_IMAGE).exists() else None
    )