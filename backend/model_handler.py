# backend/model_handler.py

"""
Model Handler for Colorectal Cancer Diagnosis System

This module handles:
1. Loading the trained Lunit DINO ViT model
2. Running inference on preprocessed images
3. Post-processing predictions (softmax, class mapping, confidence)
4. Returning structured results to Flask API

Model: Lunit DINO (ViT-Small/8) fine-tuned on NCT-CRC-HE-100K
Task: 9-class tissue classification
Classes: ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM
"""

import torch
import torch.nn as nn
import timm
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, Optional
import logging

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
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0  # Remove default head
        )
        
        # Add custom classification head (matches training architecture)
        self.head = ViTClassificationHead(
            num_features=num_features,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_batch_norm=True  # Match training config
        )
        
        logger.info(f"✓ Initialized LunitDINOClassifier")
        logger.info(f"  Backbone: {model_name}")
        logger.info(f"  Num classes: {num_classes}")
        logger.info(f"  Feature dim: {num_features}")
    
    def forward(self, x):
        """Forward pass through backbone and head."""
        features = self.backbone(x)  # [B, 384]
        logits = self.head(features)  # [B, num_classes]
        return logits


# ═══════════════════════════════════════════════════════════
# MODEL HANDLER CLASS
# ═══════════════════════════════════════════════════════════

class ModelHandler:
    """
    Handles model loading, preprocessing, inference, and post-processing.
    
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
    
    def predict(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Run inference on an image and return structured predictions.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing:
            {
                'diagnosis': str,              # Primary diagnosis
                'confidence': float,           # Confidence score (0-1)
                'tissue_type': str,           # Tissue class name
                'is_malignant': bool,         # Whether tissue is malignant
                'class_probabilities': dict,  # All class probabilities
                'top_3_predictions': list,    # Top 3 predictions with confidence
                'clinical_group': str         # Clinical grouping
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
    print("TESTING MODEL HANDLER")
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
            print("RUNNING PREDICTION TEST")
            print("="*80)
            print(f"Test image: {Path(test_image_path).name}")
            
            result = handler.predict(test_image_path)
            
            print("\nPrediction Results:")
            print("-" * 80)
            print(f"Diagnosis: {result['diagnosis']}")
            print(f"Tissue Type: {result['tissue_type']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Is Malignant: {result['is_malignant']}")
            print(f"Clinical Group: {result['clinical_group']}")
            
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