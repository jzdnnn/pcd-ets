"""
Dataset loading, splitting, and ROI generation for face/non-face classification.
Supports both pre-cropped datasets and full images with automatic ROI extraction.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from sklearn.model_selection import train_test_split
import json

logger = logging.getLogger(__name__)


class FaceDataset:
    """
    Dataset handler for face/non-face classification.
    Supports loading from directories and ROI generation.
    """
    
    def __init__(self, 
                 pos_dir: str = None, 
                 neg_dir: str = None,
                 face_cascade_path: str = None,
                 test_size: float = 0.15,
                 val_size: float = 0.15,
                 random_state: int = 42):
        """
        Initialize dataset.
        
        Args:
            pos_dir: Directory containing positive face samples
            neg_dir: Directory containing negative samples
            face_cascade_path: Path to Haar cascade for face detection
            test_size: Fraction of data for testing
            val_size: Fraction of data for validation
            random_state: Random seed for reproducibility
        """
        self.pos_dir = Path(pos_dir) if pos_dir else None
        self.neg_dir = Path(neg_dir) if neg_dir else None
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Initialize face detector for ROI generation
        self.face_cascade = None
        if face_cascade_path and Path(face_cascade_path).exists():
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Data storage
        self.X_train = []
        self.y_train = []
        self.X_val = []
        self.y_val = []
        self.X_test = []
        self.y_test = []
        
        self.train_paths = []
        self.val_paths = []
        self.test_paths = []
    
    def load_images_from_dir(self, directory: Path, label: int) -> List[Tuple[np.ndarray, int, str]]:
        """
        Load images from directory.
        
        Args:
            directory: Path to directory
            label: Label for images (1 for face, 0 for non-face)
        
        Returns:
            List of (image, label, path) tuples
        """
        samples = []
        
        if not directory.exists():
            logger.warning(f"Directory {directory} does not exist")
            return samples
        
        # Supported image formats
        formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        
        for fmt in formats:
            for img_path in directory.glob(fmt):
                img = cv2.imread(str(img_path))
                if img is not None:
                    samples.append((img, label, str(img_path)))
        
        logger.info(f"Loaded {len(samples)} images from {directory}")
        return samples
    
    def extract_face_rois(self, 
                         image: np.ndarray, 
                         min_size: Tuple[int, int] = (50, 50)) -> List[Tuple[int, int, int, int]]:
        """
        Extract face ROIs using Haar cascade.
        
        Args:
            image: Input image
            min_size: Minimum face size
        
        Returns:
            List of (x, y, w, h) face bounding boxes
        """
        if self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=min_size
        )
        
        return [tuple(face) for face in faces]
    
    def generate_rois_from_images(self, 
                                  samples: List[Tuple[np.ndarray, int, str]],
                                  roi_size: Tuple[int, int] = (128, 128)) -> List[Tuple[np.ndarray, int, str]]:
        """
        Generate ROIs from full images using face detection.
        
        Args:
            samples: List of (image, label, path) tuples
            roi_size: Target size for ROIs
        
        Returns:
            List of (roi_image, label, source_path) tuples
        """
        rois = []
        
        for img, label, path in samples:
            if label == 1:  # Positive samples - extract face ROIs
                faces = self.extract_face_rois(img)
                for (x, y, w, h) in faces:
                    roi = img[y:y+h, x:x+w]
                    roi_resized = cv2.resize(roi, roi_size)
                    rois.append((roi_resized, label, f"{path}_face_{x}_{y}"))
            else:  # Negative samples - random crops or full image
                # For negative samples, create random crops
                h, w = img.shape[:2]
                if h >= roi_size[1] and w >= roi_size[0]:
                    # Random crop
                    x = np.random.randint(0, w - roi_size[0])
                    y = np.random.randint(0, h - roi_size[1])
                    roi = img[y:y+roi_size[1], x:x+roi_size[0]]
                    rois.append((roi, label, f"{path}_crop_{x}_{y}"))
                else:
                    # Resize if too small
                    roi = cv2.resize(img, roi_size)
                    rois.append((roi, label, path))
        
        return rois
    
    def load_and_split(self, 
                      roi_size: Tuple[int, int] = (128, 128),
                      auto_generate_rois: bool = False) -> Dict[str, int]:
        """
        Load dataset and split into train/val/test sets.
        
        Args:
            roi_size: Target size for ROIs
            auto_generate_rois: Whether to auto-generate ROIs from full images
        
        Returns:
            Dictionary with dataset statistics
        """
        logger.info("Loading dataset...")
        
        # Load positive and negative samples
        pos_samples = []
        neg_samples = []
        
        if self.pos_dir:
            pos_samples = self.load_images_from_dir(self.pos_dir, label=1)
        
        if self.neg_dir:
            neg_samples = self.load_images_from_dir(self.neg_dir, label=0)
        
        # Generate ROIs if requested
        if auto_generate_rois and self.face_cascade is not None:
            logger.info("Generating ROIs from images...")
            pos_samples = self.generate_rois_from_images(pos_samples, roi_size)
            neg_samples = self.generate_rois_from_images(neg_samples, roi_size)
        else:
            # Just resize images to ROI size
            pos_samples = [(cv2.resize(img, roi_size), label, path) 
                          for img, label, path in pos_samples]
            neg_samples = [(cv2.resize(img, roi_size), label, path) 
                          for img, label, path in neg_samples]
        
        # Combine samples
        all_samples = pos_samples + neg_samples
        
        if len(all_samples) == 0:
            logger.error("No samples loaded!")
            return {}
        
        # Extract X, y, paths
        X = np.array([img for img, _, _ in all_samples])
        y = np.array([label for _, label, _ in all_samples])
        paths = [path for _, _, path in all_samples]
        
        # First split: train+val vs test
        X_temp, self.X_test, y_temp, self.y_test, paths_temp, self.test_paths = \
            train_test_split(
                X, y, paths, 
                test_size=self.test_size, 
                stratify=y,
                random_state=self.random_state
            )
        
        # Second split: train vs val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        self.X_train, self.X_val, self.y_train, self.y_val, self.train_paths, self.val_paths = \
            train_test_split(
                X_temp, y_temp, paths_temp,
                test_size=val_size_adjusted,
                stratify=y_temp,
                random_state=self.random_state
            )
        
        stats = {
            'total': len(all_samples),
            'train': len(self.X_train),
            'val': len(self.X_val),
            'test': len(self.X_test),
            'train_pos': np.sum(self.y_train == 1),
            'train_neg': np.sum(self.y_train == 0),
            'val_pos': np.sum(self.y_val == 1),
            'val_neg': np.sum(self.y_val == 0),
            'test_pos': np.sum(self.y_test == 1),
            'test_neg': np.sum(self.y_test == 0),
        }
        
        logger.info(f"Dataset split complete: {stats}")
        return stats
    
    def save_split_info(self, output_path: str):
        """Save split information to JSON file."""
        split_info = {
            'train': self.train_paths,
            'val': self.val_paths,
            'test': self.test_paths,
            'train_labels': self.y_train.tolist(),
            'val_labels': self.y_val.tolist(),
            'test_labels': self.y_test.tolist(),
        }
        
        with open(output_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info(f"Split info saved to {output_path}")
    
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get training data."""
        return self.X_train, self.y_train
    
    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get validation data."""
        return self.X_val, self.y_val
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get test data."""
        return self.X_test, self.y_test
