"""
Feature extraction using ORB and Bag of Visual Words (BoVW).
Implements codebook creation and histogram encoding for SVM classification.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from sklearn.cluster import MiniBatchKMeans
import joblib

logger = logging.getLogger(__name__)


class ORBFeatureExtractor:
    """
    ORB feature extractor with Bag of Visual Words encoding.
    """
    
    def __init__(self, 
                 n_features: int = 500,
                 scale_factor: float = 1.2,
                 n_levels: int = 8,
                 edge_threshold: int = 31):
        """
        Initialize ORB detector.
        
        Args:
            n_features: Maximum number of features to detect
            scale_factor: Pyramid decimation ratio
            n_levels: Number of pyramid levels
            edge_threshold: Size of border where features are not detected
        """
        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=edge_threshold
        )
        self.n_features = n_features
    
    def extract_descriptors(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract ORB descriptors from image.
        
        Args:
            image: Input image (BGR or grayscale)
        
        Returns:
            Descriptors array or None if no keypoints detected
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect and compute
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None or len(descriptors) == 0:
            return None
        
        return descriptors
    
    def extract_keypoints_and_descriptors(self, image: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        """
        Extract both keypoints and descriptors.
        
        Args:
            image: Input image
        
        Returns:
            Tuple of (keypoints, descriptors)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors


class BoVWEncoder:
    """
    Bag of Visual Words encoder using k-means clustering.
    """
    
    def __init__(self, k: int = 256, random_state: int = 42):
        """
        Initialize BoVW encoder.
        
        Args:
            k: Number of visual words (clusters)
            random_state: Random seed for reproducibility
        """
        self.k = k
        self.random_state = random_state
        self.kmeans = None
        self.is_fitted = False
    
    def fit(self, descriptors_list: List[np.ndarray], max_descriptors: int = 200000):
        """
        Build codebook from descriptors using k-means clustering.
        
        Args:
            descriptors_list: List of descriptor arrays from multiple images
            max_descriptors: Maximum number of descriptors to use for clustering
        
        Returns:
            Self
        """
        logger.info("Building BoVW codebook...")
        
        # Collect all descriptors
        all_descriptors = []
        for desc in descriptors_list:
            if desc is not None and len(desc) > 0:
                all_descriptors.append(desc)
        
        if len(all_descriptors) == 0:
            raise ValueError("No descriptors provided for codebook creation")
        
        # Concatenate all descriptors
        all_descriptors = np.vstack(all_descriptors)
        logger.info(f"Total descriptors collected: {len(all_descriptors)}")
        
        # Subsample if too many descriptors
        if len(all_descriptors) > max_descriptors:
            logger.info(f"Subsampling to {max_descriptors} descriptors")
            indices = np.random.choice(
                len(all_descriptors), 
                max_descriptors, 
                replace=False
            )
            all_descriptors = all_descriptors[indices]
        
        # Convert to float32 for clustering
        all_descriptors = all_descriptors.astype(np.float32)
        
        # Fit k-means
        logger.info(f"Fitting k-means with k={self.k}...")
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.k,
            random_state=self.random_state,
            batch_size=1000,
            max_iter=100,
            verbose=0
        )
        self.kmeans.fit(all_descriptors)
        self.is_fitted = True
        
        logger.info("Codebook created successfully")
        return self
    
    def encode(self, descriptors: Optional[np.ndarray], normalize: bool = True) -> np.ndarray:
        """
        Encode descriptors as BoVW histogram.
        
        Args:
            descriptors: ORB descriptors from an image
            normalize: Whether to L1-normalize the histogram
        
        Returns:
            BoVW histogram of length k
        """
        if not self.is_fitted:
            raise ValueError("Codebook not fitted. Call fit() first.")
        
        # Handle empty descriptors
        if descriptors is None or len(descriptors) == 0:
            # Return uniform histogram as fallback
            histogram = np.ones(self.k, dtype=np.float32) / self.k
            return histogram
        
        # Convert to float32
        descriptors = descriptors.astype(np.float32)
        
        # Assign each descriptor to nearest cluster
        labels = self.kmeans.predict(descriptors)
        
        # Build histogram
        histogram = np.bincount(labels, minlength=self.k).astype(np.float32)
        
        # Normalize
        if normalize:
            total = np.sum(histogram)
            if total > 0:
                histogram = histogram / total
        
        return histogram
    
    def encode_batch(self, descriptors_list: List[Optional[np.ndarray]], 
                    normalize: bool = True) -> np.ndarray:
        """
        Encode multiple descriptor sets as BoVW histograms.
        
        Args:
            descriptors_list: List of descriptor arrays
            normalize: Whether to normalize histograms
        
        Returns:
            Array of BoVW histograms, shape (n_samples, k)
        """
        histograms = []
        for desc in descriptors_list:
            hist = self.encode(desc, normalize=normalize)
            histograms.append(hist)
        return np.array(histograms)
    
    def save(self, path: str):
        """Save codebook to file."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted codebook")
        joblib.dump(self.kmeans, path)
        logger.info(f"Codebook saved to {path}")
    
    def load(self, path: str):
        """Load codebook from file."""
        self.kmeans = joblib.load(path)
        self.k = self.kmeans.n_clusters
        self.is_fitted = True
        logger.info(f"Codebook loaded from {path}")


class FeaturePipeline:
    """
    Complete feature extraction pipeline: ORB + BoVW encoding.
    """
    
    def __init__(self, 
                 orb_n_features: int = 500,
                 bovw_k: int = 256,
                 random_state: int = 42):
        """
        Initialize feature pipeline.
        
        Args:
            orb_n_features: Number of ORB features per image
            bovw_k: Number of visual words
            random_state: Random seed
        """
        self.extractor = ORBFeatureExtractor(n_features=orb_n_features)
        self.encoder = BoVWEncoder(k=bovw_k, random_state=random_state)
        self.random_state = random_state
    
    def fit(self, images: np.ndarray, max_descriptors: int = 200000):
        """
        Fit the codebook on training images.
        
        Args:
            images: Array of training images
            max_descriptors: Maximum descriptors for codebook
        
        Returns:
            Self
        """
        logger.info(f"Extracting ORB descriptors from {len(images)} images...")
        
        descriptors_list = []
        for img in images:
            desc = self.extractor.extract_descriptors(img)
            if desc is not None:
                descriptors_list.append(desc)
        
        logger.info(f"Extracted descriptors from {len(descriptors_list)} images")
        
        # Fit codebook
        self.encoder.fit(descriptors_list, max_descriptors=max_descriptors)
        
        return self
    
    def transform(self, images: np.ndarray) -> np.ndarray:
        """
        Transform images to BoVW feature vectors.
        
        Args:
            images: Array of images
        
        Returns:
            Array of BoVW feature vectors
        """
        descriptors_list = []
        for img in images:
            desc = self.extractor.extract_descriptors(img)
            descriptors_list.append(desc)
        
        return self.encoder.encode_batch(descriptors_list)
    
    def fit_transform(self, images: np.ndarray, max_descriptors: int = 200000) -> np.ndarray:
        """
        Fit codebook and transform images in one step.
        
        Args:
            images: Array of training images
            max_descriptors: Maximum descriptors for codebook
        
        Returns:
            Array of BoVW feature vectors
        """
        self.fit(images, max_descriptors)
        return self.transform(images)
    
    def save_codebook(self, path: str):
        """Save codebook."""
        self.encoder.save(path)
    
    def load_codebook(self, path: str):
        """Load codebook."""
        self.encoder.load(path)
