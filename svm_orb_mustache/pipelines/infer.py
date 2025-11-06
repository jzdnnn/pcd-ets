"""
Inference pipeline for face detection, classification, and mustache overlay.
Supports both image and video processing with NMS.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from pathlib import Path

from pipelines.features import FeaturePipeline
from pipelines.train import SVMTrainer
from pipelines.overlay import MustacheOverlay, MustacheGallery
from pipelines.utils import non_max_suppression, Timer

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Complete inference pipeline: face detection -> ORB+BoVW -> SVM -> overlay.
    """
    
    def __init__(self,
                 face_cascade_path: str,
                 feature_pipeline: FeaturePipeline,
                 svm_trainer: SVMTrainer,
                 mustache_overlay: MustacheOverlay = None,
                 nms_threshold: float = 0.3,
                 confidence_threshold: float = 0.0,
                 scale_factor: float = 1.2,
                 min_neighbors: int = 5):
        """
        Initialize inference pipeline.
        
        Args:
            face_cascade_path: Path to Haar cascade for face detection
            feature_pipeline: Fitted feature extraction pipeline
            svm_trainer: Trained SVM classifier
            mustache_overlay: Mustache overlay handler
            nms_threshold: NMS IoU threshold
            confidence_threshold: Minimum confidence for face acceptance
            scale_factor: Haar cascade scale factor (1.1-1.5, default 1.2)
            min_neighbors: Haar cascade min neighbors (3-7, default 5)
        """
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.feature_pipeline = feature_pipeline
        self.svm_trainer = svm_trainer
        self.mustache_overlay = mustache_overlay
        self.nms_threshold = nms_threshold
        self.confidence_threshold = confidence_threshold
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        
        self.timer = Timer()
    
    def detect_face_candidates(self, 
                              image: np.ndarray,
                              min_size: Tuple[int, int] = (50, 50)) -> List[Tuple[int, int, int, int]]:
        """
        Detect face candidate regions using Haar cascade.
        
        Args:
            image: Input image
            min_size: Minimum face size
        
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=min_size
        )
        
        return [tuple(face) for face in faces]
    
    def classify_faces(self, 
                      image: np.ndarray, 
                      face_boxes: List[Tuple[int, int, int, int]],
                      roi_size: Tuple[int, int] = (128, 128)) -> Tuple[List[int], List[float]]:
        """
        Classify face candidates using SVM.
        
        Args:
            image: Input image
            face_boxes: List of face candidate boxes
            roi_size: Size to resize ROIs before feature extraction
        
        Returns:
            Tuple of (predictions, scores)
        """
        if len(face_boxes) == 0:
            return [], []
        
        # Extract ROIs
        rois = []
        for (x, y, w, h) in face_boxes:
            roi = image[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, roi_size)
            rois.append(roi_resized)
        
        rois = np.array(rois)
        
        # Extract features
        features = self.feature_pipeline.transform(rois)
        
        # Classify
        predictions = self.svm_trainer.predict(features)
        scores = self.svm_trainer.predict_proba(features)
        
        return predictions.tolist(), scores.tolist()
    
    def process_image(self, 
                     image: np.ndarray,
                     apply_overlay: bool = True,
                     draw_boxes: bool = False) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]], int]:
        """
        Process single image: detect, classify, and overlay.
        
        Args:
            image: Input image
            apply_overlay: Whether to apply mustache overlay
            draw_boxes: Whether to draw bounding boxes
        
        Returns:
            Tuple of (output_image, accepted_faces, num_faces)
        """
        self.timer.start('total')
        
        # Detect face candidates
        self.timer.start('detection')
        face_candidates = self.detect_face_candidates(image)
        self.timer.stop('detection')
        
        if len(face_candidates) == 0:
            self.timer.stop('total')
            return image.copy(), [], 0
        
        # Classify faces
        self.timer.start('classification')
        predictions, scores = self.classify_faces(image, face_candidates)
        self.timer.stop('classification')
        
        # Filter by confidence and apply NMS
        accepted_boxes = []
        accepted_scores = []
        
        for box, pred, score in zip(face_candidates, predictions, scores):
            if pred == 1 and score >= self.confidence_threshold:
                accepted_boxes.append(box)
                accepted_scores.append(score)
        
        # Apply NMS
        if len(accepted_boxes) > 0:
            keep_indices = non_max_suppression(
                accepted_boxes, 
                accepted_scores, 
                self.nms_threshold
            )
            accepted_boxes = [accepted_boxes[i] for i in keep_indices]
        
        # Create output image
        output = image.copy()
        
        # Apply mustache overlay
        if apply_overlay and self.mustache_overlay is not None:
            self.timer.start('overlay')
            for box in accepted_boxes:
                output = self.mustache_overlay.overlay_mustache(output, box)
            self.timer.stop('overlay')
        
        # Draw bounding boxes
        if draw_boxes:
            for box in accepted_boxes:
                x, y, w, h = box
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        self.timer.stop('total')
        
        return output, accepted_boxes, len(accepted_boxes)
    
    def process_video(self,
                     video_path: str,
                     output_path: str = None,
                     apply_overlay: bool = True,
                     show_fps: bool = True) -> int:
        """
        Process video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to output video (optional)
            apply_overlay: Whether to apply overlay
            show_fps: Whether to display FPS
        
        Returns:
            Total number of frames processed
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return 0
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        frame_times = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                output, faces, num_faces = self.process_image(frame, apply_overlay)
                
                # Add FPS overlay
                if show_fps:
                    avg_time = self.timer.get_avg('total')
                    if avg_time > 0:
                        current_fps = 1.0 / avg_time
                        cv2.putText(
                            output,
                            f"FPS: {current_fps:.1f} | Faces: {num_faces}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )
                
                # Write frame
                if writer:
                    writer.write(output)
                
                frame_count += 1
                
                if frame_count % 30 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames} frames")
        
        finally:
            cap.release()
            if writer:
                writer.release()
        
        logger.info(f"Video processing complete: {frame_count} frames")
        return frame_count
    
    def process_webcam(self,
                      camera_id: int = 0,
                      apply_overlay: bool = True,
                      show_info: bool = True,
                      save_screenshots: bool = False,
                      screenshot_dir: str = "screenshots") -> None:
        """
        Process live webcam feed.
        
        Args:
            camera_id: Camera device ID
            apply_overlay: Whether to apply overlay
            show_info: Whether to show FPS and controls
            save_screenshots: Whether to enable screenshot saving
            screenshot_dir: Directory for screenshots
        
        Controls:
            'q' - Quit
            'h' - Toggle overlay
            's' - Save screenshot (if enabled)
            'n' - Next mustache style
            'p' - Previous mustache style
            'b' - Toggle bounding boxes
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return
        
        # Setup screenshot directory
        if save_screenshots:
            Path(screenshot_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Webcam started. Press 'q' to quit.")
        logger.info("Controls: h=toggle overlay, s=screenshot, n/p=change style, b=boxes")
        
        overlay_enabled = apply_overlay
        show_boxes = False
        screenshot_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                output, faces, num_faces = self.process_image(
                    frame, 
                    apply_overlay=overlay_enabled,
                    draw_boxes=show_boxes
                )
                
                # Add info overlay
                if show_info:
                    avg_time = self.timer.get_avg('total')
                    if avg_time > 0:
                        current_fps = 1.0 / avg_time
                        
                        info_text = [
                            f"FPS: {current_fps:.1f}",
                            f"Faces: {num_faces}",
                            f"Overlay: {'ON' if overlay_enabled else 'OFF'}",
                        ]
                        
                        # Add mustache style if available
                        if self.mustache_overlay and hasattr(self.mustache_overlay, 'gallery'):
                            gallery = self.mustache_overlay.gallery
                            if gallery and gallery.current_style:
                                info_text.append(f"Style: {gallery.current_style}")
                        
                        for i, text in enumerate(info_text):
                            cv2.putText(
                                output,
                                text,
                                (10, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2
                            )
                
                # Display
                cv2.imshow('Mustache Try-On', output)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("Quitting...")
                    break
                elif key == ord('h'):
                    overlay_enabled = not overlay_enabled
                    logger.info(f"Overlay: {'ON' if overlay_enabled else 'OFF'}")
                elif key == ord('b'):
                    show_boxes = not show_boxes
                    logger.info(f"Bounding boxes: {'ON' if show_boxes else 'OFF'}")
                elif key == ord('s') and save_screenshots:
                    screenshot_path = f"{screenshot_dir}/screenshot_{screenshot_count:04d}.png"
                    cv2.imwrite(screenshot_path, output)
                    logger.info(f"Screenshot saved: {screenshot_path}")
                    screenshot_count += 1
                elif key == ord('n'):
                    # Next style (implement in gallery)
                    logger.info("Next style")
                elif key == ord('p'):
                    # Previous style
                    logger.info("Previous style")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        logger.info("Webcam session ended")
    
    def get_timing_report(self) -> dict:
        """Get timing statistics."""
        return self.timer.report()
