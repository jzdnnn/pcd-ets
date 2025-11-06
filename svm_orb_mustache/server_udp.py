import cv2
import json
import socket
import base64
import numpy as np
from pathlib import Path
import time

from pipelines.infer import InferencePipeline


class MustacheUDPServer:
    
    def __init__(self, host='127.0.0.1', port=5005, 
                 mustache_scale=1.0, mustache_y_offset=0.6,
                 smoothing_factor=0.3, scale_factor=1.2, min_neighbors=5,
                 max_disappeared=5):
        self.host = host
        self.port = port
        self.mustache_scale = mustache_scale
        self.mustache_y_offset = mustache_y_offset
        self.smoothing_factor = smoothing_factor
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.max_disappeared = max_disappeared
        
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self.command_port = port + 1
        try:
            self.sock_recv.bind((host, self.command_port))
            self.sock_recv.setblocking(False)
            print(f"UDP Server initialized:")
            print(f"   Sending video to: {host}:{port}")
            print(f"   Listening for commands on: {host}:{self.command_port}")
        except Exception as e:
            print(f"Warning: Could not bind command socket: {e}")
            print(f"   Server will only send data, not receive commands")
            self.sock_recv = None
        
        self.prev_boxes = {}
        self.disappeared_count = {}
        self.face_id_counter = 0
        
        print("Loading models...")
        from pipelines.features import FeaturePipeline
        from pipelines.train import SVMTrainer
        
        # Load feature pipeline
        feature_pipeline = FeaturePipeline()
        feature_pipeline.load_codebook('models/codebook.pkl')
        print("    Feature pipeline loaded")
        
        # Load SVM trainer
        svm_trainer = SVMTrainer()
        svm_trainer.load('models/svm.pkl', 'models/scaler.pkl')
        print("    SVM model loaded")
        
        # Initialize mustache gallery and overlay
        print("Loading mustache assets...")
        from pipelines.overlay import MustacheOverlay, MustacheGallery
        
        # Load gallery
        self.gallery = MustacheGallery('assets/mustaches')
        styles = self.gallery.get_styles()
        print(f"    Loaded {len(styles)} mustache styles: {styles}")
        
        # Create overlay with first style
        if styles:
            first_style_path = self.gallery.get_current_path()
            mustache_overlay = MustacheOverlay(
                mustache_path=first_style_path,
                eye_cascade_path='assets/cascades/haarcascade_eye.xml'
            )
            print(f"    Using mustache style: {self.gallery.current_style}")
        else:
            mustache_overlay = None
            print("    No mustache styles found!")
        
        # Initialize inference pipeline
        print("Initializing inference pipeline...")
        self.pipeline = InferencePipeline(
            face_cascade_path='assets/cascades/haarcascade_frontalface_default.xml',
            feature_pipeline=feature_pipeline,
            svm_trainer=svm_trainer,
            mustache_overlay=None,  # We'll handle overlay manually for custom positioning
            nms_threshold=0.3,
            confidence_threshold=0.0,
            scale_factor=self.scale_factor,
            min_neighbors=self.min_neighbors
        )
        print(f"    Inference pipeline ready (scale={self.scale_factor}, neighbors={self.min_neighbors})")
        
        # Store mustache overlay for manual application
        self.mustache_overlay = mustache_overlay
        
        # Stats
        self.frame_count = 0
        self.start_time = time.time()
        
        print("Server ready!")
    
    def encode_frame(self, frame, quality=80):
        """
        Encode frame as JPEG base64 string.
        
        Args:
            frame: OpenCV BGR image
            quality: JPEG quality (1-100)
        
        Returns:
            Tuple of (base64 encoded string, scale_factor)
        """
        # Resize frame for faster transmission
        height, width = frame.shape[:2]
        max_width = 640
        scale_factor = 1.0
        
        if width > max_width:
            scale_factor = max_width / width
            new_width = max_width
            new_height = int(height * scale_factor)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Encode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        # Convert to base64
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        return jpg_as_text, scale_factor
    
    def smooth_bounding_box(self, new_box, prev_box=None):
        """
        Apply Exponential Moving Average (EMA) smoothing to bounding box.
        This reduces jittery movements from frame-to-frame detection noise.
        
        Args:
            new_box: New detected box (x, y, w, h)
            prev_box: Previous smoothed box (x, y, w, h) or None
        
        Returns:
            Smoothed bounding box (x, y, w, h)
        """
        if prev_box is None:
            # First detection - no smoothing
            return new_box
        
        # Apply EMA: smoothed = alpha * new + (1-alpha) * prev
        alpha = self.smoothing_factor
        x = alpha * new_box[0] + (1 - alpha) * prev_box[0]
        y = alpha * new_box[1] + (1 - alpha) * prev_box[1]
        w = alpha * new_box[2] + (1 - alpha) * prev_box[2]
        h = alpha * new_box[3] + (1 - alpha) * prev_box[3]
        
        return (int(x), int(y), int(w), int(h))
    
    def match_boxes_to_tracks(self, current_boxes):
        """
        Match current detected boxes to previous tracked boxes.
        Uses IoU (Intersection over Union) for matching.
        Implements persistence - keeps showing boxes for a few frames after face disappears.
        
        Args:
            current_boxes: List of newly detected boxes
        
        Returns:
            List of smoothed boxes
        """
        if not current_boxes and not self.prev_boxes:
            # No detection and no previous - return empty
            return []
        
        if not self.prev_boxes:
            # First frame - initialize tracking
            smoothed = []
            for i, box in enumerate(current_boxes):
                self.prev_boxes[i] = box
                self.disappeared_count[i] = 0
                smoothed.append(box)
            return smoothed
        
        # Match boxes using simple distance metric
        smoothed_boxes = []
        matched_ids = set()
        
        for new_box in current_boxes:
            # Find closest previous box
            best_match_id = None
            min_distance = float('inf')
            
            for prev_id, prev_box in self.prev_boxes.items():
                # Calculate center distance
                new_cx = new_box[0] + new_box[2] / 2
                new_cy = new_box[1] + new_box[3] / 2
                prev_cx = prev_box[0] + prev_box[2] / 2
                prev_cy = prev_box[1] + prev_box[3] / 2
                
                distance = np.sqrt((new_cx - prev_cx)**2 + (new_cy - prev_cy)**2)
                
                # Increased threshold for better tracking when moving
                # Changed from 100 to 200 pixels to handle faster movement
                if distance < 200 and distance < min_distance:
                    min_distance = distance
                    best_match_id = prev_id
            
            # Apply smoothing
            if best_match_id is not None:
                smoothed_box = self.smooth_bounding_box(new_box, self.prev_boxes[best_match_id])
                self.prev_boxes[best_match_id] = smoothed_box
                self.disappeared_count[best_match_id] = 0  # Reset disappeared counter
                matched_ids.add(best_match_id)
                smoothed_boxes.append(smoothed_box)
            else:
                # New face - create new track
                new_id = self.face_id_counter
                self.face_id_counter += 1
                self.prev_boxes[new_id] = new_box
                self.disappeared_count[new_id] = 0
                smoothed_box = new_box
                smoothed_boxes.append(smoothed_box)
        
        # Handle faces that disappeared (not matched in current frame)
        # Keep showing them for a few frames (persistence)
        unmatched_ids = set(self.prev_boxes.keys()) - matched_ids
        for old_id in list(unmatched_ids):
            self.disappeared_count[old_id] += 1
            
            # Keep showing box if disappeared count hasn't reached max
            if self.disappeared_count[old_id] <= self.max_disappeared:
                smoothed_boxes.append(self.prev_boxes[old_id])
            else:
                # Remove track after max_disappeared frames
                del self.prev_boxes[old_id]
                del self.disappeared_count[old_id]
        
        return smoothed_boxes
    
    def change_mustache_style(self, style_name):
        """
        Change the current mustache style.
        
        Args:
            style_name: Name of the style to switch to
        """
        from pipelines.overlay import MustacheOverlay
        
        if style_name not in self.gallery.get_styles():
            print(f"Unknown style: {style_name}")
            return False
        
        # Update gallery
        self.gallery.set_current_style(style_name)
        
        # Reload overlay with new style
        style_path = self.gallery.get_current_path()
        self.mustache_overlay = MustacheOverlay(
            mustache_path=style_path,
            eye_cascade_path='assets/cascades/haarcascade_eye.xml'
        )
        
        print(f"Switched to mustache style: {style_name}")
        return True
    
    def update_parameters(self, params):
        """
        Update detection and overlay parameters dynamically.
        
        Args:
            params: Dictionary with parameter updates
        """
        updated = []
        
        if 'scale_factor' in params:
            new_value = float(params['scale_factor'])
            if 1.05 <= new_value <= 2.0:
                self.scale_factor = new_value
                self.pipeline.scale_factor = new_value
                updated.append(f"scale_factor={new_value:.2f}")
        
        if 'min_neighbors' in params:
            new_value = int(params['min_neighbors'])
            if 1 <= new_value <= 10:
                self.min_neighbors = new_value
                self.pipeline.min_neighbors = new_value
                updated.append(f"min_neighbors={new_value}")
        
        if 'mustache_scale' in params:
            new_value = float(params['mustache_scale'])
            if 0.1 <= new_value <= 3.0:
                self.mustache_scale = new_value
                updated.append(f"mustache_scale={new_value:.2f}")
        
        if 'mustache_y_offset' in params:
            new_value = float(params['mustache_y_offset'])
            if 0.0 <= new_value <= 1.0:
                self.mustache_y_offset = new_value
                updated.append(f"mustache_y_offset={new_value:.2f}")
        
        if updated:
            print(f"Parameters updated: {', '.join(updated)}")
            return True
        return False
    
    def check_for_commands(self):
        """
        Check for incoming commands from client (non-blocking).
        Commands are JSON messages like: {"command": "change_style", "style": "style2"}
        """
        if self.sock_recv is None:
            return
        
        try:
            data, addr = self.sock_recv.recvfrom(1024)
            message = json.loads(data.decode('utf-8'))
            
            print(f"Received command from {addr}: {message}")
            
            command = message.get('command')
            
            if command == 'change_style':
                style = message.get('style')
                if style:
                    self.change_mustache_style(style)
            elif command == 'next_style':
                self.gallery.next_style()
                self.change_mustache_style(self.gallery.current_style)
            elif command == 'prev_style':
                self.gallery.prev_style()
                self.change_mustache_style(self.gallery.current_style)
            elif command == 'update_parameters':
                params = message.get('parameters', {})
                if params:
                    self.update_parameters(params)
                    
        except BlockingIOError:
            # No data available, continue
            pass
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
        except Exception as e:
            print(f"Error processing command: {e}")
    
    def send_data(self, data_dict):
        """
        Send data to Godot client via UDP.
        
        Args:
            data_dict: Dictionary containing detection results
        """
        try:
            # Convert to JSON
            json_data = json.dumps(data_dict)
            
            # Send via UDP (use send socket)
            self.sock_send.sendto(json_data.encode('utf-8'), (self.host, self.port))
            
        except Exception as e:
            print(f"Error sending data: {e}")
    
    def process_frame(self, frame, show_overlay=True):
        """
        Process frame and send results to Godot.
        
        Args:
            frame: OpenCV BGR image
            show_overlay: Whether to overlay mustache on frame
        
        Returns:
            Processed frame
        """
        # Process with inference pipeline (without overlay)
        result_frame, accepted_boxes, face_count = self.pipeline.process_image(
            frame,
            apply_overlay=False,  # We'll apply manually
            draw_boxes=False  # We'll draw boxes after smoothing
        )
        
        # Apply bounding box smoothing to reduce jitter
        smoothed_boxes = self.match_boxes_to_tracks(accepted_boxes)
        
        # DEBUG: Log box counts
        print(f"DEBUG: accepted_boxes={len(accepted_boxes)}, smoothed_boxes={len(smoothed_boxes)}")
        
        # Apply mustache overlay with smoothed boxes
        if show_overlay and self.mustache_overlay is not None:
            for bbox in smoothed_boxes:
                result_frame = self.mustache_overlay.overlay_mustache(
                    result_frame,
                    bbox,
                    scale_factor=self.mustache_scale,
                    y_offset_ratio=self.mustache_y_offset,
                    enable_rotation=False  # Disable rotation to prevent jitter
                )
        
        # Encode frame and get scale factor
        frame_encoded, frame_scale = self.encode_frame(result_frame)
        
        # Prepare data for Godot with scaled coordinates
        faces_data = []
        for bbox in smoothed_boxes:
            x, y, w, h = bbox
            
            # Scale coordinates to match the resized frame
            faces_data.append({
                'x': int(x * frame_scale),
                'y': int(y * frame_scale),
                'width': int(w * frame_scale),
                'height': int(h * frame_scale),
                'confidence': 1.0  # InferencePipeline doesn't return scores after NMS
            })
        
        # DEBUG: Log faces data being sent
        print(f"DEBUG: Sending {len(faces_data)} faces to Godot (frame_scale={frame_scale:.3f})")
        
        # Prepare data packet
        data_packet = {
            'timestamp': time.time(),
            'frame': frame_encoded,
            'faces': faces_data,
            'face_count': len(faces_data),
            'fps': self.get_fps(),
            'mustache_styles': self.gallery.get_styles(),
            'current_style': self.gallery.current_style
        }
        
        # Send to Godot
        self.send_data(data_packet)
        
        self.frame_count += 1
        
        return result_frame
    
    def get_fps(self):
        """Calculate current FPS."""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0.0
    
    def run(self, camera_id=0):
        """
        Run server with webcam capture.
        
        Args:
            camera_id: Webcam device ID
        """
        print(f"\nStarting webcam capture (Camera {camera_id})...")
        print(f"Sending data to Godot client at {self.host}:{self.port}")
        print("\nControls:")
        print("  q - Quit")
        print("  h - Toggle mustache overlay")
        print("  s - Take screenshot")
        print()
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Cannot open camera!")
            return
        
        show_mustache = True
        
        try:
            while True:
                # Check for incoming commands from Godot client
                self.check_for_commands()
                
                ret, frame = cap.read()
                
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process and send
                result_frame = self.process_frame(frame, show_overlay=show_mustache)
                
                # Display locally (optional) - DISABLED for Godot integration
                # cv2.imshow('Python Server - Mustache Detection', result_frame)
                
                # Small delay for loop timing (replaces waitKey)
                time.sleep(0.001)
                
                # Handle keyboard (disabled since no window)
                key = 0xFF  # cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.sock.close()
            
            # Final stats
            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            print("\n" + "=" * 60)
            print("Session Stats")
            print("=" * 60)
            print(f"Total frames: {self.frame_count}")
            print(f"Duration: {elapsed:.1f}s")
            print(f"Average FPS: {avg_fps:.1f}")
            print("=" * 60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Mustache Detection UDP Server')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                      help='Server host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5005,
                      help='UDP port (default: 5005)')
    parser.add_argument('--camera', type=int, default=0,
                      help='Camera device ID (default: 0)')
    parser.add_argument('--mustache-scale', type=float, default=0.8,
                      help='Mustache width scale (0.8-1.2, default: 0.8)')
    parser.add_argument('--mustache-y-offset', type=float, default=0.57,
                      help='Mustache vertical position (0.5-0.7, default: 0.57)')
    parser.add_argument('--smoothing', type=float, default=0.3,
                      help='Bounding box smoothing (0-1, default: 0.3, higher=smoother but laggier)')
    parser.add_argument('--scale-factor', type=float, default=1.2,
                      help='Haar cascade scale factor (1.1-1.5, default: 1.2, lower=more accurate but slower)')
    parser.add_argument('--min-neighbors', type=int, default=5,
                      help='Haar cascade min neighbors (3-7, default: 5, higher=stricter detection)')
    
    args = parser.parse_args()
    
    # Check if models exist
    required_files = [
        'models/svm.pkl',
        'models/scaler.pkl',
        'models/codebook.pkl'
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print("=" * 60)
        print("ERROR: Required model files not found!")
        print("=" * 60)
        print("\nMissing files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nYou need to train the model first:")
        print("  python app.py train --pos_dir data/faces --neg_dir data/non_faces")
        print("\nThen run this server again.")
        print("=" * 60)
        return
    
    # Start server
    print("=" * 60)
    print("Mustache Detection UDP Server")
    print("=" * 60)
    print(f"Mustache Scale: {args.mustache_scale}")
    print(f"Mustache Y-Offset: {args.mustache_y_offset}")
    print(f"Smoothing Factor: {args.smoothing}")
    print(f"Scale Factor: {args.scale_factor}")
    print(f"Min Neighbors: {args.min_neighbors}")
    
    server = MustacheUDPServer(
        host=args.host, 
        port=args.port,
        mustache_scale=args.mustache_scale,
        mustache_y_offset=args.mustache_y_offset,
        smoothing_factor=args.smoothing,
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors
    )
    server.run(camera_id=args.camera)


if __name__ == '__main__':
    main()
