import cv2
import numpy as np
import time
import mediapipe as mp
import pyautogui
from typing import Tuple, Optional
from collections import deque

from eyetrax.calibration.twelve_point import run_12_point_calibration, GazeSmoother
from eyetrax.utils.screen import get_screen_size
from eyetrax.gaze import GazeEstimator

class BlinkDetector:
    def __init__(self, blink_threshold: float = 0.28, min_blink_duration: float = 0.1, max_blink_duration: float = 0.5):
        self.blink_threshold = blink_threshold  # Increased threshold for better sensitivity
        self.min_blink_duration = min_blink_duration  # Reduced minimum duration
        self.max_blink_duration = max_blink_duration  # Added maximum duration
        self.blink_start_time: Optional[float] = None
        self.is_blinking = False
        self.last_blink_time = 0
        self.blink_cooldown = 0.5  # Reduced cooldown between blinks
        self.debug_info = {
            'current_ear': 0.0,
            'blink_count': 0,
            'last_blink_duration': 0.0
        }
    
    def update(self, eye_aspect_ratio: float) -> bool:
        """
        Update blink detection state and return True if a valid blink is detected
        """
        current_time = time.time()
        self.debug_info['current_ear'] = eye_aspect_ratio
        
        if eye_aspect_ratio < self.blink_threshold:
            if not self.is_blinking:
                self.blink_start_time = current_time
                self.is_blinking = True
        else:
            if self.is_blinking:
                blink_duration = current_time - self.blink_start_time
                self.debug_info['last_blink_duration'] = blink_duration
                
                if (self.min_blink_duration <= blink_duration <= self.max_blink_duration and 
                    current_time - self.last_blink_time >= self.blink_cooldown):
                    self.last_blink_time = current_time
                    self.is_blinking = False
                    self.debug_info['blink_count'] += 1
                    return True
                self.is_blinking = False
        return False

class FaceDetector:
    def __init__(self,
                 static_mode=False,
                 max_faces=1,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 draw_landmarks=True):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.draw_landmarks = draw_landmarks

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=self.static_mode,
            max_num_faces=self.max_faces,
            refine_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def detect(self, frame):
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        return results, frame

def calculate_eye_aspect_ratio(landmarks, eye_indices):
    """Calculate the eye aspect ratio for blink detection"""
    # Get the vertical eye landmarks (top and bottom)
    vertical_dist1 = np.linalg.norm(
        np.array([landmarks[eye_indices[1]].y, landmarks[eye_indices[2]].y]) -
        np.array([landmarks[eye_indices[5]].y, landmarks[eye_indices[4]].y])
    )
    vertical_dist2 = np.linalg.norm(
        np.array([landmarks[eye_indices[0]].y, landmarks[eye_indices[3]].y]) -
        np.array([landmarks[eye_indices[5]].y, landmarks[eye_indices[4]].y])
    )
    vertical_dist = (vertical_dist1 + vertical_dist2) / 2.0
    
    # Get the horizontal eye landmarks (left and right)
    horizontal_dist = np.linalg.norm(
        np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[3]].x]) -
        np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[2]].x])
    )
    
    # Calculate the eye aspect ratio
    ear = vertical_dist / (2.0 * horizontal_dist)
    return ear

def try_camera_connection(camera_indices=[0, 1, 2], max_attempts=3):
    """Try to connect to available cameras with fallback options"""
    for camera_index in camera_indices:
        for attempt in range(max_attempts):
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Successfully connected to camera {camera_index}")
                    return cap, camera_index
            cap.release()
            time.sleep(0.5)  # Wait before retry
    return None, None

def run_accuracy_demo(camera_index: int = 0):
    """
    Run the accuracy testing demo with mouse control and blinking detection
    """
    # Disable PyAutoGUI fail-safe
    pyautogui.FAILSAFE = False
    
    # Initialize screen size and try camera connection
    sw, sh = get_screen_size()
    
    # Try to connect to camera with fallback options
    cap, camera_index = try_camera_connection([camera_index, 1, 2])
    if cap is None:
        print("Error: Could not connect to any camera")
        return
    
    # Get camera dimensions
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Could not read from camera")
        cap.release()
        return
    h, w = frame.shape[:2]
    
    # Initialize face detector with improved parameters
    face_detector = FaceDetector(
        static_mode=False,
        max_faces=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        draw_landmarks=True
    )
    
    # Initialize blink detector with adjusted parameters
    blink_detector = BlinkDetector(
        blink_threshold=0.28,  # User-adjusted threshold
        min_blink_duration=0.1,
        max_blink_duration=0.5
    )
    gaze_smoother = GazeSmoother(window_size=5)
    
    # Initialize gaze estimator
    gaze_estimator = GazeEstimator()
    
    # Variables for cursor control
    last_gaze_x, last_gaze_y = None, None
    cursor_paused = False
    
    # Variables for camera reconnection
    last_frame_time = time.time()
    max_frame_interval = 1.0  # Maximum time between frames
    reconnect_attempts = 0
    max_reconnect_attempts = 3
    
    try:
        # Run calibration first
        print("Starting calibration...")
        run_12_point_calibration(gaze_estimator, camera_index)
        print("Calibration complete!")
        
        # Eye landmark indices for blink detection
        LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        
        # Mouse control variables
        last_click_time = 0
        click_cooldown = 1.0
        can_click = True
        
        # Create fullscreen window
        cv2.namedWindow("Accuracy Demo", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Accuracy Demo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Create target points for accuracy testing
        target_points = []
        for _ in range(5):  # 5 random target points
            x = np.random.randint(100, sw - 100)
            y = np.random.randint(100, sh - 100)
            target_points.append((x, y))
        
        current_target = 0
        target_radius = 30
        accuracy_data = []
        
        print("Starting accuracy demo...")
        print("Look at the targets and blink to click. Press ESC to exit.")
        
        while True:
            ret, frame = cap.read()
            current_time = time.time()
            
            # Check for camera connection issues
            if not ret or frame is None:
                if current_time - last_frame_time > max_frame_interval:
                    print("Camera connection lost. Attempting to reconnect...")
                    cap.release()
                    cap, new_camera_index = try_camera_connection([camera_index, 1, 2])
                    if cap is None:
                        print("Failed to reconnect to camera")
                        break
                    camera_index = new_camera_index
                    reconnect_attempts += 1
                    if reconnect_attempts >= max_reconnect_attempts:
                        print("Max reconnection attempts reached")
                        break
                    continue
                time.sleep(0.1)
                continue
            
            last_frame_time = current_time
            reconnect_attempts = 0  # Reset reconnect attempts on successful frame
            
            # Process frame with improved face detector
            face_mesh_results, _ = face_detector.detect(frame)
            
            # Create display frame
            display_frame = np.zeros((sh, sw, 3), dtype=np.uint8)
            
            # Draw current target
            target_x, target_y = target_points[current_target]
            cv2.circle(display_frame, (target_x, target_y), target_radius, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Target {current_target + 1}/5", 
                       (target_x - 50, target_y - target_radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if face_mesh_results.multi_face_landmarks:
                landmarks = face_mesh_results.multi_face_landmarks[0].landmark
                
                # Convert normalized coordinates to pixel coordinates
                for landmark in landmarks:
                    landmark.x *= w
                    landmark.y *= h
                    landmark.z *= w
                
                # Calculate eye aspect ratios for blink detection
                left_ear = calculate_eye_aspect_ratio(landmarks, LEFT_EYE_INDICES)
                right_ear = calculate_eye_aspect_ratio(landmarks, RIGHT_EYE_INDICES)
                ear = (left_ear + right_ear) / 2.0
                
                # Update blink detection
                blink_detected = blink_detector.update(ear)
                
                # Update cursor pause state
                cursor_paused = blink_detector.is_blinking
                
                # Display blink detection debug info with more details
                debug_text = [
                    f"EAR: {ear:.3f} (L: {left_ear:.3f}, R: {right_ear:.3f})",
                    f"Blink Count: {blink_detector.debug_info['blink_count']}",
                    f"Last Blink Duration: {blink_detector.debug_info['last_blink_duration']:.3f}s",
                    f"Blink Status: {'Blinking' if blink_detector.is_blinking else 'Open'}",
                    f"Threshold: {blink_detector.blink_threshold:.2f}",
                    f"Cursor: {'Paused' if cursor_paused else 'Moving'}"
                ]
                
                for i, text in enumerate(debug_text):
                    cv2.putText(display_frame, text, (10, 30 + i*30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Update gaze position and move cursor only if not blinking
                ft, _ = gaze_estimator.extract_features(frame)
                if ft is not None and not cursor_paused:
                    # Ensure feature vector is 2D
                    if ft.ndim == 1:
                        ft = ft.reshape(1, -1)
                    raw_x, raw_y = gaze_estimator.predict(ft)[0]
                    gaze_x, gaze_y = gaze_smoother.update(raw_x, raw_y)
                    
                    # Store last valid gaze position
                    last_gaze_x, last_gaze_y = gaze_x, gaze_y
                    
                    # Draw gaze point
                    cv2.circle(display_frame, (int(gaze_x), int(gaze_y)), 5, (0, 0, 255), -1)
                    
                    # Move mouse cursor
                    try:
                        pyautogui.moveTo(int(gaze_x), int(gaze_y))
                    except Exception as e:
                        print(f"Mouse movement error: {str(e)}")
                elif cursor_paused and last_gaze_x is not None and last_gaze_y is not None:
                    # Draw paused cursor position
                    cv2.circle(display_frame, (int(last_gaze_x), int(last_gaze_y)), 5, (0, 255, 255), -1)
                    cv2.putText(display_frame, "PAUSED", (int(last_gaze_x) + 10, int(last_gaze_y)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if blink_detected:
                    print(f"Blink detected! EAR: {ear:.3f}")
                    current_time = time.time()
                    if can_click and current_time - last_click_time >= click_cooldown:
                        # Get current gaze position
                        ft, _ = gaze_estimator.extract_features(frame)
                        if ft is not None:
                            # Ensure feature vector is 2D
                            if ft.ndim == 1:
                                ft = ft.reshape(1, -1)
                            raw_x, raw_y = gaze_estimator.predict(ft)[0]
                            gaze_x, gaze_y = gaze_smoother.update(raw_x, raw_y)
                            
                            # Calculate distance to target
                            distance = np.sqrt((gaze_x - target_x)**2 + (gaze_y - target_y)**2)
                            accuracy_data.append(distance)
                            
                            # Move to next target if close enough
                            if distance < target_radius:
                                current_target = (current_target + 1) % len(target_points)
                                if current_target == 0:
                                    # Calculate and display average accuracy
                                    avg_accuracy = np.mean(accuracy_data)
                                    print(f"\nAverage accuracy: {avg_accuracy:.2f} pixels")
                                    accuracy_data = []
                            
                            # Perform mouse click
                            try:
                                pyautogui.click(int(gaze_x), int(gaze_y))
                            except pyautogui.FailSafeException:
                                print("Mouse movement blocked by fail-safe")
                            last_click_time = current_time
                            can_click = False
                
                # Draw face mesh using the improved detector
                face_detector.mp_drawing.draw_landmarks(
                    image=display_frame,
                    landmark_list=face_mesh_results.multi_face_landmarks[0],
                    connections=face_detector.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=face_detector.drawing_spec,
                    connection_drawing_spec=face_detector.drawing_spec
                )
                
                # Display status and metrics
                status = "Blink to click" if can_click else "Opening eyes..."
                cv2.putText(display_frame, status, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Display current accuracy if available
                if accuracy_data:
                    current_accuracy = np.mean(accuracy_data[-5:])  # Last 5 measurements
                    cv2.putText(display_frame, f"Current Accuracy: {current_accuracy:.1f}px",
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Update click state
                if not can_click and ear > blink_detector.blink_threshold:
                    can_click = True
            else:
                # No face detected
                cv2.putText(display_frame, "No face detected", (sw//2 - 100, sh//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Show the frame
            cv2.imshow("Accuracy Demo", display_frame)
            
            # Check for exit
            if cv2.waitKey(1) == 27:  # ESC key
                break
    
    except Exception as e:
        print(f"Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        
        # Print final accuracy statistics
        if accuracy_data:
            print("\nFinal accuracy statistics:")
            print(f"Average accuracy: {np.mean(accuracy_data):.2f} pixels")
            print(f"Standard deviation: {np.std(accuracy_data):.2f} pixels")
            print(f"Min accuracy: {np.min(accuracy_data):.2f} pixels")
            print(f"Max accuracy: {np.max(accuracy_data):.2f} pixels")

if __name__ == "__main__":
    run_accuracy_demo() 