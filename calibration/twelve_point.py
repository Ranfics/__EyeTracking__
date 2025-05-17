import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

from eyetrax.calibration.common import (
    _pulse_and_capture,
    compute_grid_points,
    wait_for_face_and_countdown,
)
from eyetrax.utils.screen import get_screen_size

class GazeSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.x_buffer = deque(maxlen=window_size)
        self.y_buffer = deque(maxlen=window_size)
    
    def update(self, x, y):
        self.x_buffer.append(x)
        self.y_buffer.append(y)
        if len(self.x_buffer) < 2:
            return x, y
        return np.mean(self.x_buffer), np.mean(self.y_buffer)

def draw_face_mesh(frame, face_mesh, landmarks):
    """Draw detailed face mesh on the frame"""
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Get image dimensions
    h, w = frame.shape[:2]
    
    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Convert normalized coordinates to pixel coordinates
            for landmark in face_landmarks.landmark:
                landmark.x *= w
                landmark.y *= h
                landmark.z *= w  # Use width as reference for z
            
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
    
    return frame

def get_face_angle(face_mesh_results):
    """Calculate face angle from face mesh landmarks"""
    if not face_mesh_results.multi_face_landmarks:
        return None
    
    landmarks = face_mesh_results.multi_face_landmarks[0].landmark
    
    # Get key points for angle calculation
    nose_tip = landmarks[1]  # Nose tip
    left_eye = landmarks[33]  # Left eye outer corner
    right_eye = landmarks[263]  # Right eye outer corner
    
    # Calculate horizontal angle (yaw)
    eye_width = right_eye.x - left_eye.x
    yaw = np.arctan2(eye_width, 1.0) * 180 / np.pi
    
    # Calculate vertical angle (pitch)
    nose_height = nose_tip.y - (left_eye.y + right_eye.y) / 2
    pitch = np.arctan2(nose_height, 1.0) * 180 / np.pi
    
    return yaw, pitch

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

def run_12_point_calibration(gaze_estimator, camera_index: int = 0):
    """
    Enhanced 12-point calibration with face angle detection
    """
    sw, sh = get_screen_size()
    
    # Initialize face mesh with static image mode
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize gaze smoother
    gaze_smoother = GazeSmoother(window_size=5)
    
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
    
    # Variables for camera reconnection
    last_frame_time = time.time()
    max_frame_interval = 1.0  # Maximum time between frames
    reconnect_attempts = 0
    max_reconnect_attempts = 3
    
    try:
        if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 3):
            cap.release()
            cv2.destroyAllWindows()
            return
        
        # Define 12-point calibration grid
        order = [
            (0, 0), (0, 1), (0, 2),  # Top row
            (1, 0), (1, 1), (1, 2),  # Middle row
            (2, 0), (2, 1), (2, 2),  # Bottom row
            (0.5, 0.5), (0.5, 1.5), (1.5, 0.5)  # Additional points for better coverage
        ]
        
        pts = compute_grid_points(order, sw, sh)
        feats, targs = [], []
        face_angles = []
        
        # First phase: 12-point calibration
        for x, y in pts:
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            
            # Show target point
            cv2.circle(canvas, (x, y), 20, (0, 255, 0), -1)
            
            # Add visual guide
            cv2.putText(canvas, "Look at the green dot", (sw//2 - 150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Calibration", canvas)
            cv2.waitKey(1000)  # Give user time to prepare
            
            # Capture phase
            start_time = time.time()
            while time.time() - start_time < 2.0:  # 2 seconds per point
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
                            return
                        camera_index = new_camera_index
                        reconnect_attempts += 1
                        if reconnect_attempts >= max_reconnect_attempts:
                            print("Max reconnection attempts reached")
                            return
                        continue
                    time.sleep(0.1)
                    continue
                
                last_frame_time = current_time
                reconnect_attempts = 0  # Reset reconnect attempts on successful frame
                
                # Process frame with face mesh
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_mesh_results = face_mesh.process(frame_rgb)
                
                # Get face angle
                angle = get_face_angle(face_mesh_results)
                if angle is not None:
                    yaw, pitch = angle
                    face_angles.append([yaw, pitch])
                
                # Draw face mesh
                frame = draw_face_mesh(frame, face_mesh, face_mesh_results)
                
                # Extract features
                ft, blink = gaze_estimator.extract_features(frame)
                if ft is not None and not blink:
                    # Ensure feature vector is 2D
                    if ft.ndim == 1:
                        ft = ft.reshape(1, -1)
                    feats.append(ft[0])  # Take first row if 2D
                    targs.append([x, y])
                
                # Show feedback
                display_frame = frame.copy()
                if angle is not None:
                    cv2.putText(display_frame, f"Yaw: {yaw:.1f} Pitch: {pitch:.1f}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Calibration", display_frame)
                if cv2.waitKey(1) == 27:  # ESC key
                    cap.release()
                    cv2.destroyAllWindows()
                    return
        
        # Second phase: Face angle calibration
        cv2.putText(canvas, "Face Angle Calibration", (sw//2 - 150, sh//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(canvas, "Look at different angles", (sw//2 - 200, sh//2 + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Calibration", canvas)
        cv2.waitKey(2000)
        
        # Capture face angles at different positions
        angle_positions = [
            "Look straight ahead",
            "Look slightly left",
            "Look slightly right",
            "Look slightly up",
            "Look slightly down"
        ]
        
        for position in angle_positions:
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            cv2.putText(canvas, position, (sw//2 - 150, sh//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Calibration", canvas)
            cv2.waitKey(1000)
            
            start_time = time.time()
            while time.time() - start_time < 2.0:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_mesh_results = face_mesh.process(frame_rgb)
                angle = get_face_angle(face_mesh_results)
                
                if angle is not None:
                    yaw, pitch = angle
                    face_angles.append([yaw, pitch])
                    ft, blink = gaze_estimator.extract_features(frame)
                    if ft is not None and not blink:
                        # Ensure feature vector is 2D
                        if ft.ndim == 1:
                            ft = ft.reshape(1, -1)
                        feats.append(ft[0])  # Take first row if 2D
                        targs.append([sw//2, sh//2])  # Center point for angle calibration
                
                frame = draw_face_mesh(frame, face_mesh, face_mesh_results)
                cv2.imshow("Calibration", frame)
                if cv2.waitKey(1) == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    return
        
        cap.release()
        cv2.destroyAllWindows()
        
        if feats:
            # Convert lists to numpy arrays with proper shapes
            feats_array = np.array(feats)
            targs_array = np.array(targs)
            
            # Store face angles in the gaze estimator for later use
            gaze_estimator.face_angles = np.array(face_angles)
            
            # Train the model with all collected data
            gaze_estimator.train(feats_array, targs_array)

        # Update the feature extraction and prediction with smoothing
        if face_mesh_results.multi_face_landmarks:
            ft, blink = gaze_estimator.extract_features(frame)
            if ft is not None and not blink:
                # Ensure feature vector is 2D
                if ft.ndim == 1:
                    ft = ft.reshape(1, -1)
                raw_x, raw_y = gaze_estimator.predict(ft)[0]
                x, y = gaze_smoother.update(raw_x, raw_y)
                feats.append(ft[0])
                targs.append([x, y]) 

    except Exception as e:
        print(f"Error during calibration: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows() 