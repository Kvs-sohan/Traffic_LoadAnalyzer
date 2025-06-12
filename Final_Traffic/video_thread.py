import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel

class VideoThread(QThread):
    frame_ready = pyqtSignal(int, np.ndarray)  # Send raw frame for processing
    
    def __init__(self, video_path, signal_idx, detector, areas=None, get_signal_state_func=None, get_current_signal_func=None, frame_skip=2):
        super().__init__()
        self.video_path = video_path
        self.signal_idx = signal_idx
        self.running = True
        self.cap = None
        self.detector = detector
        self.area = areas[signal_idx].copy() if areas and signal_idx < len(areas) else None
        self.frame_counter = 0
        self.current_frame = None
        self.target_size = (320, 240)  # Fixed target size for display
        self.get_signal_state_func = get_signal_state_func
        self.get_current_signal_func = get_current_signal_func
        self.original_width = 1280
        self.original_height = 720
        self.frame_skip = frame_skip
    
    def run(self):
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print(f"Error: Could not open video source for Signal {self.signal_idx}")
                return
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps < 1 or fps > 120:
                fps = 25
            frame_time = 1.0 / fps
            if self.area is not None:
                area_points = np.array(self.area, dtype=np.int32)
                print(f"Signal {self.signal_idx} using area points: {area_points}")
            else:
                print(f"No area points found for Signal {self.signal_idx}")
            while self.running:
                for _ in range(self.frame_skip - 1):
                    self.cap.grab()
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.current_frame = frame.copy()
                self.frame_ready.emit(self.signal_idx, frame.copy())  # Emit original frame
                self.msleep(int(frame_time * 1000))
        except Exception as e:
            print(f"Error in video thread {self.signal_idx}: {str(e)}")
        finally:
            if self.cap is not None:
                self.cap.release()
    
    def stop(self):
        self.running = False
        self.wait()  # Wait for the thread to finish
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def process_video_frame(self, signal_idx, frame):
        """Process video frame and update display with bboxes and vehicle count"""
        try:
            if not self.areas or signal_idx >= len(self.areas):
                display_frame = cv2.resize(frame, (320, 240))
                vehicle_count = 0
                traffic_weight = 0
                vehicle_counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0, 'bicycle': 0}
            else:
                # Run detection on the original frame (not resized)
                vehicle_count, traffic_weight, processed_frame, vehicle_counts = \
                    self.detector.detect_vehicles_in_area(frame, self.areas[signal_idx])
                # Update signal data
                if signal_idx < len(self.signals):
                    signal = self.signals[signal_idx]
                    signal.vehicle_count = vehicle_count
                    signal.traffic_weight = traffic_weight
                    signal.calculate_adaptive_green_time(vehicle_count, traffic_weight)
                display_frame = cv2.resize(processed_frame, (320, 240))
            
            # Convert to QImage and update QLabel
            rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            if signal_idx < len(self.video_labels) and self.video_labels[signal_idx]:
                self.video_labels[signal_idx].setPixmap(pixmap)
            
            # Update vehicle class counts in the UI
            vehicle_info_frame = self.findChild(QFrame, f'vehicle_info_{signal_idx}')
            if vehicle_info_frame:
                for class_name, count in vehicle_counts.items():
                    label = vehicle_info_frame.findChild(QLabel, f'{class_name}_label')
                    if label:
                        emoji = label.text().split()[0]
                        label.setText(f"{emoji} {count}")
            
            # Update vehicle count and weight label if present
            count_label = getattr(self, f'count_label_{signal_idx}', None)
            if count_label:
                count_label.setText(f"Vehicles: {vehicle_count} | Weight: {traffic_weight:.1f}")
        except Exception as e:
            print(f"Error processing frame for signal {signal_idx}: {e}") 