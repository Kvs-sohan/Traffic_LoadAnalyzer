import os
import cv2
import numpy as np
import json
import time
from datetime import datetime
import threading
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QFrame, QMessageBox, QDialog, QLineEdit,
                           QGridLayout, QGroupBox, QTextEdit, QScrollArea, QCheckBox,
                           QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar)
from PyQt5.QtCore import Qt, QTimer, QMetaObject, Q_ARG, QRect, QPoint
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont, QPainter, QPen, QBrush

from detector import EnhancedVehicleDetector
from traffic_signal import EnhancedTrafficSignal
from traffic_database import TrafficDatabase
from video_thread import VideoThread
from helpers import load_areas_from_file, validate_area_shape, calculate_efficiency_score

class EnhancedTrafficManagementSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Smart Traffic Management System with YOLOv8")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize components
        self.detector = EnhancedVehicleDetector()
        self.signals = [EnhancedTrafficSignal(i) for i in range(4)]
        self.database = TrafficDatabase()
        
        # Thread safety
        self.signal_lock = threading.Lock()
        self.last_update_time = time.time()
        
        # Video threads
        self.video_threads = [None] * 4
        self.video_labels = [None] * 4
        
        # Configuration
        self.areas_file = "areas.json"
        self.config_file = "config.json"
        
        # Video sources - one for each signal
        self.video_sources = [
            os.path.abspath(r"video.mp4"),    # Signal A
            os.path.abspath(r"video2.mp4"),   # Signal B
            os.path.abspath(r"video3.mp4"),   # Signal C
            os.path.abspath(r"video4.mp4")    # Signal D
        ]
        
        # Verify video files exist
        for i, video_path in enumerate(self.video_sources):
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found for Signal {chr(65+i)}: {video_path}")
        
        self.areas = []
        self.is_running = False
        self.emergency_mode = False
        self.current_signal = 0
        self.cycle_start_time = time.time()
        
        self.analytics_data = {
            'timestamps': deque(maxlen=100),
            'vehicle_counts': [deque(maxlen=100) for _ in range(4)],
            'green_times': [deque(maxlen=100) for _ in range(4)]
        }
        
        # Thread-safe logging
        self.log_mutex = threading.Lock()
        
        self.setup_ui()
        self.load_config()
        self.load_areas()  # Load areas at startup if available

    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Header
        header = QLabel("🚦 Enhanced Smart Traffic Management System")
        header.setStyleSheet("""
            QLabel {
                background-color: #34495e;
                color: white;
                padding: 15px;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        main_layout.addWidget(header)
        
        # Control Panel
        control_panel = QGroupBox("System Controls")
        control_layout = QHBoxLayout()
        
        buttons = [
            ("🎯 New Areas", self.define_new_areas, "#3498db"),
            ("📁 Load Areas", self.load_areas, "#2ecc71"),
            ("▶️ Start System", self.start_system, "#27ae60"),
            ("⏹️ Stop System", self.stop_system, "#e74c3c"),
            ("🚨 Emergency Mode", self.toggle_emergency_mode, "#f39c12"),
            ("📊 Analytics", self.show_analytics, "#9b59b6"),
            ("⚙️ Settings", self.show_settings, "#34495e")
        ]
        
        for text, callback, color in buttons:
            btn = QPushButton(text)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    padding: 8px 15px;
                    border: none;
                    border-radius: 4px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {color}dd;
                }}
            """)
            btn.clicked.connect(callback)
            control_layout.addWidget(btn)
        
        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel)
        
        # Video Display Grid
        video_grid = QGridLayout()
        for i in range(2):
            for j in range(2):
                signal_idx = i * 2 + j
                signal_frame = QGroupBox(f"Signal {chr(65 + signal_idx)}")
                signal_layout = QVBoxLayout()
                
                # Video display
                video_label = QLabel()
                video_label.setMinimumSize(320, 240)
                video_label.setStyleSheet("background-color: black;")
                signal_layout.addWidget(video_label)
                
                # Add vehicle class information display
                vehicle_info_frame = QFrame()
                vehicle_info_frame.setObjectName(f"vehicle_info_{signal_idx}")
                vehicle_info_frame.setStyleSheet("""
                    QFrame {
                        background-color: #2c3e50;
                        border-radius: 4px;
                        padding: 5px;
                    }
                """)
                vehicle_info_layout = QHBoxLayout()
                vehicle_info_layout.setSpacing(5)
                vehicle_info_layout.setContentsMargins(5, 5, 5, 5)
                vehicle_info_layout.addStretch()
                vehicle_classes = {
                    'car': ('🚗', '#2ecc71'),
                    'truck': ('🚚', '#e74c3c'),
                    'bus': ('🚌', '#f1c40f'),
                    'motorcycle': ('🏍️', '#3498db'),
                    'bicycle': ('🚲', '#9b59b6')
                }
                for class_name, (emoji, color) in vehicle_classes.items():
                    label = QLabel(f"{emoji} 0")
                    label.setObjectName(f"{class_name}_label")
                    label.setStyleSheet(f"""
                        QLabel {{
                            background-color: {color};
                            color: white;
                            border-radius: 3px;
                            padding: 2px 18px;
                            font-weight: bold;
                            min-width: 60px;
                            text-align: center;
                        }}
                    """)
                    vehicle_info_layout.addWidget(label)
                vehicle_info_layout.addStretch()
                vehicle_info_frame.setLayout(vehicle_info_layout)
                vehicle_info_container = QWidget()
                vehicle_info_container.setFixedWidth(320)  # Match video width
                vehicle_info_container.setStyleSheet("background: transparent;")
                vehicle_info_layout = QHBoxLayout()
                vehicle_info_layout.setSpacing(5)
                vehicle_info_layout.setContentsMargins(0, 0, 0, 0)
                vehicle_info_layout.setAlignment(Qt.AlignCenter)
                vehicle_info_frame = QFrame()
                vehicle_info_frame.setObjectName(f"vehicle_info_{signal_idx}")
                vehicle_info_frame.setStyleSheet("""
                    QFrame {
                        background-color: #2c3e50;
                        border-radius: 4px;
                        padding: 5px;
                    }
                """)
                vehicle_bar_layout = QHBoxLayout()
                vehicle_bar_layout.setSpacing(5)
                vehicle_bar_layout.setContentsMargins(5, 5, 5, 5)
                vehicle_bar_layout.setAlignment(Qt.AlignCenter)
                vehicle_classes = {
                    'car': ('🚗', '#2ecc71'),
                    'truck': ('🚚', '#e74c3c'),
                    'bus': ('🚌', '#f1c40f'),
                    'motorcycle': ('🏍️', '#3498db'),
                    'bicycle': ('🚲', '#9b59b6')
                }
                for class_name, (emoji, color) in vehicle_classes.items():
                    label = QLabel(f"{emoji} 0")
                    label.setObjectName(f"{class_name}_label")
                    label.setStyleSheet(f"""
                        QLabel {{
                            background-color: {color};
                            color: white;
                            border-radius: 3px;
                            padding: 2px 18px;
                            font-weight: bold;
                            min-width: 60px;
                            text-align: center;
                        }}
                    """)
                    vehicle_bar_layout.addWidget(label)
                vehicle_info_frame.setLayout(vehicle_bar_layout)
                vehicle_info_layout.addWidget(vehicle_info_frame)
                vehicle_info_container.setLayout(vehicle_info_layout)
                # Center the bar under the video
                signal_layout.addWidget(vehicle_info_container, alignment=Qt.AlignHCenter)
                
                # Status indicators
                status_label = QLabel("🔴 RED")
                status_label.setStyleSheet("color: red; font-weight: bold;")
                signal_layout.addWidget(status_label)
                
                time_label = QLabel("Time: 0s")
                signal_layout.addWidget(time_label)
                
                count_label = QLabel("Vehicles: 0 | Weight: 0.0")
                signal_layout.addWidget(count_label)
                
                efficiency_label = QLabel("Efficiency: 0.0%")
                signal_layout.addWidget(efficiency_label)
                
                signal_frame.setLayout(signal_layout)
                video_grid.addWidget(signal_frame, i, j)
                
                # Store references
                self.video_labels[signal_idx] = video_label
                setattr(self, f'status_label_{signal_idx}', status_label)
                setattr(self, f'time_label_{signal_idx}', time_label)
                setattr(self, f'count_label_{signal_idx}', count_label)
                setattr(self, f'efficiency_label_{signal_idx}', efficiency_label)
        
        main_layout.addLayout(video_grid)
        
        # System Status
        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout()
        
        self.total_vehicles_label = QLabel("Total Vehicles: 0")
        self.efficiency_label = QLabel("System Efficiency: 0.0%")
        self.cycle_time_label = QLabel("Cycle Time: 0s")
        self.current_signal_label = QLabel("Active Signal: A")
        
        status_layout.addWidget(self.total_vehicles_label)
        status_layout.addWidget(self.efficiency_label)
        status_layout.addWidget(self.cycle_time_label)
        status_layout.addWidget(self.current_signal_label)
        
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # Status Log
        log_group = QGroupBox("System Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        # Setup timers
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_system)
        self.update_timer.start(100)  # Start update timer immediately
        
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(150)  # Update UI every 150ms

    def log_message(self, message):
        """Thread-safe logging"""
        try:
            with self.log_mutex:
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] {message}\n"
                # Use invokeMethod to update UI from any thread
                QMetaObject.invokeMethod(self.log_text, 
                                       "append",
                                       Qt.QueuedConnection,
                                       Q_ARG(str, log_entry))
                
                # Keep only last 100 messages
                text = self.log_text.toPlainText()
                lines = text.split('\n')
                if len(lines) > 100:
                    QMetaObject.invokeMethod(self.log_text,
                                           "setPlainText",
                                           Qt.QueuedConnection,
                                           Q_ARG(str, '\n'.join(lines[-100:])))
        except Exception as e:
            print(f"Logging error: {str(e)}")

    def start_system(self):
        # Reload areas before starting
        self.log_message("Starting system - loading areas...")
        if not self.load_areas():
            QMessageBox.warning(self, "Warning", "Could not load detection areas. Please check areas.json!")
            return
            
        if not self.areas:
            QMessageBox.warning(self, "Warning", "Please define detection areas first!")
            return

        if not self.is_running:
            self.is_running = True
            self.cycle_start_time = time.time()
            self.last_update_time = time.time()  # Reset last update time
            self.log_message("🚀 Traffic management system started")
            self.log_message(f"Using areas: {self.areas}")
            
            # Start video threads with fresh areas
            for i, video_path in enumerate(self.video_sources):
                if os.path.exists(video_path):
                    # Log the area being used for this signal
                    self.log_message(f"Setting up Signal {chr(65 + i)} with area: {self.areas[i]}")
                    
                    # Create a new thread with just this signal's area
                    self.video_threads[i] = VideoThread(
                        video_path, 
                        i, 
                        self.detector,
                        self.areas,  # Pass all areas
                        get_signal_state_func=lambda idx: self.signals[idx].current_state,
                        get_current_signal_func=lambda: self.current_signal
                    )
                    self.video_threads[i].frame_ready.connect(self.process_video_frame)
                    self.video_threads[i].start()
                    self.log_message(f"✅ Video thread started for Signal {chr(65 + i)}")
            
            # Set all signals to RED initially
            with self.signal_lock:
                for signal in self.signals:
                    signal.current_state = 'RED'
                    signal.remaining_time = 0
            
            self.current_signal = 0
            # Immediately run YOLO for signal A and set to GREEN
            self.run_initial_detection_for_signal(0)
            
            # Start system update timer (10 times per second)
            self.update_timer.start(100)
            
            QMessageBox.information(self, "System Started", 
                                  "Traffic management system is now running!")

    def run_initial_detection_for_signal(self, signal_idx):
        # Run YOLO for signal_idx, set to GREEN for calculated time
        def detection():
            time.sleep(0.2)
            frame = None
            if self.video_threads[signal_idx] and self.video_threads[signal_idx].isRunning():
                frame = self.video_threads[signal_idx].current_frame
            if frame is not None and signal_idx < len(self.areas):
                vehicle_count, traffic_weight, _, vehicle_type_counts = self.detector.detect_vehicles_in_area(frame, self.areas[signal_idx], draw_area=False)
                with self.signal_lock:
                    signal = self.signals[signal_idx]
                    green_time = signal.calculate_adaptive_green_time(vehicle_count, traffic_weight, datetime.now())
                    signal.vehicle_count = vehicle_count
                    signal.traffic_weight = traffic_weight
                    signal.vehicle_type_counts = vehicle_type_counts.copy()  # Store vehicle type counts
                    signal.current_state = 'GREEN'
                    signal.remaining_time = green_time
                    self.log_message(f"🟢 Signal {chr(65 + signal_idx)} → GREEN for {green_time}s (Vehicles: {vehicle_count}, Weight: {traffic_weight:.1f})")
            else:
                # Fallback
                with self.signal_lock:
                    signal = self.signals[signal_idx]
                    signal.current_state = 'GREEN'
                    signal.remaining_time = signal.default_green_time
                    self.log_message(f"🟢 Signal {chr(65 + signal_idx)} → GREEN for {signal.default_green_time}s (default)")
        threading.Thread(target=detection, daemon=True).start()

    def handle_signal_transitions(self, elapsed):
        with self.signal_lock:
            active_signal = self.signals[self.current_signal]
            # Update remaining time
            if active_signal.remaining_time > 0:
                active_signal.remaining_time = max(0, active_signal.remaining_time - elapsed)
                self.log_message(f"Timer update: Signal {chr(65 + self.current_signal)} - Remaining: {active_signal.remaining_time:.1f}s, Elapsed: {elapsed:.3f}s")
            
            # Handle state transitions
            if active_signal.remaining_time <= 0:
                if active_signal.current_state == 'GREEN':
                    # Turn GREEN to YELLOW
                    active_signal.current_state = 'YELLOW'
                    active_signal.remaining_time = active_signal.yellow_time
                    self.log_message(f"🟡 Signal {chr(65 + self.current_signal)} → YELLOW")
                    # During yellow, run YOLO for next signal
                    next_signal_idx = (self.current_signal + 1) % 4
                    self.run_detection_for_next_signal(next_signal_idx)
                
                elif active_signal.current_state == 'YELLOW':
                    # Turn YELLOW to RED, next signal to GREEN
                    active_signal.current_state = 'RED'
                    active_signal.remaining_time = 0
                    next_signal_idx = (self.current_signal + 1) % 4
                    next_signal = self.signals[next_signal_idx]
                    
                    # Set next signal to GREEN for its calculated time
                    if hasattr(next_signal, 'pending_green_time') and next_signal.pending_green_time > 0:
                        next_signal.current_state = 'GREEN'
                        next_signal.remaining_time = next_signal.pending_green_time
                        self.log_message(f"🟢 Signal {chr(65 + next_signal_idx)} → GREEN for {next_signal.pending_green_time}s")
                        next_signal.pending_green_time = 0
                    else:
                        # Fallback
                        next_signal.current_state = 'GREEN'
                        next_signal.remaining_time = next_signal.default_green_time
                        self.log_message(f"🟢 Signal {chr(65 + next_signal_idx)} → GREEN for {next_signal.default_green_time}s (default)")
                    
                    self.current_signal = next_signal_idx
                    self.cycle_start_time = time.time()  # Update cycle start time
                
                elif active_signal.current_state == 'RED':
                    active_signal.remaining_time = 0

    def calculate_efficiency_score(self, signal_idx, vehicle_count, traffic_weight):
        if signal_idx >= len(self.signals):
            return 0.0
        
        signal = self.signals[signal_idx]
        
        if signal.remaining_time > 0:
            throughput = vehicle_count / signal.remaining_time
        else:
            throughput = 0
        
        weighted_efficiency = throughput * (1 + traffic_weight * 0.1)
        efficiency = min(weighted_efficiency * 10, 100)
        
        return efficiency

    def update_analytics(self):
        current_time = datetime.now()
        self.analytics_data['timestamps'].append(current_time)
        
        # Update analytics data for each signal
        for i in range(4):
            signal = self.signals[i]
            
            # Update vehicle counts with current data
            self.analytics_data['vehicle_counts'][i].append(signal.vehicle_count)
            
            # Update green times based on current state and calculated times
            if signal.current_state == 'GREEN':
                green_time = signal.calculated_green_time if hasattr(signal, 'calculated_green_time') else signal.default_green_time
            else:
                green_time = signal.pending_green_time if hasattr(signal, 'pending_green_time') and signal.pending_green_time > 0 else signal.default_green_time
            
            self.analytics_data['green_times'][i].append(green_time)

    def show_analytics(self):
        analytics_dialog = QDialog(self)
        analytics_dialog.setWindowTitle("Traffic Analytics Dashboard")
        analytics_dialog.setGeometry(200, 200, 1000, 800)
        
        layout = QVBoxLayout()
        
        # Create matplotlib figure with adjusted layout
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        
        fig.suptitle('Traffic Analytics Dashboard', fontsize=16, fontweight='bold', y=0.95)
        
        # Add matplotlib canvas to dialog
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        def format_timedelta(x, pos):
            if len(self.analytics_data['timestamps']) > int(x):
                td = self.analytics_data['timestamps'][int(x)] - self.analytics_data['timestamps'][0]
                return f"{int(td.total_seconds())}s"
            return ""
        
        def update_plots():
            # Clear all axes
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
            
            # Vehicle count trends
            times = list(range(len(self.analytics_data['timestamps'])))
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f']
            
            # Plot vehicle counts
            ax1.set_title('Vehicle Count Trends', fontsize=12, fontweight='bold', pad=10)
            ax1.set_xlabel('Time (seconds)', fontsize=10)
            ax1.set_ylabel('Number of Vehicles', fontsize=10)
            
            for i in range(4):
                data = list(self.analytics_data['vehicle_counts'][i])
                if data:
                    line = ax1.plot(times, data, 
                                  label=f'Signal {chr(65+i)}',
                                  color=colors[i],
                                  marker='o',
                                  markersize=4,
                                  linewidth=2)
            
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_timedelta))
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
            # Green time optimization
            ax2.set_title('Green Time Optimization', fontsize=12, fontweight='bold', pad=10)
            ax2.set_xlabel('Time (seconds)', fontsize=10)
            ax2.set_ylabel('Green Time Duration (s)', fontsize=10)
            
            for i in range(4):
                data = list(self.analytics_data['green_times'][i])
                if data:
                    line = ax2.plot(times, data,
                                  label=f'Signal {chr(65+i)}',
                                  color=colors[i],
                                  marker='s',
                                  markersize=4,
                                  linewidth=2)
            
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_timedelta))
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
            # Current Signal Efficiencies
            efficiency_scores = []
            signal_labels = []
            bar_colors = []
            
            for i in range(4):
                score = self.calculate_efficiency_score(i, self.signals[i].vehicle_count, self.signals[i].traffic_weight)
                efficiency_scores.append(score)
                signal_labels.append(f'Signal {chr(65+i)}')
                if score >= 80:
                    bar_colors.append('#2ecc71')  # Green
                elif score >= 60:
                    bar_colors.append('#f1c40f')  # Yellow
                else:
                    bar_colors.append('#e74c3c')  # Red
            
            ax3.set_title('Current Signal Efficiencies', fontsize=12, fontweight='bold', pad=10)
            ax3.set_ylabel('Efficiency (%)', fontsize=10)
            
            bars = ax3.bar(signal_labels, efficiency_scores, color=bar_colors)
            ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom',
                        fontsize=10)
            
            # Traffic distribution pie chart
            vehicle_counts = [signal.vehicle_count for signal in self.signals]
            total_vehicles = sum(vehicle_counts)
            
            ax4.set_title('Current Traffic Distribution', fontsize=12, fontweight='bold', pad=10)
            
            if total_vehicles > 0:
                traffic_distribution = [count/total_vehicles * 100 for count in vehicle_counts]
                
                wedges, texts, autotexts = ax4.pie(traffic_distribution, 
                                                 labels=[f'Signal {chr(65+i)}\n({count} vehicles)' 
                                                        for i, count in enumerate(vehicle_counts)],
                                                 colors=colors,
                                                 autopct='%1.1f%%',
                                                 pctdistance=0.85)
                
                # Add total vehicles in center
                ax4.text(0, 0, f'Total\n{total_vehicles}\nVehicles',
                        ha='center', va='center',
                        fontsize=12, fontweight='bold')
                
                plt.setp(autotexts, size=9, weight="bold")
                plt.setp(texts, size=10)
            else:
                ax4.text(0.5, 0.5, 'No vehicles detected', 
                        ha='center', va='center',
                        fontsize=12)
            
            # Adjust layout to prevent overlapping
            plt.tight_layout(rect=[0, 0, 0.9, 0.95])
            canvas.draw()
        
        # Initial plot
        update_plots()
        
        # Add control buttons
        button_layout = QHBoxLayout()
        
        # Update button
        update_btn = QPushButton("Update Analytics")
        update_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                padding: 8px 15px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        update_btn.clicked.connect(update_plots)
        
        # Auto-update checkbox
        auto_update_check = QCheckBox("Auto Update")
        auto_update_check.setStyleSheet("""
            QCheckBox {
                color: #2c3e50;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        
        # Auto-update timer
        auto_update_timer = QTimer()
        auto_update_timer.timeout.connect(update_plots)
        
        def toggle_auto_update(state):
            if state:
                auto_update_timer.start(1000)  # Update every second
            else:
                auto_update_timer.stop()
        
        auto_update_check.stateChanged.connect(toggle_auto_update)
        
        # Add buttons to layout
        button_layout.addWidget(update_btn)
        button_layout.addWidget(auto_update_check)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        analytics_dialog.setLayout(layout)
        
        # --- Fix: Stop timer when dialog closes ---
        def cleanup():
            auto_update_timer.stop()
            auto_update_timer.deleteLater()
        analytics_dialog.finished.connect(cleanup)
        # --- End fix ---
        
        analytics_dialog.exec_()

    def show_settings(self):
        settings_dialog = QDialog(self)
        settings_dialog.setWindowTitle("System Settings")
        settings_dialog.setGeometry(300, 300, 500, 400)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Signal Timing Settings")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        # Settings for each signal
        min_entries = []
        max_entries = []
        
        for i in range(4):
            group = QGroupBox(f"Signal {chr(65+i)}")
            group_layout = QHBoxLayout()
            
            # Min green time
            min_label = QLabel("Min Green Time:")
            min_entry = QLineEdit()
            min_entry.setText(str(self.signals[i].min_green_time))
            min_entries.append(min_entry)
            
            # Max green time
            max_label = QLabel("Max Green Time:")
            max_entry = QLineEdit()
            max_entry.setText(str(self.signals[i].max_green_time))
            max_entries.append(max_entry)
            
            group_layout.addWidget(min_label)
            group_layout.addWidget(min_entry)
            group_layout.addWidget(max_label)
            group_layout.addWidget(max_entry)
            
            group.setLayout(group_layout)
            layout.addWidget(group)
        
        # Save button
        save_btn = QPushButton("Save Settings")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                padding: 8px 15px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #219a52;
            }
        """)
        
        def save_settings():
            try:
                for i in range(4):
                    self.signals[i].min_green_time = int(min_entries[i].text())
                    self.signals[i].max_green_time = int(max_entries[i].text())
                
                config = {}
                for i in range(4):
                    config[f'signal_{i}'] = {
                        'min_green_time': self.signals[i].min_green_time,
                        'max_green_time': self.signals[i].max_green_time
                    }
                
                with open(self.config_file, 'w') as f:
                    json.dump(config, f, indent=4)
                
                QMessageBox.information(settings_dialog, "Success", 
                                      "Settings saved successfully!")
                settings_dialog.accept()
                
            except ValueError:
                QMessageBox.critical(settings_dialog, "Error",
                                   "Please enter valid numbers for timing values")
        
        save_btn.clicked.connect(save_settings)
        layout.addWidget(save_btn)
        
        settings_dialog.setLayout(layout)
        settings_dialog.exec_()

    def define_new_areas(self):
        if not self.video_sources:
            QMessageBox.critical(self, "Error", "No video sources configured!")
            return
        
        # Verify all configured video files exist and are accessible
        missing_videos = []
        for video_file in self.video_sources:
            if not os.path.exists(video_file):
                missing_videos.append(video_file)
        
        if missing_videos:
            QMessageBox.critical(self, "Error", 
                "Missing required video files:\n" + "\n".join(missing_videos) +
                "\n\nPlease ensure all video files are present and accessible.")
            return
        
        # Test each video file
        for video_path in self.video_sources:
            try:
                test_cap = cv2.VideoCapture(video_path)
                if not test_cap.isOpened():
                    QMessageBox.critical(self, "Error", f"Cannot open video file: {video_path}")
                    return
                # Read a test frame
                ret, frame = test_cap.read()
                if not ret:
                    QMessageBox.critical(self, "Error", f"Cannot read frames from: {video_path}")
                    return
                test_cap.release()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error testing video file {video_path}: {str(e)}")
                return
        
        # Update video sources with verified files
        self.video_sources = self.video_sources.copy()
        
        # Log the video sources
        self.log_message("Using video sources:")
        for i, video in enumerate(self.video_sources):
            self.log_message(f"Signal {chr(65+i)}: {video}")
        
        area_dialog = QDialog(self)
        area_dialog.setWindowTitle("Define Detection Areas")
        area_dialog.setGeometry(200, 200, 800, 600)
        
        layout = QVBoxLayout()
        
        # Debug info label
        debug_label = QLabel("")
        debug_label.setStyleSheet("color: #e74c3c; font-size: 10px;")
        layout.addWidget(debug_label)
        
        # Instructions with signal-specific guidance
        instructions = QLabel(
            "Click to define 4 corners of detection area for each signal:\n"
            "Signal A & C: Define points from top to bottom\n"
            "Press 'Next Signal' when done with current signal\n"
            "Points Selected: 0/4"
        )
        instructions.setStyleSheet("font-size: 12px; color: #2c3e50; padding: 10px;")
        layout.addWidget(instructions)
        
        # Current signal indicator with specific instructions
        current_signal_label = QLabel("Defining area for Signal A\nDefine points from top-left to bottom-right")
        current_signal_label.setStyleSheet("color: #e74c3c; font-size: 14px; font-weight: bold; padding: 5px;")
        layout.addWidget(current_signal_label)
        
        # Video display
        video_frame = QFrame()
        video_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        video_frame.setLineWidth(2)
        video_layout = QVBoxLayout()
        
        class ClickableLabel(QLabel):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setMouseTracking(True)
                self.points = []
                self.display_size = [640, 480]
                self.setMinimumSize(640, 480)
            
            def mousePressEvent(self, event):
                if event.button() == Qt.LeftButton and len(self.points) < 4:
                    pos = event.pos()
                    if self.pixmap():
                        # Get the actual image rect
                        img_rect = self.get_image_rect()
                        if img_rect.contains(pos):
                            # Convert coordinates from widget space to image space
                            x = int((pos.x() - img_rect.x()) / img_rect.width() * self.display_size[0])
                            y = int((pos.y() - img_rect.y()) / img_rect.height() * self.display_size[1])
                            
                            # Validate point is not too close to existing points
                            too_close = False
                            for p in self.points:
                                dx = x - p[0]
                                dy = y - p[1]
                                if (dx * dx + dy * dy) < 100:  # Points too close together
                                    too_close = True
                                    break
                            
                            if not too_close:
                                point = (x, y)
                                self.points.append(point)
                                debug_label.setText(f"Added point {len(self.points)} at ({point[0]}, {point[1]})")
                                instructions.setText(
                                    "Click to define 4 corners of detection area for each signal:\n"
                                    "Signal A & C: Define points from top to bottom\n"
                                    "Press 'Next Signal' when done with current signal\n"
                                    f"Points Selected: {len(self.points)}/4"
                                )
                                self.update()
                            else:
                                debug_label.setText("Point too close to existing point. Please choose a different location.")
            
            def paintEvent(self, event):
                super().paintEvent(event)
                if not self.pixmap() or not self.points:
                    return
                    
                painter = QPainter(self)
                img_rect = self.get_image_rect()
                
                # Draw points and lines
                for i, point in enumerate(self.points):
                    # Convert image coordinates to widget coordinates
                    x = int(point[0] * img_rect.width() / self.display_size[0] + img_rect.x())
                    y = int(point[1] * img_rect.height() / self.display_size[1] + img_rect.y())
                    
                    # Draw point
                    painter.setPen(QPen(Qt.white, 3))  # White outline
                    painter.setBrush(QBrush(Qt.red))
                    painter.drawEllipse(x - 6, y - 6, 12, 12)
                    
                    # Draw number
                    painter.setPen(QPen(Qt.white, 2))
                    painter.setFont(QFont("Arial", 12, QFont.Bold))
                    painter.drawText(x - 8, y - 8, 16, 16, Qt.AlignCenter, str(i + 1))
                    
                    # Draw lines
                    if i > 0:
                        prev_x = int(self.points[i-1][0] * img_rect.width() / self.display_size[0] + img_rect.x())
                        prev_y = int(self.points[i-1][1] * img_rect.height() / self.display_size[1] + img_rect.y())
                        painter.setPen(QPen(QColor(255, 255, 0), 2))  # Yellow lines
                        painter.drawLine(prev_x, prev_y, x, y)
                
                # Close the polygon if all points are selected
                if len(self.points) == 4:
                    first_x = int(self.points[0][0] * img_rect.width() / self.display_size[0] + img_rect.x())
                    first_y = int(self.points[0][1] * img_rect.height() / self.display_size[1] + img_rect.y())
                    last_x = int(self.points[-1][0] * img_rect.width() / self.display_size[0] + img_rect.x())
                    last_y = int(self.points[-1][1] * img_rect.height() / self.display_size[1] + img_rect.y())
                    painter.setPen(QPen(QColor(255, 255, 0), 2))
                    painter.drawLine(last_x, last_y, first_x, first_y)
            
            def get_image_rect(self):
                if self.pixmap():
                    # Get the scaled pixmap size
                    scaled_size = self.pixmap().size()
                    scaled_size.scale(self.size(), Qt.KeepAspectRatio)
                    
                    # Calculate position to center the image
                    x = (self.width() - scaled_size.width()) // 2
                    y = (self.height() - scaled_size.height()) // 2
                    
                    return QRect(x, y, scaled_size.width(), scaled_size.height())
                return QRect()
            
            def clear_points(self):
                self.points.clear()
                instructions.setText(
                    "Click to define 4 corners of detection area for each signal:\n"
                    "Signal A & C: Define points from top to bottom\n"
                    "Press 'Next Signal' when done with current signal\n"
                    "Points Selected: 0/4"
                )
                debug_label.setText("Points cleared")
                self.update()
        
        # Create and setup video label
        video_label = ClickableLabel()
        video_label.setMinimumSize(640, 480)
        video_label.setStyleSheet("""
            QLabel {
                background-color: black;
                border: 2px solid #3498db;
                border-radius: 4px;
            }
        """)
        video_layout.addWidget(video_label, alignment=Qt.AlignCenter)
        video_frame.setLayout(video_layout)
        layout.addWidget(video_frame)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        next_btn = QPushButton("Next Signal")
        next_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px 15px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        
        clear_btn = QPushButton("Clear Points")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                padding: 8px 15px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                padding: 8px 15px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        
        button_layout.addWidget(next_btn)
        button_layout.addWidget(clear_btn)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        area_dialog.setLayout(layout)
        
        # State variables
        current_signal_idx = [0]
        all_areas = []
        current_cap = [None]
        original_frame = [None]
        scale_factor = [1.0, 1.0]
        
        def init_video_capture():
            # Release previous capture if it exists
            if current_cap[0] is not None:
                current_cap[0].release()
                current_cap[0] = None
            
            if current_signal_idx[0] < len(self.video_sources):
                try:
                    # Get the correct video source for the current signal
                    video_path = self.video_sources[current_signal_idx[0]]
                    debug_label.setText(f"Opening video for Signal {chr(65+current_signal_idx[0])}: {video_path}")
                    
                    # Force reopen the video file
                    if os.path.exists(video_path):
                        # Try to release any system handles to the video
                        import gc
                        gc.collect()
                        
                        # Wait a bit to ensure file handle is released
                        time.sleep(0.1)
                        
                        cap = cv2.VideoCapture(video_path)
                        if not cap.isOpened():
                            debug_label.setText(f"Failed to open video: {video_path}")
                            QMessageBox.critical(area_dialog, "Error", f"Failed to open video: {video_path}")
                            return False
                        
                        current_cap[0] = cap
                        
                        # Set capture properties
                        current_cap[0].set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        current_cap[0].set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        current_cap[0].set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
                        current_cap[0].set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
                        
                        # Read first frame
                        ret, frame = current_cap[0].read()
                        if not ret or frame is None:
                            debug_label.setText(f"Failed to read frame from: {video_path}")
                            QMessageBox.critical(area_dialog, "Error", f"Failed to read frame from: {video_path}")
                            return False
                        
                        original_frame[0] = frame.copy()
                        h, w = frame.shape[:2]
                        scale_factor[0] = w / video_label.display_size[0]
                        scale_factor[1] = h / video_label.display_size[1]
                        
                        # Reset video label points when switching videos
                        video_label.clear_points()
                        
                        debug_label.setText(f"Successfully opened video for Signal {chr(65+current_signal_idx[0])}: {video_path} ({w}x{h})")
                        return True
                    else:
                        debug_label.setText(f"Video file not found: {video_path}")
                        QMessageBox.critical(area_dialog, "Error", f"Video file not found: {video_path}")
                        return False
                    
                except Exception as e:
                    error_msg = f"Error opening video source: {str(e)}"
                    debug_label.setText(error_msg)
                    QMessageBox.critical(area_dialog, "Error", error_msg)
                    return False
            return False
        
        def update_preview():
            if current_cap[0] is not None and current_cap[0].isOpened():
                try:
                    # Get the current video source
                    current_video = self.video_sources[current_signal_idx[0]]
                    debug_label.setText(f"Previewing Signal {chr(65+current_signal_idx[0])} - {current_video}")
                    
                    ret, frame = current_cap[0].read()
                    if not ret or frame is None:
                        # Try to reopen the video if we can't read a frame
                        current_cap[0].release()
                        current_cap[0] = cv2.VideoCapture(current_video)
                        ret, frame = current_cap[0].read()
                        if not ret or frame is None:
                            debug_label.setText(f"Failed to read frame from {current_video}")
                            return
                    
                    original_frame[0] = frame.copy()
                    display_frame = cv2.resize(frame, (video_label.display_size[0], video_label.display_size[1]))
                    
                    # Add text overlay showing which signal and video
                    cv2.putText(display_frame, 
                              f"Signal {chr(65+current_signal_idx[0])} - {current_video}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Convert to QImage and display
                    rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    
                    # Scale pixmap to fit label while maintaining aspect ratio
                    scaled_pixmap = pixmap.scaled(video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    video_label.setPixmap(scaled_pixmap)
                    
                except Exception as e:
                    debug_label.setText(f"Error in update_preview: {str(e)}")
            
            if area_dialog.isVisible():
                QTimer.singleShot(30, update_preview)  # Update every 30ms
        
        def init_next_video():
            if not init_video_capture():
                QMessageBox.critical(area_dialog, "Error",
                                f"Could not open video source for Signal {chr(65+current_signal_idx[0])}")
                area_dialog.reject()
            else:
                update_preview()

        def next_signal():
            if len(video_label.points) == 4:
                # Get original video size for this signal
                video_path = self.video_sources[current_signal_idx[0]]
                cap = cv2.VideoCapture(video_path)
                ret, frame = cap.read()
                cap.release()
                if not ret or frame is None:
                    QMessageBox.critical(area_dialog, "Error", f"Could not read frame from {video_path}")
                    return
                original_height, original_width = frame.shape[:2]
                display_width, display_height = video_label.display_size
                video_points = []
                for x, y in video_label.points:
                    video_x = int((x / display_width) * original_width)
                    video_y = int((y / display_height) * original_height)
                    video_points.append([video_x, video_y])
                
                if not self.validate_area_shape(video_points, current_signal_idx[0]):
                    QMessageBox.warning(area_dialog, "Invalid Area",
                                    "Please define a valid area (points should form a proper polygon)")
                    return
                
                all_areas.append(video_points)
                self.log_message(f"Added area for Signal {chr(65+current_signal_idx[0])}: {video_points}")
                video_label.clear_points()
                current_signal_idx[0] += 1
                
                if current_signal_idx[0] < 4:
                    current_signal_label.setText(f"Defining area for Signal {chr(65+current_signal_idx[0])}\n" +
                                            ("Define points from top to bottom" if current_signal_idx[0] in [0, 2] else ""))
                    
                    if current_cap[0] is not None:
                        current_cap[0].release()
                        current_cap[0] = None
                    QTimer.singleShot(500, lambda: init_next_video())
                else:
                    try:
                        with open(self.areas_file, 'w') as f:
                            json.dump(all_areas, f, indent=4)
                        self.log_message("Areas saved to file successfully")
                        QMessageBox.information(area_dialog, "Success",
                                            "All detection areas saved successfully!")
                        self.areas = all_areas.copy()
                        area_dialog.accept()
                        self.log_message("New detection areas defined and saved")
                    except Exception as e:
                        self.log_message(f"Error saving areas: {e}")
                        QMessageBox.critical(area_dialog, "Error",
                                        f"Error saving areas: {e}")
            else:
                QMessageBox.warning(area_dialog, "Incomplete",
                                "Please define all 4 corners before proceeding.")
        
        def cleanup_resources():
            if current_cap[0] is not None:
                current_cap[0].release()
                current_cap[0] = None

        area_dialog.finished.connect(lambda: cleanup_resources())
        
        # Connect signals
        next_btn.clicked.connect(next_signal)
        clear_btn.clicked.connect(lambda: video_label.clear_points())
        cancel_btn.clicked.connect(lambda: area_dialog.reject())
        
        # Initialize first video capture
        if not init_video_capture():
            QMessageBox.critical(area_dialog, "Error", "Could not open first video source!")
            return
        
        # Start video preview
        update_preview()
        
        # Show dialog
        area_dialog.exec_()
        
        # Cleanup
        if current_cap[0] is not None:
            current_cap[0].release()

    def load_areas(self):
        """Load detection areas from JSON file."""
        try:
            if not os.path.exists(self.areas_file):
                self.log_message("⚠️ Areas file not found!")
                QMessageBox.warning(self, "Warning", "Areas file not found. Please define new areas first.")
                return False
            
            with open(self.areas_file, 'r') as f:
                loaded_areas = json.load(f)
            
            # Validate loaded areas
            if not isinstance(loaded_areas, list) or len(loaded_areas) != 4:
                self.log_message("⚠️ Invalid areas format - expected list of 4 areas")
                QMessageBox.warning(self, "Warning", "Invalid areas format in file. Please define new areas.")
                return False
            
            # Validate each area
            for i, area in enumerate(loaded_areas):
                if not isinstance(area, list) or len(area) != 4:
                    self.log_message(f"⚠️ Invalid format for area {i} - expected list of 4 points")
                    QMessageBox.warning(self, "Warning", f"Invalid format for Signal {chr(65+i)} area. Please define new areas.")
                    return False
                for point in area:
                    if not isinstance(point, list) or len(point) != 2:
                        self.log_message(f"⚠️ Invalid point format in area {i} - expected [x, y]")
                        QMessageBox.warning(self, "Warning", f"Invalid point format in Signal {chr(65+i)}. Please define new areas.")
                        return False
                    x, y = point
                    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                        self.log_message(f"⚠️ Invalid coordinates in area {i} - expected numbers")
                        QMessageBox.warning(self, "Warning", f"Invalid coordinates in Signal {chr(65+i)}. Please define new areas.")
                        return False
                    
                # Convert to integer coordinates
                loaded_areas[i] = [[int(x), int(y)] for x, y in area]
                
                # Validate the area shape
                if not self.validate_area_shape(loaded_areas[i], i):
                    self.log_message(f"⚠️ Invalid area shape for Signal {chr(65+i)}")
                    QMessageBox.warning(self, "Warning", f"Invalid area shape for Signal {chr(65+i)}. Please define new areas.")
                    return False
            
            # Stop any running threads before updating areas
            if self.is_running:
                self.stop_system()
            
            self.areas = loaded_areas
            self.log_message("✅ Areas loaded successfully")
            self.log_message(f"Loaded areas: {self.areas}")
            QMessageBox.information(self, "Success", "Areas loaded successfully!")
            return True
            
        except Exception as e:
            self.log_message(f"⚠️ Error loading areas: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load areas: {str(e)}")
            return False

    def validate_area_shape(self, points, signal_idx):
        """Validate the shape and size of an area"""
        try:
            if len(points) != 4:
                return False

            # Calculate area using shoelace formula
            area = 0
            for i in range(len(points)):
                j = (i + 1) % len(points)
                area += points[i][0] * points[j][1]
                area -= points[j][0] * points[i][1]
            area = abs(area) / 2

            # Only do basic validation for Signal A and D
            if signal_idx in [0, 3]:  # Signal A and D
                # Just ensure the area is not zero and points form a valid polygon
                return area > 0

            # More strict validation for other signals
            elif signal_idx == 2:  # Signal C
                # Validate trapezoid shape
                top_width = abs(points[1][0] - points[0][0])
                bottom_width = abs(points[3][0] - points[2][0])
                if top_width < 35 or bottom_width < 35:
                    return False
                
                # Check height
                height = max(p[1] for p in points) - min(p[1] for p in points)
                if height < 70:
                    return False

            # General validation - very minimal requirements
            min_area = 1000  # Greatly reduced minimum area
            if area < min_area:
                return False

            return True

        except Exception as e:
            self.log_message(f"Error in area validation: {str(e)}")
            return False

    def stop_system(self):
        """Stop the traffic management system"""
        if not self.is_running:
            return
            
        self.log_message("🛑 Stopping traffic management system...")
        self.is_running = False
        
        # Stop timers first
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
        if hasattr(self, 'ui_timer'):
            self.ui_timer.stop()
        
        # Stop video threads
        for i, thread in enumerate(self.video_threads):
            if thread:
                self.log_message(f"Stopping video thread for Signal {chr(65 + i)}...")
                thread.stop()
                thread.wait()  # Ensure thread is fully stopped
                self.video_threads[i] = None
        
        # Reset all signals
        for signal in self.signals:
            signal.current_state = 'RED'
            signal.remaining_time = 0
            signal.vehicle_count = 0
            signal.traffic_weight = 0
        
        # Clear video displays
        for video_label in self.video_labels:
            if video_label:
                video_label.clear()
                video_label.setText("Video Stopped")
                video_label.setStyleSheet("background-color: #2c3e50; color: white; font-size: 12px;")
        
        self.log_message("System stopped and reset.")
        QMessageBox.information(self, "System Stopped", "Traffic management system has been stopped.")

    def load_config(self):
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    for i, signal in enumerate(self.signals):
                        if f'signal_{i}' in config:
                            signal_config = config[f'signal_{i}']
                            signal.min_green_time = signal_config.get('min_green_time', 10)
                            signal.max_green_time = signal_config.get('max_green_time', 45)
        except Exception as e:
            self.log_message(f"Error loading config: {e}")

    def toggle_emergency_mode(self):
        self.emergency_mode = not self.emergency_mode
        status = "ACTIVATED" if self.emergency_mode else "DEACTIVATED"
        self.log_message(f"Emergency mode {status}")
        
        if self.emergency_mode:
            QMessageBox.information(
                self, "Emergency Mode",
                "Emergency mode activated!\nAll signals will prioritize emergency vehicles."
            )

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

    def update_system(self):
        if not self.is_running:
            return
            
        try:
            current_time = time.time()
            elapsed = current_time - self.last_update_time
            self.last_update_time = current_time
            
            # Ensure elapsed time is reasonable (not too large)
            if elapsed > 1.0:  # If more than 1 second has passed, cap it
                elapsed = 1.0
                self.log_message(f"Warning: Large time gap detected ({elapsed:.1f}s)")
            
            self.handle_signal_transitions(elapsed)
            self.update_analytics()
            
            # Force UI update for timers using QTimer.singleShot
            QTimer.singleShot(0, self.update_ui)
            
        except Exception as e:
            self.log_message(f"Error in system update: {str(e)}")

    def update_ui(self):
        if not self.is_running:
            return
            
        try:
            for i, signal in enumerate(self.signals):
                # Update signal status
                if signal.current_state == 'GREEN':
                    status_text = "🟢 GREEN"
                    color = 'green'
                elif signal.current_state == 'YELLOW':
                    status_text = "🟡 YELLOW"
                    color = 'orange'
                else:
                    status_text = "🔴 RED"
                    color = 'red'
                
                status_label = getattr(self, f'status_label_{i}')
                time_label = getattr(self, f'time_label_{i}')
                count_label = getattr(self, f'count_label_{i}')
                efficiency_label = getattr(self, f'efficiency_label_{i}')
                
                status_label.setText(status_text)
                status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
                
                # Format remaining time with one decimal place
                remaining_time = max(0, signal.remaining_time)
                time_label.setText(f"Time: {remaining_time:.1f}s")
                
                count_label.setText(
                    f"Vehicles: {signal.vehicle_count} | Weight: {signal.traffic_weight:.1f}"
                )
                
                efficiency = self.calculate_efficiency_score(
                    i, signal.vehicle_count, signal.traffic_weight
                )
                efficiency_label.setText(f"Efficiency: {efficiency:.1f}%")
            
            # Update system metrics
            total_vehicles = sum(signal.vehicle_count for signal in self.signals)
            self.total_vehicles_label.setText(f"Total Vehicles: {total_vehicles}")
            
            avg_efficiency = sum(
                self.calculate_efficiency_score(i, signal.vehicle_count, signal.traffic_weight)
                for i, signal in enumerate(self.signals)
            ) / 4
            self.efficiency_label.setText(f"System Efficiency: {avg_efficiency:.1f}%")
            
            cycle_time = int(time.time() - self.cycle_start_time)
            self.cycle_time_label.setText(f"Cycle Time: {cycle_time}s")
            
            self.current_signal_label.setText(f"Active Signal: {chr(65 + self.current_signal)}")
            
        except Exception as e:
            self.log_message(f"Error updating UI: {str(e)}")

    # ... [Rest of the EnhancedTrafficManagementSystem class methods from new.py] ... 