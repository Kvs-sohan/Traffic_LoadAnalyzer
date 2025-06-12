from collections import deque
from datetime import datetime
import time

class EnhancedTrafficSignal:
    def __init__(self, signal_id):
        self.signal_id = signal_id
        self.min_green_time = 10
        self.max_green_time = 45
        self.default_green_time = 15
        self.yellow_time = 3
        self.all_red_time = 2
        self.current_state = 'RED'
        self.remaining_time = 0
        self.vehicle_history = deque(maxlen=10)
        self.priority_mode = False
        self.last_detection_time = 0
        self.vehicle_count = 0
        self.traffic_weight = 0
        self.calculated_green_time = self.default_green_time
        self.is_active = False
        self.pending_green_time = 0  # Ensure this is always present
        self.last_update_time = time.time()
        # Add vehicle type counts
        self.vehicle_type_counts = {
            'car': 0,
            'truck': 0,
            'bus': 0,
            'motorcycle': 0,
            'bicycle': 0
        }

    def calculate_adaptive_green_time(self, vehicle_count, traffic_weight, time_of_day=None):
        self.vehicle_count = vehicle_count
        self.traffic_weight = traffic_weight
        
        base_time = self.min_green_time
        
        if traffic_weight > 0:
            density_time = min(traffic_weight * 3, self.max_green_time - base_time)
        else:
            density_time = 0
        
        time_factor = 1.0
        if time_of_day:
            hour = time_of_day.hour
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                time_factor = 1.2
            elif 22 <= hour or hour <= 6:
                time_factor = 0.8
        
        if len(self.vehicle_history) > 3:
            avg_traffic = sum(self.vehicle_history) / len(self.vehicle_history)
            if traffic_weight > avg_traffic * 1.5:
                time_factor *= 1.3
        
        calculated_time = int(base_time + (density_time * time_factor))
        self.vehicle_history.append(traffic_weight)
        
        self.calculated_green_time = max(self.min_green_time, min(calculated_time, self.max_green_time))
        return self.calculated_green_time 