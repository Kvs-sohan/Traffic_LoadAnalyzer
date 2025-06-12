import cv2
import numpy as np
import json
import os

def load_areas_from_file(areas_file):
    """Load detection areas from JSON file."""
    try:
        if not os.path.exists(areas_file):
            print("⚠️ Areas file not found!")
            return None
        
        with open(areas_file, 'r') as f:
            loaded_areas = json.load(f)
        
        # Validate loaded areas
        if not isinstance(loaded_areas, list) or len(loaded_areas) != 4:
            print("⚠️ Invalid areas format - expected list of 4 areas")
            return None
        
        # Validate each area
        for i, area in enumerate(loaded_areas):
            if not isinstance(area, list) or len(area) != 4:
                print(f"⚠️ Invalid format for area {i} - expected list of 4 points")
                return None
            for point in area:
                if not isinstance(point, list) or len(point) != 2:
                    print(f"⚠️ Invalid point format in area {i} - expected [x, y]")
                    return None
                x, y = point
                if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                    print(f"⚠️ Invalid coordinates in area {i} - expected numbers")
                    return None
                
            # Convert to integer coordinates
            loaded_areas[i] = [[int(x), int(y)] for x, y in area]
            
            # Validate the area shape
            if not validate_area_shape(loaded_areas[i], i):
                print(f"⚠️ Invalid area shape for Signal {chr(65+i)}")
                return None
        
        return loaded_areas
            
    except Exception as e:
        print(f"⚠️ Error loading areas: {e}")
        return None

def validate_area_shape(points, signal_idx):
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
        print(f"Error in area validation: {str(e)}")
        return False

def calculate_efficiency_score(signal, vehicle_count, traffic_weight):
    """Calculate efficiency score for a traffic signal"""
    if signal.remaining_time > 0:
        throughput = vehicle_count / signal.remaining_time
    else:
        throughput = 0
    
    weighted_efficiency = throughput * (1 + traffic_weight * 0.1)
    efficiency = min(weighted_efficiency * 10, 100)
    
    return efficiency

def draw_detection_overlay(frame, vehicle_count, traffic_weight, vehicle_counts, area_points=None):
    """Draw detection overlay on frame with vehicle counts and traffic weight"""
    processed_frame = frame.copy()
    
    # Draw detection area if provided
    if area_points is not None:
        area_points_np = np.array(area_points, dtype=np.int32)
        cv2.polylines(processed_frame, [area_points_np], True, (0, 255, 255), 3)
        cv2.putText(processed_frame, "Detection Area", 
                   (area_points_np[0][0], area_points_np[0][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Add detection info with background
    info_bg_color = (0, 0, 0)
    info_text_color = (255, 255, 255)
    
    # Vehicle count background
    cv2.rectangle(processed_frame, (10, 10), (150, 35), info_bg_color, -1)
    cv2.putText(processed_frame, f"Vehicles: {vehicle_count}", 
               (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_text_color, 2)
    
    # Traffic weight background
    cv2.rectangle(processed_frame, (10, 40), (200, 65), info_bg_color, -1)
    cv2.putText(processed_frame, f"Traffic Weight: {traffic_weight:.1f}", 
               (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_text_color, 2)
    
    # Vehicle type counts
    y_offset = 90
    for vehicle_type, count in vehicle_counts.items():
        cv2.rectangle(processed_frame, (10, y_offset), (200, y_offset + 25), info_bg_color, -1)
        cv2.putText(processed_frame, f"{vehicle_type.capitalize()}: {count}", 
                   (15, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_text_color, 2)
        y_offset += 30
    
    return processed_frame 