import sqlite3
from datetime import datetime

class TrafficDatabase:
    def __init__(self, db_path="traffic_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                signal_id INTEGER,
                vehicle_count INTEGER,
                traffic_weight REAL,
                green_time INTEGER,
                efficiency_score REAL
            )
        ''')
        conn.commit()
        conn.close()
    
    def log_traffic_data(self, signal_id, vehicle_count, traffic_weight, green_time, efficiency_score):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO traffic_logs 
            (timestamp, signal_id, vehicle_count, traffic_weight, green_time, efficiency_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.now(), signal_id, vehicle_count, traffic_weight, green_time, efficiency_score))
        conn.commit()
        conn.close() 