import os
import json
from datetime import datetime

class ConfigManager:
    """
    Mengelola konfigurasi aplikasi
    """
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.default_config = {
            "model_path": "models/best.pt",
            "confidence_threshold": 0.5,
            "iou_threshold": 0.5,
            "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
            "max_image_size": (1024, 1024),
            "recent_files": []
        }
        self.config = self.load_config()
    
    def load_config(self):
        """Load konfigurasi dari file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge dengan default config
                for key, value in self.default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            else:
                return self.default_config.copy()
        except:
            return self.default_config.copy()
    
    def save_config(self):
        """Simpan konfigurasi ke file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key, default=None):
        """Ambil nilai konfigurasi"""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Set nilai konfigurasi"""
        self.config[key] = value
        self.save_config()

class Logger:
    """
    Simple logger untuk tracking aktivitas aplikasi
    """
    def __init__(self, log_file="app.log"):
        self.log_file = log_file
    
    def log(self, message, level="INFO"):
        """Log pesan dengan timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except:
            pass  # Ignore logging errors
        
        # Print ke console juga
        print(log_entry.strip())
    
    def info(self, message):
        self.log(message, "INFO")
    
    def warning(self, message):
        self.log(message, "WARNING")
    
    def error(self, message):
        self.log(message, "ERROR")
