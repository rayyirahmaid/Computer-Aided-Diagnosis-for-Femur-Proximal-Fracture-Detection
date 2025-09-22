import tkinter as tk
import sys
import os
from src.gui_components import FractureDetectionApp
import re

def main():
    # Path ke model yang telah dilatih
    model_path = "femur_fracture_app/models/best.pt"  # Sesuaikan dengan path model Anda
    
    # Periksa apakah model ada
    if not os.path.exists(model_path):
        print(f"Error: Model tidak ditemukan di {model_path}")
        print("Silakan pastikan model YOLO Anda sudah dilatih dan tersimpan di path yang benar.")
        return
    
    # Buat root window
    root = tk.Tk()
    
    try:
        # Inisialisasi aplikasi
        app = FractureDetectionApp(root, model_path)
        
        # Jalankan aplikasi
        root.mainloop()
        
    except Exception as e:
        print(f"Error menjalankan aplikasi: {e}")
        return

if __name__ == "__main__":
    main()
