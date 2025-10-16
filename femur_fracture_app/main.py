import tkinter as tk
import os
from src.gui_components import FractureDetectionApp

def main():
    # Path ke model yang telah dilatih
    model_path = "C:/RayFile/KuliahBro/Semester8/TugasAkhir/Model/Femur/best.pt"  # Sesuaikan dengan path model Anda
    
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
