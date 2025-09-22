import cv2
import numpy as np
from PIL import Image, ImageTk, ImageOps

class ImageProcessor:
    @staticmethod
    def load_image(image_path):
        """
        Load gambar dari path
        """
        try:
            image = Image.open(image_path)
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    @staticmethod
    def resize_image_for_display(image, max_width=800, max_height=800):
        """
        Resize gambar untuk ditampilkan di GUI
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        width, height = image.size
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    @staticmethod
    def convert_to_tkinter(image):
        """
        Konversi gambar untuk tkinter
        """
        if isinstance(image, np.ndarray):
            # Handle different numpy array formats
            if image.ndim == 2:  # Grayscale
                image = Image.fromarray(image, mode='L')
            elif image.ndim == 3:  # RGB or BGR
                if image.shape[2] == 3:
                    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    image = Image.fromarray(image)
            else:
                image = Image.fromarray(image)
        return ImageTk.PhotoImage(image)

    @staticmethod
    def preprocess_xray(image):
        """
        Preprocessing untuk gambar X-ray:
        1. Auto-orient sesuai EXIF (menggunakan PIL.ImageOps.exif_transpose)
        2. Konversi ke RGB hanya jika bukan mode RGB
        """
        # 1. Auto-orientasi berdasarkan data EXIF
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            # Lewati jika tidak ada EXIF atau terjadi error
            pass
        
        # 2. Konversi mode ke RGB jika perlu
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image

    @staticmethod
    def apply_view_mode(image, view_mode):
        """
        Aplikasi view mode ke gambar
        """
        # Konversi PIL Image ke numpy array jika diperlukan
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Pastikan dalam format RGB
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            # Jika sudah RGB, tetap gunakan
            pass
        elif img_array.ndim == 2:
            # Jika grayscale, konversi ke RGB untuk konsistensi
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        if view_mode == "original":
            return img_array
        elif view_mode == "grayscale":
            return ImageProcessor.to_grayscale(img_array)
        elif view_mode == "inverted_grayscale":
            return ImageProcessor.to_inverted_grayscale(img_array)
        elif view_mode == "clahe":
            return ImageProcessor.apply_clahe(img_array)
        else:
            return img_array

    @staticmethod
    def to_grayscale(image):
        """
        Konversi gambar ke grayscale
        """
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                # Konversi RGB ke grayscale menggunakan weighted average
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                return gray
            else:
                return image
        else:
            # PIL Image
            return np.array(image.convert('L'))

    @staticmethod
    def to_inverted_grayscale(image):
        """
        Konversi gambar ke inverted grayscale
        """
        gray = ImageProcessor.to_grayscale(image)
        # Invert menggunakan bitwise not atau subtraction
        inverted = 255 - gray
        return inverted

    @staticmethod
    def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Aplikasi CLAHE (Contrast Limited Adaptive Histogram Equalization)
        """
        # Konversi ke grayscale terlebih dahulu
        gray = ImageProcessor.to_grayscale(image)
        
        # Buat CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # Aplikasi CLAHE
        clahe_img = clahe.apply(gray)
        
        return clahe_img

    @staticmethod
    def apply_histogram_equalization(image):
        """
        Aplikasi histogram equalization
        """
        # Konversi ke grayscale terlebih dahulu
        gray = ImageProcessor.to_grayscale(image)
        
        # Aplikasi histogram equalization
        equalized = cv2.equalizeHist(gray)
        
        return equalized

    @staticmethod
    def flip(image, vertical=False, horizontal=False):
        """Flip vertikal dan/atau horizontal."""
        arr = np.array(image)
        if vertical:
            arr = cv2.flip(arr, 0)
        if horizontal:
            arr = cv2.flip(arr, 1)
        return arr

    @staticmethod
    def rotate(image, angle):
        """Rotate di sekitar center."""
        arr = np.array(image)
        (h, w) = arr.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(arr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def zoom(image, scale):
        """Zoom in (>1) atau zoom out (<1)."""
        arr = np.array(image)
        h, w = arr.shape[:2]
        new_w, new_h = int(w*scale), int(h*scale)
        resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized

    @staticmethod
    def pan(image, dx, dy):
        """Pan (geser) sebanyak dx, dy piksel."""
        arr = np.array(image)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        h, w = arr.shape[:2]
        return cv2.warpAffine(arr, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    @staticmethod
    def zoom_to_area(image, x1, y1, x2, y2, target_width=800, target_height=800):
        """
        Zoom ke area tertentu yang dipilih oleh rectangle selection
        """
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Pastikan koordinat valid
        h, w = img_array.shape[:2] if img_array.ndim == 3 else (img_array.shape[0], img_array.shape[1])
        x1, y1 = max(0, min(x1, x2)), max(0, min(y1, y2))
        x2, y2 = min(w, max(x1, x2)), min(h, max(y1, y2))
        
        # Crop area yang dipilih
        if img_array.ndim == 3:
            cropped = img_array[y1:y2, x1:x2]
        else:
            cropped = img_array[y1:y2, x1:x2]
        
        # Resize untuk fit ke target size sambil mempertahankan aspect ratio
        crop_h, crop_w = cropped.shape[:2] if cropped.ndim == 3 else (cropped.shape[0], cropped.shape[1])
        ratio = min(target_width / crop_w, target_height / crop_h)
        
        new_w = int(crop_w * ratio)
        new_h = int(crop_h * ratio)
        
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    @staticmethod
    def reset_transformations():
        """
        Reset semua transformasi ke default
        """
        return {
            'flip_vert': False,
            'flip_horiz': False,
            'rotate_angle': 0,
            'zoom_scale': 1.0,
            'pan_dx': 0,
            'pan_dy': 0,
            'zoom_area': None
        }