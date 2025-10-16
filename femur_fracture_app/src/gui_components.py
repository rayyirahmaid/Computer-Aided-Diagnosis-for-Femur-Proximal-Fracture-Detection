import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageEnhance, ImageDraw

from .model_handler import YOLOModelHandler
from .image_processor import ImageProcessor
from .evaluation import BoundingBoxEvaluator, BoxCountErrorEvaluator

class FractureDetectionApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Femur Fracture Detection App")
        self.root.geometry("1200x800")

        # Model dan processor
        self.model_handler = YOLOModelHandler(model_path)
        self.class_list = self.model_handler.model.names
        self.image_processor = ImageProcessor()
        self.evaluator = BoundingBoxEvaluator()

        # State gambar & anotasi
        self.original_image = None
        self.processed_image = None
        self.clean_image = None
        self.current_image = None
        self.model_results = None
        self.user_boxes = []
        self.user_classes = []

        # Mode aplikasi dan view - MODIFIED: Added guidance mode
        self.current_mode = "detection"
        self.current_view_mode = "original"
        self.view_modes = [
            ("Original","original"),
            ("Grayscale","grayscale"),
            ("Inverted Grayscale","inverted_grayscale"),
            ("CLAHE","clahe"),
        ]

        # Variabel orientasi
        self.flip_vert = False
        self.flip_horiz = False
        self.rotate_angle = 0

        # Mode kursor
        self.cursor_mode = "neutral"
        self.cursor_modes = {
            "neutral": "arrow",
            "pan": "fleur",
            "pointer": "crosshair",
            "annotate":"tcross" # cursor untuk anotasi
        }

        # Magnify state
        self.zoom_scale = 1.0 # inisialisasi zoom_scale

        # Track evaluasi dan zoom default
        self.magnify_var = tk.DoubleVar(value=1.0) # inisialisasi var slider
        self._pan_start = None
        self._pointer_start= None
        self.brightness_var = tk.DoubleVar(value=1.0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.reset_annotation_button = None
        self.save_guidance_button = None  # NEW: Save button for guidance mode
        self.evaluation_done = False # Track if evaluate has been called

        # Save functionality variables
        self.current_image_path = None
        self.class_map = {} # mapping nama kelas → class_id

        self.user_results_folder = "C:/RayFile/KuliahBro/Semester8/TugasAkhir/femur_fracture_app/Hasil Deteksi/Training Mode" #evaluate
        self.results_folder = "C:/RayFile/KuliahBro/Semester8/TugasAkhir/femur_fracture_app/Hasil Deteksi/Detection Mode" #detection
        self.guidance_results_folder = "C:/RayFile/KuliahBro/Semester8/TugasAkhir/femur_fracture_app/Hasil Deteksi/Guidance Mode" #guidance

        
        # Buat folder images/ dan labels/
        os.makedirs(os.path.join(self.results_folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.results_folder, 'labels'), exist_ok=True)

        
        os.makedirs(os.path.join(self.user_results_folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.user_results_folder, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.user_results_folder, 'metrics'), exist_ok=True)

        # NEW: Guidance mode folder structure

        os.makedirs(os.path.join(self.guidance_results_folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.guidance_results_folder, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.guidance_results_folder, 'combined'), exist_ok=True)

        self.setup_ui()

    def setup_ui(self):
        self.create_menu()
        self.create_toolbar()
        self.create_main_content()
        self.create_status_bar()
        self.create_orientation_toolbar()
        self.display_available_classes()
                
    def create_orientation_toolbar(self):
        frame = ttk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        #Flip & rotate controls
        ttk.Button(frame, text="Flip Vert", command=self.toggle_flip_vert).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame, text="Flip Horz", command=self.toggle_flip_horiz).pack(side=tk.LEFT, padx=2)
        
        angles = [str(a) for a in range(0,361,45)]
        self.rotate_var = tk.StringVar(value="0")
        cb = ttk.Combobox(frame, textvariable=self.rotate_var, values=angles,
                          width=5, state="readonly")
        cb.pack(side=tk.LEFT, padx=5)
        cb.bind("<<ComboboxSelected>>",
                lambda e: self.change_rotate(int(self.rotate_var.get())))
        
        # Slider Magnify (zoom via slider)
        ttk.Separator(frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Label(frame, text="Magnify:").pack(side=tk.LEFT)
        self.magnify_var = tk.DoubleVar(value=1.0)
        ttk.Scale(frame, from_=0.5, to=3.0, orient=tk.HORIZONTAL,
                  variable=self.magnify_var,
                  length=150,
                  command=self.on_magnify_change).pack(side=tk.LEFT, padx=(5,10))
        self.magnify_label = ttk.Label(frame, text="100%")
        self.magnify_label.pack(side=tk.LEFT)
        
        ttk.Separator(frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
    
        # Slider Brightness
        ttk.Label(frame, text="Brightness:").pack(side=tk.LEFT)
        ttk.Scale(frame,
                from_=0.0, to=2.0, orient=tk.HORIZONTAL,
                variable=self.brightness_var, length=150,
                command=self.update_brightness).pack(side=tk.LEFT, padx=5)
        self.brightness_label = ttk.Label(frame, text="100%")
        self.brightness_label.pack(side=tk.LEFT, padx=(0,10))
        
        # Slider Contrast
        ttk.Label(frame, text="Contrast:").pack(side=tk.LEFT)
        ttk.Scale(frame,
                from_=0.0, to=2.0, orient=tk.HORIZONTAL,
                variable=self.contrast_var, length=150,
                command=self.update_contrast).pack(side=tk.LEFT, padx=5)
        self.contrast_label = ttk.Label(frame, text="100%")
        self.contrast_label.pack(side=tk.LEFT)
        
        ttk.Button(frame, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=2)

    def apply_transformations(self):
        if self.original_image is None:
            return None
        
        # 1. Terapkan view mode, flip, rotate
        img = self.image_processor.apply_view_mode(self.original_image, self.current_view_mode)
        img = self.image_processor.flip(img, self.flip_vert, self.flip_horiz)
        img = self.image_processor.rotate(img, self.rotate_angle)
        
        # 2. Jika numpy array, konversi ke PIL Image sebelum enhancement
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = Image.fromarray(img, mode='L')
            else:
                img = Image.fromarray(img)
        
        # 3. Terapkan brightness & contrast enhancement
        img = ImageEnhance.Brightness(img).enhance(self.brightness_var.get())
        img = ImageEnhance.Contrast(img).enhance(self.contrast_var.get())
        
        return img

    def update_display(self):
        img = self.apply_transformations()
        if img is None:
            return

        self.processed_image = img
        self.current_image = self.image_processor.resize_image_for_display(img)
        self.display_image(self.current_image)

        if self.current_mode == "detection" and self.model_results:
            self.update_detection_display()
        elif self.current_mode == "training":
            # Always redraw user annotations
            self.redraw_user_annotations()
            
            # Only redraw model BB if evaluation has been done
            if self.evaluation_done and self.model_results:
                self.redraw_model_bounding_boxes()

    def _generate_save_filenames(self, input_path, folder):
        """
        Menghasilkan tuple (img_path, txt_path) dengan format:
        {basename}_{YYYYMMDD}.(jpg/txt)
        """
        basename = os.path.splitext(os.path.basename(input_path))[0]
        name_part = f"{basename}"
        img_path = os.path.join(folder, 'images', name_part + ".jpg")
        txt_path = os.path.join(folder, 'labels', name_part + ".txt")
        return img_path, txt_path

    
    def _save_yolo_label(self, label_path, boxes, names):
        """
        Simpan label YOLO untuk deteksi sistem dengan class_id sesuai class_list.
        """
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        img_w, img_h = self.original_image.size
        
        with open(label_path, 'w') as f:
            for box, name in zip(boxes, names):
                # Dapatkan class_id dari dict class_list
                class_id = next(cid for cid, cname in self.class_list.items() if cname == name)
                
                x1, y1, x2, y2 = box
                cx = ((x1 + x2) / 2.0) / img_w
                cy = ((y1 + y2) / 2.0) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    def _save_yolo_label_user(self, label_path):
        """
        Simpan label YOLO untuk anotasi user dengan class_id sesuai class_list.
        """
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        img_w, img_h = self.original_image.size
        
        with open(label_path, 'w') as f:
            for box, name in zip(self.user_boxes, self.user_classes):
                # Dapatkan class_id dari dict class_list
                class_id = next(cid for cid, cname in self.class_list.items() if cname == name)
                
                x1, y1, x2, y2 = box
                cx = ((x1 + x2) / 2.0) / img_w
                cy = ((y1 + y2) / 2.0) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


    # Tambahkan method handler untuk slider
    def update_brightness(self, value):
        pct = int(float(value) * 100)
        self.brightness_label.config(text=f"{pct}%")
        
        if self.current_mode == "training":
            # Apply transformations and display
            img = self.apply_transformations()
            if img is None:
                return
            
            self.processed_image = img
            self.current_image = self.image_processor.resize_image_for_display(img)
            self.display_image(self.current_image)
            
            # Always redraw user annotations
            self.redraw_user_annotations()
            
            # Only redraw model BB if evaluation has been done
            if self.evaluation_done and self.model_results:
                self.redraw_model_bounding_boxes()
        else:
            # For detection mode, use original behavior
            self.update_display()

    def update_contrast(self, value):
        pct = int(float(value) * 100)
        self.contrast_label.config(text=f"{pct}%")
        
        if self.current_mode == "training":
            # Apply transformations and display
            img = self.apply_transformations()
            if img is None:
                return
            
            self.processed_image = img
            self.current_image = self.image_processor.resize_image_for_display(img)
            self.display_image(self.current_image)
            
            # Always redraw user annotations
            self.redraw_user_annotations()
            
            # Only redraw model BB if evaluation has been done
            if self.evaluation_done and self.model_results:
                self.redraw_model_bounding_boxes()
        else:
            # For detection mode, use original behavior
            self.update_display()
    
    # Handlers orientation
    def toggle_flip_vert(self):
        self.flip_vert = not self.flip_vert
        self.update_display()

    def toggle_flip_horiz(self):
        self.flip_horiz = not self.flip_horiz
        self.update_display()

    def change_rotate(self, angle):
        self.rotate_angle = angle
        self.update_display()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.load_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Mode deteksi/training/guidance - MODIFIED: Added guidance mode
        mode_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Mode", menu=mode_menu)
        mode_menu.add_command(label="Detect Fracture", command=self.set_detection_mode)
        mode_menu.add_command(label="Training", command=self.set_training_mode)
        mode_menu.add_command(label="Guidance", command=self.set_guidance_mode)  # NEW

        # View mode
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        for name, mode in self.view_modes:
            view_menu.add_command(
                label=name,
                command=lambda vm=mode: self.change_view_mode(vm)
            )

        # Cursor mode
        cursor_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Cursor", menu=cursor_menu)
        cursor_menu.add_command(label="Pointer", command=lambda: self.set_cursor_mode("neutral"))
        cursor_menu.add_command(label="Pan", command=lambda: self.set_cursor_mode("pan"))
        cursor_menu.add_command(label="Annotate", command=lambda: self.set_cursor_mode("annotate"))

    def create_toolbar(self):
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Tombol utama
        ttk.Button(toolbar, text="Open Image", command=self.load_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Detect Fracture", command=self.detect_fractures).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Evaluate", command=self.evaluate_annotations).pack(side=tk.LEFT, padx=2)
        
        self.reset_annotation_button = ttk.Button(toolbar, text="Reset Annotation", command=self.clear_annotations)
        # NEW: Save guidance button
        self.save_guidance_button = ttk.Button(toolbar, text="Save Result", command=self.save_guidance_results)

        # Separator & View Mode
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Label(toolbar, text="View Mode:").pack(side=tk.LEFT)
        self.view_mode_var = tk.StringVar(value="Original")
        cb = ttk.Combobox(toolbar, textvariable=self.view_mode_var,
                        values=[n for n,_ in self.view_modes],
                        state="readonly", width=15)
        cb.pack(side=tk.LEFT, padx=5)
        cb.bind("<<ComboboxSelected>>", self.on_view_mode_change)

        # Confidence slider
        cf_frame = ttk.Frame(toolbar)
        cf_frame.pack(side=tk.LEFT, padx=20)
        ttk.Label(cf_frame, text="Confidence:").pack(side=tk.LEFT)
        self.confidence_var = tk.DoubleVar(value=0.5)
        ttk.Scale(cf_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL,
                variable=self.confidence_var,
                command=self.update_confidence_label,
                length=100).pack(side=tk.LEFT)
        self.confidence_label = ttk.Label(cf_frame, text="50%")
        self.confidence_label.pack(side=tk.LEFT, padx=5)

        # Status mode
        self.mode_label = ttk.Label(toolbar, text="Mode: Fracture Detection", foreground="blue")
        self.mode_label.pack(side=tk.RIGHT)

        # Update visibilitas awal
        self.update_button_visibility()

    def update_button_visibility(self):
        """MODIFIED: Added guidance mode button visibility"""
        if self.current_mode == "training":
            self.reset_annotation_button.pack(side=tk.LEFT, padx=2)
            self.save_guidance_button.pack_forget()
        elif self.current_mode == "guidance":
            self.reset_annotation_button.pack(side=tk.LEFT, padx=2)
            self.save_guidance_button.pack(side=tk.LEFT, padx=2)
        else:
            self.reset_annotation_button.pack_forget()
            self.save_guidance_button.pack_forget()

    def reset_view(self):
        """
        Reset semua transformasi tampilan:
        - magnify / zoom
        - flip vertical & horizontal
        - rotate
        - brightness & contrast
        - view mode di-reset ke 'original'
        Kemudian tampilkan ulang gambar dalam kondisi awal (sebagaimana di-load).
        """
        # 1. Reset state transformasi
        if self.clean_image is not None:
            display_img = self.image_processor.resize_image_for_display(self.clean_image)
            self.current_image = display_img
            self.display_image(display_img)
        elif self.original_image is not None:
            self.apply_current_view_mode()
        
        self.magnify_label.config(text="100%")
        self.flip_vert = False
        self.flip_horiz = False
        self.rotate_angle = 0
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.brightness_label.config(text="100%")
        self.contrast_label.config(text="100%")

        # 5. Update tampilan status
        self.status_var.set("View telah di-reset ke kondisi awal.")
    
    def update_confidence_label(self, value):
        pct = int(float(value) * 100)
        self.confidence_label.config(text=f"{pct}%")
    
    def on_magnify_change(self, value):
        pct = int(float(value) * 100)
        self.magnify_label.config(text=f"{pct}%")
        self.zoom_scale = float(value)
        self.apply_magnify()

    def apply_magnify(self):
        if not self.clean_image:
            return
        
        w, h = self.clean_image.size
        new_w = int(w * self.zoom_scale)
        new_h = int(h * self.zoom_scale)
        
        resized = self.clean_image.resize((new_w, new_h), Image.LANCZOS)
        self.current_image = resized
        
        self.display_image(self.current_image, fit=False)
        
        # In training mode, preserve user annotations and model BB based on evaluation state
        if self.current_mode == "training":
            # Always redraw user annotations
            self.redraw_user_annotations()
            
            # Only redraw model BB if evaluation has been done
            if self.evaluation_done and self.model_results:
                self.redraw_model_bounding_boxes()


    def create_main_content(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        left = ttk.LabelFrame(main, text="Gambar X-Ray")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5))
        
        self.canvas = tk.Canvas(left, bg="white", cursor="arrow")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        right = ttk.LabelFrame(main, text="Informasi")
        right.config(width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(5,0))
        
        self.results_text = tk.Text(right, wrap=tk.WORD, width=35, height=20)
        
        sb = ttk.Scrollbar(right, command=self.results_text.yview)
        
        self.results_text.configure(yscrollcommand=sb.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas_events()

    def create_status_bar(self):
        self.status_var = tk.StringVar(value="Siap")
        status = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status.pack(side=tk.BOTTOM, fill=tk.X)
    
    # 3. Ubah set_cursor_mode agar menggunakan self.cursor_modes:
    def set_cursor_mode(self, mode):
        if mode not in self.cursor_modes:
            return
        self.cursor_mode = mode
        self.canvas.config(cursor=self.cursor_modes[mode])

    # 4. Di canvas_events(), binding event click/tarik lepaskan:
    def canvas_events(self):
        self.canvas.bind("<ButtonPress-1>",    self._on_left_press)
        self.canvas.bind("<B1-Motion>",        self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>",  self._on_left_release)
        
    def _on_left_press(self, event):
        if self.cursor_mode == "pan":
            return self.pan_start(event)
        if self.cursor_mode == "pointer":
            return self.pointer_start(event)
        if self.cursor_mode == "annotate" and self.current_mode in ("training","guidance"):
            return self.start_draw(event)

    def _on_left_drag(self, event):
        if self.cursor_mode == "pan":
            return self.pan_move(event)
        if self.cursor_mode == "pointer":
            return self.pointer_drag(event)
        if self.cursor_mode == "annotate" and self.current_mode in ("training","guidance"):
            return self.draw_rectangle(event)

    def _on_left_release(self, event):
        if self.cursor_mode == "pointer":
            return self.pointer_end(event)
        if self.cursor_mode == "annotate" and self.current_mode in ("training","guidance"):
            return self.end_draw(event)

    # Pan handlers
    def pan_start(self, event):
        if self.cursor_mode != "pan": return
        self.canvas.scan_mark(event.x, event.y)

    def pan_move(self, event):
        if self.cursor_mode != "pan": return
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    # Pointer handlers
    def pointer_start(self, event):
        if self.cursor_mode != "pointer":
            return
        
        x0, y0 = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self._pointer_start = (x0, y0)
        # Buat mask hitam penuh
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.canvas.delete("mask")
        self.canvas.create_rectangle(0, 0, w, h,
                                    fill="black", stipple="gray25",
                                    tags="mask")

    def pointer_drag(self, event):
        if self.cursor_mode != "pointer" or not self._pointer_start:
            return

        x0, y0 = self._pointer_start
        x1, y1 = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        # Hapus overlay & highlight lama
        self.canvas.delete("mask")
        self.canvas.delete("highlight_rect")

        # Ukuran kanvas
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        # Overlay hitam semi-transparan di seluruh kanvas
        # gunakan stipple ringan (misal 'gray75') agar lebih gelap tapi area kotak masih terlihat
        self.canvas.create_rectangle(
            0, 0, w, h,
            fill="black", stipple="gray75",
            outline="", tags="mask"
        )

        # Hitung batas kotak highlight
        x_min, x_max = sorted((x0, x1))
        y_min, y_max = sorted((y0, y1))

        # Ambil sub-image dari current_image dan tampilkan di atas mask
        img = self.current_image.crop((x_min - self.image_offset_x,
                                    y_min - self.image_offset_y,
                                    x_max - self.image_offset_x,
                                    y_max - self.image_offset_y))
        tk_img = ImageTk.PhotoImage(img)
        # Simpan reference untuk mencegah garbage collection
        self._highlight_photo = tk_img
        self.canvas.create_image(
            x_min, y_min,
            anchor=tk.NW,
            image=tk_img,
            tags="mask"
        )

        # Gambar kotak merah
        self.canvas.create_rectangle(
            x_min, y_min, x_max, y_max,
            outline="red", width=2,
            tags="highlight_rect"
        )

    def pointer_end(self, event):
        if self.cursor_mode != "pointer":
            return
        # Highlight tetap dipertahankan hingga reset
        self._pointer_start = None

    def load_image(self):
        """Load a new X-Ray image and reset view in all modes, including Guidance."""
        path = filedialog.askopenfilename(
            title="Pilih Gambar X-Ray",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if not path:
            return

        # Simpan path gambar saat ini
        self.current_image_path = path

        # Muat gambar asli
        img = self.image_processor.load_image(path)
        if not img:
            messagebox.showerror("Error", f"Gagal memuat gambar: {path}")
            return

        self.original_image = img

        # Reset semua transformasi dan anotasi
        # Reset view mode, flip, rotate, brightness, contrast, zoom
        self.current_view_mode = "original"
        self.view_mode_var.set("Original")
        self.flip_vert = False
        self.flip_horiz = False
        self.rotate_angle = 0
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.magnify_var.set(1.0)
        self.zoom_scale = 1.0
        self.brightness_label.config(text="100%")
        self.contrast_label.config(text="100%")
        self.magnify_label.config(text="100%")

        # Clear any existing annotations and model results
        self.user_boxes = []
        self.user_classes = []
        self.evaluation_done = False
        self.model_results = None

        # Apply view and display
        self.apply_current_view_mode()  # this sets processed_image & clean_image
        self.current_image = self.image_processor.resize_image_for_display(self.clean_image)
        self.display_image(self.current_image)

        # Update status and instructions panel
        self.status_var.set(f"Gambar dimuat: {os.path.basename(path)}")
        if self.current_mode == "guidance":
            # Tampilkan instruksi ulang untuk guidance mode
            instr = (
                "=== MODE GUIDANCE ===\n\n"
                "Mode guidance aktif untuk memandu deteksi fraktur.\n\n"
                "INSTRUKSI:\n"
                "1. Klik 'Detect Fracture' untuk menjalankan deteksi\n"
                "2. Sistem akan menampilkan bounding boxes hasil deteksi\n"
                "3. Buat anotasi Anda dengan menggambar kotak\n"
                "4. Klik 'Save Result' untuk menyimpan hasil\n\n"
                "Model akan menampilkan prediksi untuk memandu Anda."
            )
            self.update_results_text(instr)
        
        # NEW: Auto-detect in training mode when image is loaded
        if self.current_mode == "training":
            # Automatically run detection
            self.detect_fractures()
        
        else:
            # Default instruction after loading image
            self.update_results_text("Gambar berhasil dimuat. Pilih mode dan mulai analisis.")

    def apply_current_view_mode(self):
        if self.original_image is None:
            return
        processed_img = self.image_processor.apply_view_mode(self.original_image, self.current_view_mode)
        if isinstance(processed_img, np.ndarray):
            if processed_img.ndim == 2:
                processed_img = Image.fromarray(processed_img, mode='L')
            else:
                processed_img = Image.fromarray(processed_img)
        self.processed_image = processed_img
        self.clean_image = processed_img
        self.current_image = self.image_processor.resize_image_for_display(processed_img)
        self.display_image(self.current_image)
        self.apply_magnify()

    def change_view_mode(self, view_mode):
        self.current_view_mode = view_mode
        for view_name, vm in self.view_modes:
            if vm == view_mode:
                self.view_mode_var.set(view_name)
                break
        if self.original_image:
            self.apply_current_view_mode()
            if self.current_mode == "detection" and self.model_results:
                self.update_detection_display()
            elif self.model_results:
                self.redraw_model_bounding_boxes()
                self.redraw_user_annotations()

    def on_view_mode_change(self, event):
        selected_name = self.view_mode_var.get()
        for view_name, view_mode in self.view_modes:
            if view_name == selected_name:
                self.change_view_mode(view_mode)
                break

    def display_image(self, img, fit=True):
        if fit:
            img = self.image_processor.resize_image_for_display(img)
        self.photo = self.image_processor.convert_to_tkinter(img)
        self.canvas.delete("all")
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        x = max((cw - self.photo.width()) // 2, 0)
        y = max((ch - self.photo.height()) // 2, 0)
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
        self.image_offset_x, self.image_offset_y = x, y

    def detect_fractures(self):
        """
        MODIFIED: Added guidance mode handling
        Jalankan deteksi model:
        - Mode detection: tampilkan bounding boxes dan simpan seperti biasa.
        - Mode training: simpan hasil deteksi ke direktori yang sama seperti mode detection, 
        namun tidak menampilkan bounding boxes hingga user menekan 'Evaluate'. 
        - Mode guidance: tampilkan bounding boxes dan izinkan user annotation.
        """
        if self.original_image is None:
            messagebox.showwarning("Warning", "Silakan muat gambar terlebih dahulu.")
            return

        try:
            # Mulai deteksi model
            self.status_var.set("Mendeteksi fraktur...")
            self.root.update()

            conf = self.confidence_var.get()
            self.model_results = self.model_handler.predict(self.original_image, conf)

            # Simpan hasil deteksi (gambar ter-annotasi & YOLO labels)
            try:
                img_path, txt_path = self._generate_save_filenames(
                    self.current_image_path,
                    folder=self.results_folder
                )

                # Simpan annotated_image
                img_rgb = cv2.cvtColor(self.model_results['annotated_image'], cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                pil_img.save(img_path, format='JPEG', quality=95)

                # Simpan YOLO label
                self._save_yolo_label(
                    txt_path,
                    self.model_results['boxes'],
                    self.model_results['class_names']
                )

            except Exception as e:
                messagebox.showerror("Error Save", f"Gagal menyimpan hasil: {e}")

            # Atur tampilan menurut mode
            if self.current_mode == "detection":
                # Mode detection: tampilkan BB segera
                self.update_detection_display()
                self.update_detection_results()
                self.status_var.set(
                    f"Deteksi selesai — gambar: {os.path.basename(img_path)}, "
                    f"label: {os.path.basename(txt_path)}"
                )

            elif self.current_mode == "guidance":
                # NEW: Mode guidance: tampilkan BB segera dan izinkan anotasi
                self.update_guidance_display()
                self.update_guidance_results()
                self.status_var.set(
                    f"Deteksi selesai (guidance) — hasil disimpan: {os.path.basename(img_path)}"
                )

            elif self.current_mode == "training":
                # Mode training: jangan tampilkan BB dulu, jaga ukuran gambar
                instruction = (
                    "=== MODE TRAINING ===\n\n"
                    "Deteksi model telah selesai dan hasil disimpan.\n"
                    "Bounding boxes akan muncul setelah Anda menekan 'Evaluate'.\n\n"
                    "Silakan buat anotasi Anda atau langsung klik 'Evaluate'."
                )

                self.update_results_text(instruction)
                self.status_var.set(
                    f"Deteksi selesai (training) — hasil disimpan: {os.path.basename(img_path)}"
                )
                # Tampilkan gambar bersih pada canvas tanpa resize ulang
                self.display_image(self.current_image, fit=False)

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error")

    def update_detection_display(self):
        if not self.model_results:
            return
        annotated_img = self.model_results['annotated_image']
        if self.current_view_mode != "original":
            processed = self.image_processor.apply_view_mode(self.original_image, self.current_view_mode)
            display_img = self.draw_bounding_boxes_on_image(processed)
        else:
            display_img = annotated_img
        if isinstance(display_img, np.ndarray):
            if display_img.ndim == 2:
                display_img = Image.fromarray(display_img, mode='L')
            else:
                display_img = Image.fromarray(display_img)
        self.current_image = self.image_processor.resize_image_for_display(display_img)
        self.display_image(self.current_image)

    def draw_bounding_boxes_on_image(self, image):
        img_copy = np.array(image) if not isinstance(image, np.ndarray) else image.copy()
        if img_copy.ndim == 2:
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
        for box, conf, name in zip(self.model_results['boxes'], self.model_results['confidences'], self.model_results['class_names']):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name}: {conf:.2f}"
            cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img_copy

    def update_detection_results(self):
        if not self.model_results:
            return
        txt = "=== HASIL DETEKSI ===\n\n"
        for i, (box, conf, name) in enumerate(zip(self.model_results['boxes'], self.model_results['confidences'], self.model_results['class_names'])):
            txt += (
                f"{i+1}. {name}\n"
                f"   Confidence: {conf:.3f} ({int(conf*100)}%)\n"
                f"   Lokasi: ({box[0]}, {box[1]}) - ({box[2]}, {box[3]})\n\n"
            )
        self.update_results_text(txt)

    def start_draw(self, event):
        if self.current_mode not in ("training", "guidance") or self.current_image is None:
            return
        self.drawing = True
        self.start_x = event.x - self.image_offset_x
        self.start_y = event.y - self.image_offset_y

    def draw_rectangle(self, event):
        if not self.drawing:
            return
        self.canvas.delete("temp_rect")
        cx, cy = event.x - self.image_offset_x, event.y - self.image_offset_y
        self.canvas.create_rectangle(
            self.start_x + self.image_offset_x,
            self.start_y + self.image_offset_y,
            cx + self.image_offset_x,
            cy + self.image_offset_y,
            outline="red", width=2, tags="temp_rect"
        )

    def end_draw(self, event):
        """MODIFIED: Handle annotation for both training and guidance modes"""
        if not self.drawing:
            return

        self.drawing = False
        ex, ey = event.x - self.image_offset_x, event.y - self.image_offset_y

        if abs(ex - self.start_x) < 10 or abs(ey - self.start_y) < 10:
            self.canvas.delete("temp_rect")
            return

        mw, mh = self.current_image.width, self.current_image.height
        sx, sy = self.original_image.width / mw, self.original_image.height / mh

        x1 = int(min(self.start_x, ex) * sx)
        y1 = int(min(self.start_y, ey) * sy)
        x2 = int(max(self.start_x, ex) * sx)
        y2 = int(max(self.start_y, ey) * sy)

        self.user_boxes.append([x1, y1, x2, y2])
        self.canvas.delete("temp_rect")
        
        # Draw user rectangle
        self.canvas.create_rectangle(
            min(self.start_x, ex) + self.image_offset_x,
            min(self.start_y, ey) + self.image_offset_y,
            max(self.start_x, ex) + self.image_offset_x,
            max(self.start_y, ey) + self.image_offset_y,
            outline="red", width=2, tags="user_rect"
        )

        # Validasi input kelas user - harus dari kelas tersedia
        available_classes = list(self.class_list.values())
        kelas = None
        while kelas not in available_classes:
            if kelas is not None: # Jika sudah pernah input salah
                messagebox.showwarning("Kelas Tidak Valid",
                                    f"Kelas '{kelas}' tidak tersedia. Pilih dari: {available_classes}")
            
            kelas = simpledialog.askstring(
                "Input Kelas",
                f"Masukkan nama kelas (pilih dari {available_classes}):",
                parent=self.root
            )

            if kelas is None: # User cancel
                # Hapus box yang baru dibuat
                self.user_boxes.pop()
                self.canvas.delete("user_rect")
                return

        self.user_classes.append(kelas)

        self.canvas.create_text(
            min(self.start_x, ex) + self.image_offset_x + 4,
            min(self.start_y, ey) + self.image_offset_y + 4,
            text=kelas,
            anchor="nw", fill="red", font=("TkDefaultFont", 12), tags="user_label"
        )

        # Update results based on mode
        if self.current_mode == "guidance":
            self.update_results_text(f"Anotasi guidance ditambahkan: ({x1},{y1})-({x2},{y2}), Kelas: {kelas}")
        else:
            self.update_results_text(f"Anotasi ditambahkan: ({x1},{y1})-({x2},{y2}), Kelas: {kelas}")

    def clear_annotations(self):
        """MODIFIED: Reset anotasi dan kembali ke gambar bersih, with guidance mode support"""
        self.user_boxes = []
        self.user_classes = []
        self.evaluation_done = False # Reset evaluation flag
        self.canvas.delete("all")

        # Reset model results jika dalam mode detection
        if self.current_mode == "detection":
            self.model_results = None

        # Tampilkan gambar bersih
        if self.clean_image is not None:
            display_img = self.image_processor.resize_image_for_display(self.clean_image)
            self.current_image = display_img
            self.display_image(display_img)
        elif self.original_image is not None:
            self.apply_current_view_mode()

        # Update status dan panel hasil berdasarkan mode
        if self.current_mode == "training":
            if self.model_results is not None:
                # Jika ada hasil deteksi tersimpan, tampilkan instruksi
                instruction_text = (
                    "=== MODE TRAINING ===\n\n"
                    "Anotasi telah direset. Hasil deteksi model masih tersimpan.\n\n"
                    "INSTRUKSI:\n"
                    "1. Buat anotasi baru dengan menggambar kotak pada gambar\n"
                    "2. Klik tombol 'Evaluate' untuk melihat perbandingan\n"
                    "3. Hasil deteksi model akan muncul bersamaan dengan evaluasi\n\n"
                    f"Model mendeteksi {len(self.model_results['boxes'])} objek.\n"
                    "Silakan buat anotasi Anda!"
                )
                self.update_results_text(instruction_text)
                self.status_var.set("Anotasi direset. Hasil deteksi model masih tersimpan.")
            else:
                # Jika belum ada hasil deteksi
                self.update_results_text("Anotasi telah direset. Silakan jalankan deteksi terlebih dahulu.")
                self.status_var.set("Anotasi direset. Siap untuk deteksi.")
        
        elif self.current_mode == "guidance":
            # NEW: Guidance mode handling
            if self.model_results is not None:
                # Redraw model bounding boxes
                self.update_guidance_display()
                instruction_text = (
                    "=== MODE GUIDANCE ===\n\n"
                    "Anotasi user telah direset. Model bounding boxes ditampilkan kembali.\n\n"
                    f"Model mendeteksi {len(self.model_results['boxes'])} objek.\n"
                    "Silakan buat anotasi baru atau klik 'Save Result'."
                )
                self.update_results_text(instruction_text)
                self.status_var.set("Anotasi user direset. Model BB ditampilkan.")
            else:
                self.update_results_text("Anotasi telah direset. Silakan jalankan deteksi terlebih dahulu.")
                self.status_var.set("Anotasi direset. Siap untuk deteksi.")
        
        else:
            # Mode detection
            self.update_results_text("Anotasi telah direset. Siap untuk deteksi.")
            self.status_var.set("Anotasi direset. Siap untuk deteksi.")

    def evaluate_annotations(self):
        """
        Evaluasi anotasi user vs hasil deteksi model.
        Pada mode training, jika user belum membuat anotasi,
        tetap tampilkan hasil model dan metrik hitung kotak.
        """
        if self.model_results is None:
            messagebox.showwarning("Warning", "Jalankan deteksi model terlebih dahulu.")
            return

        # Set evaluation done flag
        self.evaluation_done = True
        
        # Pada mode training, user boleh skip anotasi:
        if self.current_mode == "training" and not self.user_boxes:
            # Hitung saja metrik count error (seluruh model menjadi FN=0, TP=0 jika user kosong)
            count_err = BoxCountErrorEvaluator.evaluate([], self.model_results['boxes'])
            det_metrics = BoundingBoxEvaluator.compute_detection_metrics([], self.model_results['boxes'], iou_threshold=0.5)
            overlap_eval = self.evaluator.evaluate_user_annotation([], self.model_results['boxes'], iou_threshold=0.5)
        else:
            # Mode detection atau training dengan anotasi
            if not self.user_boxes:
                messagebox.showwarning("Warning", "Tidak ada anotasi user untuk dievaluasi.")
                return
            overlap_eval = self.evaluator.evaluate_user_annotation(
                self.user_boxes, self.model_results['boxes'], iou_threshold=0.5)
            count_err = BoxCountErrorEvaluator.evaluate(
                self.user_boxes, self.model_results['boxes'])
            det_metrics = BoundingBoxEvaluator.compute_detection_metrics(
                self.user_boxes, self.model_results['boxes'], iou_threshold=0.5)

        # Render gambar evaluasi:
        img_eval = self._render_evaluation_image()
        basename = os.path.splitext(os.path.basename(self.current_image_path))[0]
        name_part = f"{basename}"

        # Simpan gambar evaluasi
        img_path = os.path.join(self.user_results_folder, 'images', name_part + ".jpg")
        img_eval.save(img_path, format='JPEG', quality=95)

        # Simpan label YOLO user (bisa kosong)
        txt_path = os.path.join(self.user_results_folder, 'labels', name_part + ".txt")
        self._save_yolo_label_user(txt_path)

        # Simpan metrik evaluasi
        metrics_path = os.path.join(self.user_results_folder, 'metrics', name_part + ".txt")
        with open(metrics_path, 'w') as f:
            f.write(f"TP: {det_metrics['TP']}\n")
            f.write(f"FP: {det_metrics['FP']}\n")
            f.write(f"FN: {det_metrics['FN']}\n")
            f.write(f"Precision: {det_metrics['Precision']:.4f}\n")
            f.write(f"Recall: {det_metrics['Recall']:.4f}\n")

        # Tampilkan hasil evaluasi di GUI
        self.display_evaluation_results(overlap_eval, count_err)

        if self.current_mode == "training":
            self.status_var.set("Evaluasi selesai. Model results dan anotasi user (jika ada) ditampilkan.")
        messagebox.showinfo(
            "Simpan Hasil",
            f"Files saved:\nImage: {os.path.basename(img_path)}\n"
            f"Label: {os.path.basename(txt_path)}\n"
            f"Metrics: {os.path.basename(metrics_path)}"
        )

    def display_evaluation_results(self, overlap_eval, count_err):
        """
        Tampilkan:
        - BB sistem (hijau) + nama objek model
        - BB user (merah) + nama objek user
        - Panel teks hasil evaluasi overlap dan box count error
        """
        # Render ulang gambar
        self.display_image(self.current_image)
        
        # Skala gambar
        iw, ih = self.current_image.size
        sx, sy = iw / self.original_image.width, ih / self.original_image.height
        
        # Gambar BB model (hijau)
        for box, name in zip(self.model_results['boxes'], self.model_results['class_names']):
            x1, y1, x2, y2 = box
            sx1, sy1 = x1 * sx + self.image_offset_x, y1 * sy + self.image_offset_y
            sx2, sy2 = x2 * sx + self.image_offset_x, y2 * sy + self.image_offset_y
            
            self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline="green", width=2, tags="model_eval_rect")
            self.canvas.create_text(sx1 + 4, sy1 + 4, text=name, anchor="nw", fill="green",
                                    font=("TkDefaultFont", 12, "bold"), tags="model_eval_label")
        
        # Gambar BB user (merah)
        for box, obj_name in zip(self.user_boxes, self.user_classes):
            x1, y1, x2, y2 = box
            ux1, uy1 = x1 * sx + self.image_offset_x, y1 * sy + self.image_offset_y
            ux2, uy2 = x2 * sx + self.image_offset_x, y2 * sy + self.image_offset_y
            
            self.canvas.create_rectangle(ux1, uy1, ux2, uy2, outline="red", width=2, tags="user_eval_rect")
            self.canvas.create_text(ux1 + 4, uy1 + 4, text=obj_name or "—", anchor="nw", fill="red",
                                    font=("TkDefaultFont", 12), tags="user_eval_label")
        
        # Panel teks hasil evaluasi
        txt = "=== HASIL DETEKSI ===\n\n"
        txt += f"Total Model Boxes : {overlap_eval['total_model']}\n\n"
        
        for i, (box, conf, name) in enumerate(zip(
            self.model_results['boxes'],
            self.model_results['confidences'],
            self.model_results['class_names']
        )):
            txt += (
                f"{i+1}. {name}\n"
                f"   Confidence: {conf:.3f} ({int(conf*100)}%)\n"
                f"   Lokasi: ({box[0]}, {box[1]}) - ({box[2]}, {box[3]})\n\n"
            )
        
        txt += "=== EVALUATION ===\n"
        txt += f"Total User Boxes : {count_err['total_user']}\n"
        txt += f"Boxes Count Error : {count_err['total_user'] - overlap_eval['total_model']}\n\n"
        
        txt += "=== DETAIL OVERLAP ===\n"
        for cmp in overlap_eval['comparisons']:
            mi, ov, status = cmp['model_idx'], cmp['best_iou'], ("✔" if cmp['match'] else "✘")
            txt += f"  IoU {mi+1}: {ov*100:.1f}% {status}\n"
        
        txt += f"Correct Overlaps : {overlap_eval['correct_count']}\n"
        txt += f"Score : {overlap_eval['percentage_correct']*100:.1f}%\n"
        
        self.update_results_text(txt)

    def set_detection_mode(self):
        self.current_mode = "detection"
        self.mode_label.config(text="Mode: Fracture Detection", foreground="blue")
        self.canvas.config(cursor="arrow")
        self.clear_annotations()
        self.update_button_visibility()

    def set_training_mode(self):
        self.current_mode = "training"
        self.evaluation_done = False  # Reset evaluation flag
        self.mode_label.config(text="Mode: Training", foreground="green")
        
        # Set cursor to annotate mode directly
        self.set_cursor_mode("annotate")
        self.canvas.config(cursor="crosshair")
        
        # Reset annotations, tapi biarkan model_results jika ada
        self.user_boxes = []
        self.user_classes = []
        
        # Clear canvas dan redraw gambar bersih
        self.canvas.delete("all")
        if self.clean_image:
            self.current_image = self.image_processor.resize_image_for_display(self.clean_image)
            self.display_image(self.current_image)
        elif self.original_image:
            self.apply_current_view_mode()
        self.update_button_visibility()
        
        # Instruksi training
        if self.model_results:
            instr = (
                "=== MODE TRAINING ===\n\n"
                "Mode training aktif. Hasil deteksi model tersimpan.\n\n"
                "INSTRUKSI:\n"
                "1. (Opsional) Buat anotasi dengan menggambar kotak\n"
                "2. Klik tombol 'Evaluate' untuk melihat hasil model dan evaluasi\n"
            )
            self.status_var.set("Mode training aktif. Model results ready.")
        else:
            instr = (
                "=== MODE TRAINING ===\n\n"
                "Mode training aktif.\n\n"
                "INSTRUKSI:\n"
                "1. Muat gambar X-Ray\n"
                "2. Klik 'Detect Fracture' untuk menjalankan deteksi\n"
                "3. Buat anotasi dengan menggambar kotak\n"
                "4. Klik 'Evaluate' untuk melihat hasil model dan evaluasi\n\n"
            )
            self.status_var.set("Mode training aktif. Siap untuk deteksi.")
        self.update_results_text(instr)
    
        # 7. If an image is already loaded, run detection automatically
        if self.original_image is not None and not self.model_results:
            # Automatically detect fractures without user clicking the button
            self.detect_fractures()

    def update_results_text(self, text):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        
    def display_available_classes(self):
        """
        Menampilkan semua kelas model beserta ID-nya
        di jendela informasi setiap aplikasi dibuka.
        """
        lines = ["=== KELAS TERSEDIA DI MODEL ===\n"]
        for cid, cname in self.class_list.items():
            lines.append(f"{cid}: {cname}\n")
        # Tampilkan di panel informasi
        self.update_results_text("".join(lines))

        
    def redraw_user_annotations(self):
        """
        Gambar ulang semua anotasi user di canvas berdasarkan self.user_boxes dan self.user_classes.
        """
        if not self.user_boxes:
            return
        # Hitung skala
        iw, ih = self.current_image.width, self.current_image.height
        sx = iw / self.original_image.width
        sy = ih / self.original_image.height

        # Gambar tiap kotak
        for box, obj_name in zip(self.user_boxes, self.user_classes):
            x1, y1, x2, y2 = box
            ux1 = x1 * sx + self.image_offset_x
            uy1 = y1 * sy + self.image_offset_y
            ux2 = x2 * sx + self.image_offset_x
            uy2 = y2 * sy + self.image_offset_y

            self.canvas.create_rectangle(
                ux1, uy1, ux2, uy2,
                outline="red", width=2, tags="user_rect"
            )
            self.canvas.create_text(
                ux1 + 4, uy1 + 4,
                text=obj_name or "—",
                anchor="nw", fill="red",
                font=("TkDefaultFont", 12), tags="user_label"
            )

    def redraw_model_bounding_boxes(self):
        """
        Gambar ulang semua bounding box model di canvas berdasarkan self.model_results.
        """
        if not self.model_results:
            return
        # Hitung skala
        iw, ih = self.current_image.width, self.current_image.height
        sx = iw / self.original_image.width
        sy = ih / self.original_image.height

        for box, name, conf in zip(
            self.model_results['boxes'],
            self.model_results['class_names'],
            self.model_results['confidences']
        ):
            x1, y1, x2, y2 = box
            mx1 = x1 * sx + self.image_offset_x
            my1 = y1 * sy + self.image_offset_y
            mx2 = x2 * sx + self.image_offset_x
            my2 = y2 * sy + self.image_offset_y

            # Rectangle hijau
            self.canvas.create_rectangle(
                mx1, my1, mx2, my2,
                outline="green", width=2, tags="model_rect"
            )
            # Label nama+confidence
            label = f"{name}: {int(conf*100)}%"
            self.canvas.create_text(
                mx1 + 4, my1 + 4,
                text=label,
                anchor="nw", fill="green",
                font=("TkDefaultFont", 12, "bold"), tags="model_label"
            )
    
    def _render_evaluation_image(self):
        """
        Menggambar ulang bounding box model (hijau) dan user (merah)
        lalu mengembalikan PIL Image.
        """
        # Gunakan annotated_image asli dari model_results
        base = Image.fromarray(self.model_results['annotated_image'])
        draw = ImageDraw.Draw(base)
        # Gambar bounding box user dengan warna merah
        for box in self.user_boxes:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        return base
    
    def set_guidance_mode(self):
        """NEW: Set application to guidance mode"""
        self.current_mode = "guidance"
        self.mode_label.config(text="Mode: Guidance", foreground="purple")
        self.canvas.config(cursor="crosshair")
        self.clear_annotations()
        self.update_button_visibility()
        
        # Instruksi guidance
        instr = (
            "=== MODE GUIDANCE ===\n\n"
            "Mode guidance aktif untuk memandu deteksi fraktur.\n\n"
            "INSTRUKSI:\n"
            "1. Muat gambar X-Ray\n"
            "2. Klik 'Detect Fracture' untuk menjalankan deteksi\n"
            "3. Sistem akan menampilkan bounding boxes hasil deteksi\n"
            "4. Buat anotasi Anda dengan menggambar kotak\n"
            "5. Klik 'Save Result' untuk menyimpan hasil\n\n"
            "Model akan menampilkan prediksi untuk memandu Anda.\n"
        )
        
        self.update_results_text(instr)
        self.status_var.set("Mode guidance aktif. Siap untuk deteksi.")

    def update_guidance_display(self):
        """NEW: Update display for guidance mode - shows model BB immediately"""
        if not self.model_results:
            return

        # Tampilkan gambar dengan model bounding boxes
        self.update_detection_display()
        
        # Set cursor untuk anotasi
        self.set_cursor_mode("annotate")

    def update_guidance_results(self):
        """NEW: Update results panel for guidance mode"""
        if not self.model_results:
            return

        txt = "=== GUIDANCE MODE - HASIL DETEKSI ===\n\n"
        txt += f"Total deteksi: {len(self.model_results['boxes'])}\n\n"
        
        for i, (box, conf, name) in enumerate(zip(
            self.model_results['boxes'], 
            self.model_results['confidences'], 
            self.model_results['class_names']
        )):
            txt += (
                f"{i+1}. {name}\n"
                f"   Confidence: {conf:.3f} ({int(conf*100)}%)\n"
                f"   Lokasi: ({box[0]}, {box[1]}) - ({box[2]}, {box[3]})\n\n"
            )
        
        txt += "=== PANDUAN ===\n"
        txt += "• Kotak HIJAU = Hasil deteksi sistem\n"
        txt += "• Kotak MERAH = Anotasi Anda\n\n"
        txt += "Silakan buat anotasi dengan menggambar kotak pada area fraktur.\n"
        txt += "Klik 'Save Result' untuk menyimpan hasil."
        
        self.update_results_text(txt)

    def save_guidance_results(self):
        """NEW: Save guidance mode results"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Tidak ada gambar untuk disimpan.")
            return
        
        if self.model_results is None:
            messagebox.showwarning("Warning", "Jalankan deteksi terlebih dahulu.")
            return
        
        try:
            # Generate filenames
            basename = os.path.splitext(os.path.basename(self.current_image_path))[0]
            name_part = f"{basename}"
            
            # 1. Save user annotations in YOLO format
            user_label_path = os.path.join(self.guidance_results_folder, 'labels', name_part + "_user.txt")
            self._save_yolo_label_user_guidance(user_label_path)
            
            # 2. Save X-ray with user BB only
            user_only_path = os.path.join(self.guidance_results_folder, 'images', name_part + "_user_only.jpg")
            user_only_img = self._render_user_only_image()
            user_only_img.save(user_only_path, format='JPEG', quality=95)
            
            # 3. Save X-ray with both user and model BB
            combined_path = os.path.join(self.guidance_results_folder, 'combined', name_part + "_combined.jpg")
            combined_img = self._render_combined_image()
            combined_img.save(combined_path, format='JPEG', quality=95)
            
            # Show success message
            messagebox.showinfo(
                "Simpan Berhasil",
                f"Hasil guidance disimpan:\n"
                f"• Label user: {os.path.basename(user_label_path)}\n"
                f"• Gambar user: {os.path.basename(user_only_path)}\n"
                f"• Gambar gabungan: {os.path.basename(combined_path)}"
            )
            
            self.status_var.set("Hasil guidance berhasil disimpan.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan hasil guidance: {e}")

    def _save_yolo_label_user_guidance(self, label_path):
        """NEW: Save user YOLO labels for guidance mode"""
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        
        if not self.user_boxes:
            # Create empty file if no user annotations
            with open(label_path, 'w') as f:
                pass
            return
        
        img_w, img_h = self.original_image.size
        
        with open(label_path, 'w') as f:
            for box, name in zip(self.user_boxes, self.user_classes):
                # Get class_id from class_list
                class_id = next(cid for cid, cname in self.class_list.items() if cname == name)
                
                x1, y1, x2, y2 = box
                cx = ((x1 + x2) / 2.0) / img_w
                cy = ((y1 + y2) / 2.0) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    def _render_user_only_image(self):
        """NEW: Render image with user bounding boxes only"""
        # Start with original image (convert from PIL to cv2 format)
        img_array = np.array(self.original_image)
        if img_array.ndim == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        
        # Draw user bounding boxes in red
        for box, name in zip(self.user_boxes, self.user_classes):
            x1, y1, x2, y2 = box
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color
            
            # Add label
            label = name if name else "User"
            cv2.putText(img_array, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Convert back to PIL Image
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    def _render_combined_image(self):
        """NEW: Render image with both model and user bounding boxes"""
        # Start with model's annotated image (already has green model BB)
        img_array = self.model_results['annotated_image'].copy()
        
        # Add user bounding boxes in red
        for box, name in zip(self.user_boxes, self.user_classes):
            x1, y1, x2, y2 = box
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color
            
            # Add label
            label = f"User: {name}" if name else "User"
            cv2.putText(img_array, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Convert to PIL Image
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
