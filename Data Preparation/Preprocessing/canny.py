import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from PIL import Image

class ImageProcessor:
    """
    A class to apply various edge detection methods (Canny-like, Sobel, ANN)
    and K-means clustering to images.
    """

    def __init__(self):
        self.original_image = None
        self.processed_results = {}
        self.ann_model = None

    def load_image(self, image_path):
        """Load an image from file path"""
        try:
            self.original_image = np.array(Image.open(image_path).convert('RGB'))
            print(f"Image loaded successfully. Shape: {self.original_image.shape}")
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False

    def kmeans_clustering(self, n_clusters=5, random_state=41):
        """K-means clustering for segmentation"""
        img = self.original_image
        pixels = img.reshape(-1, 3)
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = km.fit_predict(pixels)
        centers = km.cluster_centers_
        seg = centers[labels].reshape(img.shape).astype(np.uint8)
        self.processed_results['segmented'] = seg
        return seg

    def visualize_results(self):
        """
        Figure 1: original, canny, segmented
        Figure 2: sobel_h, sobel_v, ann
        """
        # Prepare results
        orig = self.original_image
        seg = self.processed_results['segmented']

        # Figure 1: Original image
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.imshow(orig)
        ax1.set_title('Original Image')
        ax1.axis('off')
        fig1.tight_layout()
        
        # Figure 3: K-means segmentation
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.imshow(seg)
        ax3.set_title('K-means Segmentation')
        ax3.axis('off')
        fig3.tight_layout()
        
        plt.show()

    # --- Helper methods ---

    def _to_grayscale(self, img):
        return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]) if img.ndim == 3 else img

    def _non_maximum_suppression(self, mag, direction):
        suppressed = np.zeros_like(mag)
        angle = (np.degrees(direction) % 180)
        H, W = mag.shape
        for i in range(1, H-1):
            for j in range(1, W-1):
                q = r = 0
                a = angle[i, j]
                if (0 <= a < 22.5) or (157.5 <= a <= 180):
                    q, r = mag[i, j+1], mag[i, j-1]
                elif 22.5 <= a < 67.5:
                    q, r = mag[i+1, j-1], mag[i-1, j+1]
                elif 67.5 <= a < 112.5:
                    q, r = mag[i+1, j], mag[i-1, j]
                else:
                    q, r = mag[i-1, j-1], mag[i+1, j+1]
                suppressed[i, j] = mag[i, j] if mag[i, j] >= q and mag[i, j] >= r else 0
        return suppressed

    def _hysteresis_threshold(self, img, low_ratio, high_ratio):
        high = img.max() * high_ratio
        low = high * low_ratio
        res = np.zeros_like(img, dtype=np.uint8)
        strong, weak = 255, 75
        s_i, s_j = np.where(img >= high)
        w_i, w_j = np.where((img < high) & (img >= low))
        res[s_i, s_j] = strong
        res[w_i, w_j] = weak
        H, W = img.shape
        for i in range(1, H-1):
            for j in range(1, W-1):
                if res[i, j] == weak and np.any(res[i-1:i+2, j-1:j+2] == strong):
                    res[i, j] = strong
        return res

    def process_all(self):
        """Run all methods in sequence"""
        if self.original_image is None:
            raise RuntimeError("No image loaded")
        self.kmeans_clustering()

if __name__ == "__main__":
    processor = ImageProcessor()
    # Ganti dengan path gambar Anda
    if not processor.load_image("Datasets/Uji/N (63).png"):
        raise RuntimeError("Failed to load image")
    processor.process_all()
    processor.visualize_results()
