import cv2
import numpy as np

class FloorPlanPreprocessor:

    def __init__(
        self,
        gamma=1.1,
        use_clahe=True,
        use_denoise=True,
        use_sharpen=False,
        use_morphology=True,
        adaptive_threshold=True,
        thresh_block=31, 
        thresh_C=10,
        resize=(512, 512),
        normalize=True,
        kernel=None,
    ):
        self.gamma = gamma
        self.use_clahe = use_clahe
        self.use_denoise = use_denoise
        self.use_sharpen = use_sharpen
        self.use_morphology = use_morphology
        self.adaptive_threshold = adaptive_threshold
        self.thresh_block = thresh_block
        self.thresh_C = thresh_C
        self.resize = resize
        self.normalize = normalize

        self.kernel = kernel if kernel is not None else np.ones((3, 3), dtype=np.uint8)

    def _gamma_correction(self, image, gamma):
        """Private helper method for gamma correction."""
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(256)
        ]).astype("uint8")
        return cv2.LUT(image, table)

    def process(self, image):

        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # CLAHE (contrast normalization)
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        # Gamma correction
        gray = self._gamma_correction(gray, self.gamma)

        # Denoising (edge-preserving)
        if self.use_denoise:
            gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # Sharpening (optional)
        if self.use_sharpen:
            kernel = np.array([[0,-1,0],
                               [-1,5,-1],
                               [0,-1,0]])
            gray = cv2.filter2D(gray, -1, kernel)

        # Adaptive thresholding
        if self.adaptive_threshold:
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.thresh_block,
                self.thresh_C,
            )
        else:
            binary = gray

        # Morphology
        if self.use_morphology:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)

        # Resize
        if self.resize is not None:
            binary = cv2.resize(binary, self.resize, interpolation=cv2.INTER_AREA)

        # Normalize
        if self.normalize:
            binary = binary.astype(np.float32) / 255.0

        return binary