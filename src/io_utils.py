import cv2
from pathlib import Path
import json
import numpy as np

def load_image(path: str | Path) -> np.ndarray:
    """
    Loads an image from disk and automatically converts it from BGR to RGB.
    """
    path_str = str(path)
    if not Path(path_str).exists():
        raise FileNotFoundError(f"Image not found at path: {path_str}")
        
    img = cv2.imread(path_str)
    if img is None:
        raise ValueError(f"File at {path_str} is corrupted or not a valid image.")
        
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(image: np.ndarray, path: str | Path):
    """
    Saves an RGB image to disk. Handles normalized float normalization and BGR conversion.
    """
    img_to_save = image.copy()
    
    # Convert float [0.0, 1.0] to uint8 [0, 255]
    if img_to_save.dtype != np.uint8:
        img_to_save = (img_to_save * 255).astype(np.uint8)
        
    # Convert from project RGB back to OpenCV BGR format
    if len(img_to_save.shape) == 3 and img_to_save.shape[2] == 3:
        img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
        
    cv2.imwrite(str(path), img_to_save)

def load_json(path: str | Path) -> dict:
    """
    Safely loads and parses a JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: dict | list, path: str | Path, indent: int = 4):
    """
    Saves data to a JSON file, automatically creating parent folders if missing.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
