import cv2
import numpy as np
from typing import Tuple, List, Dict, Any

class CoinGeometry:
    """
    Encapsulates the geometric logic for converting bboxes into circular/elliptical
        masks and physical properties.
    """

    @staticmethod
    def calculate_radius(width: float, height: float) -> float:
        """
        Calculate radius based on the max-dimension of the bounding box
        to account for slant. This is a heuristic and can be adjusted based on the dataset.

        Thought process:
        - For perfectly circular coins, the radius should be half of the width or height (since they are equal).
        - The longest dimension is the best approximation of the diameter, especially for slanted coins.
        """
        return max(width, height) / 2.0
    
    @staticmethod
    def get_ellipse_params(bbox: List[float]) -> Dict[str, Any]:
        """
        Converts a [x,y,width,height] bbox into parameters for an ellipse (center, axes, angle).

        Returns:
            A dictionary with keys: 'center', 'axes', 'angle' (cv2.ellipse format).
        """
        x, y, w, h = bbox
        center = (int(x + w / 2), int(y + h / 2))
        axes = (int(w / 2), int(h / 2))
        return {
            'center': center,
            'axes': axes,
            'angle': 0.0  # Assuming coins are aligned with the axes; for simplicity.
        }
    
    @classmethod
    def generate_mask(cls, bbox: List[float], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Generates a binary mask for given coin using idea of ellipse fitting.

        Args:
            bbox: The bounding box [x, y, width, height] of the coin.
            image_shape: The shape of the original image (height, width) to create the mask.
        Returns:
            np.ndarray: Binary mask (0s and 255s).
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        params = cls.get_ellipse_params(bbox)

        #form the ellips
        cv2.ellipse(mask, params['center'], params['axes'], params['angle'], 0, 360, 255, -1)
        return mask
    
    @classmethod
    def analyze_detection(cls, bbox: List[float]) -> Dict[str, Any]:
        """
        Wrapper to return all geometric properties (extensible in the future).

        Returns:
            A dictionary with geometric properties of the coin.
        """
        width = bbox[2]
        height = bbox[3]

        return {
            'radius': cls.calculate_radius(width, height),
            "center_point": (bbox[0] + width / 2, bbox[1] + height / 2),
            "aspect_ratio": width / height if height != 0 else 0,
            "is_slanted": width / height < 0.8 or width / height > 1.2
        }