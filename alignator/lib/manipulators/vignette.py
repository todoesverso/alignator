import logging
import numpy as np
from cv2 import cv2

from .manipulator import Manipulator


logger = logging.getLogger(__name__)


class Vignette(Manipulator):
    """Adds vignettes to an image"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sigma = self.config[self.class_name]["sigma"]

    def manipulate(self):
        logger.debug("Vigneter...")
        sigma = self.sigma
        rows, cols = self.image.data.shape[:2]
        zeros = np.copy(self.image.data)
        zeros[:, :, :] = 0
        a = cv2.getGaussianKernel(cols, sigma)
        b = cv2.getGaussianKernel(rows, sigma)
        c = b * a.T
        d = c / c.max()
        zeros[:, :, 0] = self.image.data[:, :, 0] * d
        zeros[:, :, 1] = self.image.data[:, :, 1] * d
        zeros[:, :, 2] = self.image.data[:, :, 2] * d

        return zeros
