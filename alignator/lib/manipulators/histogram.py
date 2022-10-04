import logging
from cv2 import cv2

from .manipulator import Manipulator


logger = logging.getLogger(__name__)


class Histogram(Manipulator):
    """Histogram improver"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.grid_size = self.config[self.class_name]["grid_size"]
        self.clip_limit = self.config[self.class_name]["clip_limit"]

    def manipulate(self):
        logger.debug("Histogram improvments ...")
        gridsize = self.grid_size
        clip_limit = self.grid_size
        lab = cv2.cvtColor(self.image.data, cv2.COLOR_BGR2LAB)
        lab_planes = list(cv2.split(lab))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(gridsize, gridsize))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return image
