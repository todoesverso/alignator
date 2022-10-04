import logging
from cv2 import cv2

from .manipulator import Manipulator


logger = logging.getLogger(__name__)


class Resize(Manipulator):
    """Resize image"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.width = self.config[self.class_name]["width"]
        self.height = self.config[self.class_name]["height"]

    def manipulate(self):
        logger.debug("Running resize")
        image = self.image.data
        inter = cv2.INTER_AREA
        dim = None
        (h, w) = image.shape[:2]

        if self.width is None and self.height is None:
            return image

        if self.width is None:
            r = self.height / float(h)
            dim = (int(w * r), self.height)
        else:
            r = self.width / float(w)
            dim = (self.width, int(h * r))

        resized = cv2.resize(image, dim, interpolation=inter)
        return resized
