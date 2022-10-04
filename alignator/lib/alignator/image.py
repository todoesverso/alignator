from pathlib import Path
import numpy as np
from cv2 import cv2
import logging
import os

logger = logging.getLogger(__name__)


class AlignatorImage:

    EXTENSION = "jpg"

    def __init__(self, path: Path, dest_path: Path, config: dict) -> None:
        self.path = path
        self.filename = path.name
        self.dest_path = dest_path
        self._data_gray = None
        self._data = None
        self._keypoints_and_descriptors = None
        self.features = config.get("features")

    @property
    def data(self) -> np.ndarray:
        if self._data is not None:
            return self._data

        logger.debug("Reading image")
        self._data = cv2.imread(str(self.path))
        return self._data

    @data.setter
    def data(self, _data) -> None:
        self._data = _data

    @property
    def data_gray(self) -> np.ndarray:
        if self._data_gray is not None:
            return self._data_gray
        logger.debug("Reading gray image")
        self._data_gray = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
        return self._data_gray

    @property
    def keypoints_and_descriptors(self) -> tuple:
        if self._keypoints_and_descriptors is not None:
            return self._keypoints_and_descriptors
        logger.debug("Setting Keypoints and Descriptors")
        detect = cv2.ORB_create(self.features)
        self._keypoints_and_descriptors = detect.detectAndCompute(self.data, None)
        return self._keypoints_and_descriptors

    def _get_dest_path(self) -> str:
        # score = self.score or '0'
        return os.path.join(self.dest_path, self.filename)

    def write(self):
        dest = self._get_dest_path()
        logger.debug("Writing image to: %s", dest)
        cv2.imwrite(dest, self.data, [cv2.IMWRITE_JPEG_QUALITY, 100])
