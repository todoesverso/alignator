import logging
import numpy as np
from cv2 import cv2

from .manipulator import Manipulator

logger = logging.getLogger(__name__)


class Align(Manipulator):
    """Aligns an image"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.feature_retention = self.config[self.class_name]["feature_retention"]

    def manipulate(self):
        logger.debug("Aligning ...")
        keypoints1, descriptors1 = self.image.keypoints_and_descriptors
        keypoints2, descriptors2 = self.reference_image.keypoints_and_descriptors

        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(descriptors1, descriptors2)

        # Sort matches by score
        matches = sorted(matches, key=lambda x: x.distance)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * float(self.feature_retention))
        matches = matches[:numGoodMatches]

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, _match in enumerate(matches):
            points1[i, :] = keypoints1[_match.queryIdx].pt
            points2[i, :] = keypoints2[_match.trainIdx].pt

        # Find homography
        h, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

        # Use homography
        height, width, _ = self.reference_image.data.shape
        return cv2.warpPerspective(self.image.data, h, (width, height))
