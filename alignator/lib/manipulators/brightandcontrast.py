import logging
from cv2 import cv2

from .manipulator import Manipulator

logger = logging.getLogger(__name__)


class BrightAndContrast(Manipulator):
    """Automatic bright and contrast"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clip_hist_percent = self.config[self.class_name]["clip_hist_percent"]

    def manipulate(self):
        logger.debug("Adjusting bright and contrast...")
        clip_hist_percent = self.clip_hist_percent

        # Calculate grayscale histogram
        hist = cv2.calcHist([self.image.data_gray], [0], None, [256], [0, 256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= maximum / 100.0
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        auto_result = cv2.convertScaleAbs(self.image.data, alpha=alpha, beta=beta)
        return auto_result
