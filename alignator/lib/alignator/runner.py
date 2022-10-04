import concurrent.futures
import logging
import os
from pathlib import Path


from .image import AlignatorImage
from alignator.lib.manipulators import (
    Align,
    Histogram,
    BrightAndContrast,
    Resize,
    Vignette,
    Video,
)
from alignator.lib.utils import get_images_from_a_path


logger = logging.getLogger(__name__)


class Alignator:
    """The pipeline that manipulates all the images"""

    # The order here is important
    # TODO: Add a "way" to sort the manipulators
    _MANIPULATORS = [Align, Histogram, BrightAndContrast, Vignette, Resize]
    _POST_MANIPULATORS = [Video]

    def __init__(
        self,
        in_path: Path,
        dest_path: Path,
        reference_path: Path,
        config: dict,
    ) -> None:
        self.in_path = in_path
        self.dest_path = dest_path
        self.config = config
        self.config["dest_path"] = dest_path
        self.num_workers = os.environ.get("NUM_WORKERS")
        self.reference_image = AlignatorImage(reference_path, dest_path, self.config)
        self.manipulators = self.get_manipulators(self._MANIPULATORS)
        self.post_manipulators = self.get_manipulators(self._POST_MANIPULATORS)

    def get_manipulators(self, manipulator):
        ret = []
        for p in manipulator:
            class_name = p.__name__.lower()
            class_config = self.config.get(f"{class_name}")
            if not class_config or not class_config.get("disabled", False):
                ret.append(p)
        return ret

    def _run_manipulators(self, align_image):
        for manipulator in self.manipulators:
            logger.info("[%s] %s ...ing", align_image.filename, manipulator.__name__)
            manipulator = manipulator(align_image, self.reference_image, self.config)
            align_image.data = manipulator.manipulate()
        align_image.write()

    def _run_post_manipulators(self):
        for manipulator in self.post_manipulators:
            manipulator = manipulator(None, None, self.config)
            manipulator.manipulate()

    def _run_debug(self):
        # Its easier to have single thread, no concurrent or anything for
        # performance analysis
        for image in get_images_from_a_path(self.in_path):
            align_image = AlignatorImage(Path(image), self.dest_path, self.config)
            self._run_manipulators(align_image)

    def _run_parallel(self):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            futures = []
            for image in get_images_from_a_path(self.in_path):
                logger.info("[%s] Manipulating...", image)
                align_image = AlignatorImage(Path(image), self.dest_path, self.config)
                futures.append(executor.submit(self._run_manipulators, align_image))

            for future in concurrent.futures.as_completed(futures):
                future.result()

    def run(self):
        if os.environ.get("DEBUG", False):
            self._run_debug()
        else:
            self._run_parallel()

        self._run_post_manipulators()
