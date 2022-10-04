from abc import ABC, abstractmethod
import numpy as np


class Manipulator(ABC):
    """Manipulates an image and overwrites it"""

    def __init__(
        self,
        image: "AlignatorImage",
        reference_image: "AlignatorImage",
        config: dict = {},
    ) -> None:
        self.image = image
        self.reference_image = reference_image
        self.config = config
        self.class_name = self.__class__.__name__.lower()

    @abstractmethod
    def manipulate(self) -> np.ndarray:
        pass
