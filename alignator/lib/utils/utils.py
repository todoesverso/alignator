import imghdr
import typer
import logging
from cv2.cv2 import imread
import os
from pathlib import Path
from typing import Generator
from time import time


logger = logging.getLogger(__name__)


def decotimeit(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        logger.debug("Function %s executed in: %.4f seconds", func, t2 - t1)
        return result

    return wrap_func


def get_images_from_a_path(
    path: Path, msorted: bool = False
) -> Generator[str, None, None]:
    """
    Return the list of images in a directory.

    :param path Path: The path to look for images
    :param msorted bool: If True the files will be sorted by modification time

    :return Generator: A generator with the sorted list of images
    """
    assert path.is_dir()

    files = os.listdir(path)

    if msorted:
        files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(path, x)))

    for f in files:
        full = os.path.join(path, f)
        if os.path.isfile(full) and imghdr.what(full):
            yield full


def reference_is_a_valid_image(reference: Path):
    if reference is None or reference.is_dir() or not reference.exists():
        typer.echo("A reference image is mandatory!")
        raise typer.Abort()

    if not reference.exists():
        typer.echo(f"{reference} is not a valid reference image")
        raise typer.Abort()

    if reference.is_file():
        try:
            imread(str(reference))
        except TypeError:
            typer.echo(f"{reference} is not a valid reference image")
            raise typer.Abort()


def is_a_valid_path(path: Path):
    if not path.is_dir() or not path.exists():
        typer.echo(f"{path} must be a valid path")
        raise typer.Abort()
