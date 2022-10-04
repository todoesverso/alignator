import typer
from pathlib import Path
from alignator.lib.alignator import Alignator
from alignator.lib.utils import (
    is_a_valid_path,
    reference_is_a_valid_image,
    setup_logging,
    Config,
)
import logging

logger = logging.getLogger(__name__)


def main(
    reference: Path = typer.Option(...),
    source: Path = typer.Option(...),
    dest: Path = typer.Option(...),
    features: int = typer.Option(
        5000, help="Number of features to extract from the images"
    ),
    align_feature_retention: float = typer.Option(
        0.15, help="Percentage of features to keep"
    ),
    align_disabled: bool = typer.Option(False, help="Disable align manipulation"),
    vignette_sigma: int = typer.Option(1200, help="Vignetter sigma value"),
    vignette_disabled: bool = typer.Option(False, help="Disable vignette manipulation"),
    histogram_grid_size: int = typer.Option(10, help="Histogram grid size"),
    histogram_clip_limit: float = typer.Option(1.5, help="Histogram clip limit"),
    histogram_disabled: bool = typer.Option(
        False, help="Disable histogram manipulation"
    ),
    brightandcontrast_clip_hist_percent: float = typer.Option(
        1.0, help="Brightness and Contrast clip histogram"
    ),
    brightandcontrast_disabled: bool = typer.Option(
        False, help="Disable brigntness and contrast manipulation"
    ),
    resize_width: int = typer.Option(None, help="New width og the image"),
    resize_height: int = typer.Option(None, help="New Height of the image"),
    video_framerate: float = typer.Option(1.0, help="FrameRate of the generated video"),
    video_disabled: bool = typer.Option(False, help="Disable video generation"),
    video_output: str = typer.Option("output.mp4", help="Video output file name"),
):
    reference_is_a_valid_image(reference)
    is_a_valid_path(source)
    is_a_valid_path(dest)
    setup_logging()
    config = Config(locals())
    logger.debug("CONFIG:")
    logger.debug(config.as_str)
    alignator = Alignator(source, dest, reference, config.as_dict)
    alignator.run()


def cli():
    typer.run(main)


if __name__ == "__main__":
    cli()
