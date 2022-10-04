import ffmpeg
import logging

from .manipulator import Manipulator

logger = logging.getLogger(__name__)


class Video(Manipulator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dest_path = self.config.get("dest_path")
        self.framerate = self.config[self.class_name]["framerate"]
        self.output_filename = self.config[self.class_name]["output"]

    def manipulate(self):

        stream = ffmpeg.input(
            self.dest_path / "*.jpg", pattern_type="glob", framerate=self.framerate
        )
        stream = ffmpeg.output(
            stream,
            str(self.dest_path / self.output_filename),
            crf=20,
            preset="slower",
            movflags="faststart",
            pix_fmt="yuv420p",
        ).overwrite_output()

        logger.debug("FFMPEG CMD: %s", stream.compile())
        try:
            ffmpeg.run(stream)
        except ffmpeg.Error as e:
            logger.error(e.stderr.decode())
