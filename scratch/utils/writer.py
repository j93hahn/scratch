import imageio.v2 as iio
import os
from typing import Any
from pathlib import Path


class MetricWriter():
    def __init__(self, output_dir: str='./'):
        self.output_dir = Path(output_dir)

    def _init_writer(self):
        pass

    def add_metric(self, metric: Any):
        pass

    def close(self):
        pass


class VideoWriter():
    def __init__(self, output_path: str='video.mp4', fps: int=30):
        assert output_path.endswith('.mp4'), \
            f"VideoWriter only supports .mp4 files but got {output_path}."
        self.output_path = os.path.abspath(output_path)
        self.fps = fps
        self._init_writer()

    def _init_writer(self):
        self._writer = iio.get_writer(
            self.output_path,
            fps=self.fps,
            mode='I'    # I for multiple images
        )
        self._frame_count = 0

    def add_frame(self, frame: str):
        """Add a frame to the video. Must be in a png, jpg, or jpeg format."""
        assert frame.endswith('.png') or frame.endswith('.jpg') or frame.endswith('.jpeg'), \
            f"Frame must be a .png or .jpg file but got {frame}"
        self._writer.append_data(iio.imread(frame))
        self._frame_count += 1

    def frame_count(self):
        return self._frame_count

    def close(self):
        self._writer.close()
