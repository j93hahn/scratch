import imageio.v2 as iio
import os


class VideoWriter():
    def __init__(self, output_path='video.mp4', fps=30):
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
        assert frame.endswith('.png') or frame.endswith('.jpg')
        self._writer.append_data(iio.imread(frame))
        self._frame_count += 1

    def frame_count(self):
        return self._frame_count

    def close(self):
        self._writer.close()


"""TODO: Create metric logger class that can be used to log metrics to a file."""
class Metric():
    def __init__(self, name: str, label: str, color: str):
        self.name = name
        self.label = label
        self.color = color
        self._values = []

    def add_value(self, value: float):
        self._values.append(value)

    def values(self):
        return self._values
