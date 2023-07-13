import os
import os.path as osp
from typing import List
from pathlib import Path
from .writer import MetricWriter, VideoWriter


_METRIC_STORAGE_STACK = []


def get_metric_storage(level: int=0):
    """
    Args:
        level (int): The logging level to get the :class:`MetricStorage` object for.

    Returns:
        The :class:`MetricStorage` object that's currently being used.
        Throws an error if no :class:`MetricStorage` is currently enabled.
    """
    assert len(
        _METRIC_STORAGE_STACK
    ), "get_metric_storage() has to be called inside a 'with MetricStorage(...)' context!"
    storage: MetricStorage = _METRIC_STORAGE_STACK[-1]
    storage.set_logging_level(level)
    return storage


class MetricStorage():
    def __init__(self, output_dir: str='./'):
        self._CWDIR = (Path(os.getcwd()) / output_dir).resolve()
        self._LOGGING_LEVELS = 1

        self._init_logging_buffer(output_dir)
        self._active = True

    def _init_logging_buffer(self, output_dir):
        self._CURRENT_LOGGING_LEVEL = 0
        self.all_metrics = {
            self._CURRENT_LOGGING_LEVEL: {
                'output_dir': osp.abspath(output_dir)
            }
        }

    def set_logging_level(self, level: int):
        """Set the current logging level."""
        assert level < self._LOGGING_LEVELS, \
            f"Logging level must be less than {self._LOGGING_LEVELS} but got {level}."
        self._CURRENT_LOGGING_LEVEL = level

    def get_logging_level(self) -> List[int]:
        """Return the current logging level and the total number of logging levels."""
        return [self._CURRENT_LOGGING_LEVEL, self._LOGGING_LEVELS]

    def create_logging_level(self, output_dir: str=None):
        """Creates a new logging level and sets it as the current logging level."""
        self._LOGGING_LEVELS += 1
        self.set_logging_level(self._LOGGING_LEVELS - 1)
        if output_dir is None:
            print(f"WARNING: No output directory specified for logging level \
                  {self._CURRENT_LOGGING_LEVEL}, using 'level_{self._CURRENT_LOGGING_LEVEL}'.")
            output_dir = f"level_{self._CURRENT_LOGGING_LEVEL}"
        self.all_metrics[self._CURRENT_LOGGING_LEVEL] = {
            'output_dir': osp.abspath(output_dir)
        }

    def add_metrics_writer(self, metrics: dict):
        ...

    def add_video_writer(self, output_path: str, fps: int):
        """Add a writer to the metric storage."""
        if 'writers' not in self.all_metrics[self._CURRENT_LOGGING_LEVEL].keys():
            self.all_metrics[self._CURRENT_LOGGING_LEVEL]['writers'] = [VideoWriter(output_path, fps)]
        else:
            self.all_metrics[self._CURRENT_LOGGING_LEVEL]['writers'].append(VideoWriter(output_path, fps))

        if output_path not in self.writers:
            self.writers[output_path] = [self._CURRENT_LOGGING_LEVEL, VideoWriter(output_path, fps)]
        else:
            raise ValueError(f"VideoWriter for {output_path} already exists.")
        return self.writers[output_path]

    def retrieve_writer(self, type: str, output_path: str):
        assert type in ['metrics', 'video'], f"Writer type must be 'metrics' or 'video' but got {type}."
        ...

    def _close(self):
        """Close all writers for the current logging level. Should only be called by close_all."""
        if 'writers' in self.all_metrics[self._CURRENT_LOGGING_LEVEL].keys():
            for writer in self.all_metrics[self._CURRENT_LOGGING_LEVEL]['writers']:
                writer.close()

    def close_all(self):
        """Close all writers for all logging levels."""
        if not self._active:
            return
        for i in range(self._LOGGING_LEVELS):
            self._CURRENT_LOGGING_LEVEL = i
            self._close()
        self._active = False

    def __enter__(self):
        assert len(_METRIC_STORAGE_STACK) == 0, \
            "Cannot use MetricStorage as a context manager when it is already in use."
        _METRIC_STORAGE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _METRIC_STORAGE_STACK.pop()
        self.close_all()


if __name__ == '__main__':
    with MetricStorage() as x:
        testW = x.add_writer('test.mp4', 30)
        breakpoint()
        x.close_all()
