import os
import os.path as osp
from typing import List
from pathlib import Path
from writer import MetricWriter, VideoWriter


_METRIC_STORAGE_STACK = []


def get_metric_storage():
    """Returns:
        The :class:`MetricStorage` object that's currently being used.
        Throws an error if no :class:`MetricStorage` is currently enabled.
    """
    assert len(
        _METRIC_STORAGE_STACK
    ), "get_metric_storage() has to be called inside a 'with MetricStorage(...)' context!"
    storage: MetricStorage = _METRIC_STORAGE_STACK[-1]
    return storage


class MetricStorage():
    def __init__(
        self,
        output_dir: str='./'
    ) -> None:
        self._LOGGING_LEVELS = 1

        self._init_base_logging(output_dir)
        self._active = True

    def _init_base_logging(self, output_dir):
        self._BASE_DIR = osp.abspath(output_dir)
        os.makedirs(self._BASE_DIR, exist_ok=True)
        self._CURRENT_LOGGING_LEVEL = 0
        self._all_metrics = {
            self._CURRENT_LOGGING_LEVEL: {
                # output_dir locates the directory where all the metrics for the
                # current logging level are stored and is always an absolute path.
                'output_dir': self._BASE_DIR
            }
        }

    def add_logging_level(self, output_dir: str="no_use_default"):
        """Creates a new logging level/directory and sets it as the current logging level relative
        to the _BASE_DIR logging level created at initialization"""
        self._LOGGING_LEVELS += 1
        self._set_logging_level(self._LOGGING_LEVELS - 1)

        if output_dir == "no_use_default":
            print(f"WARNING: No output directory specified for logging level \
                  {self._CURRENT_LOGGING_LEVEL}, using 'level_{self._CURRENT_LOGGING_LEVEL}'.")
            output_dir = f"level_{self._CURRENT_LOGGING_LEVEL}"
        parent = self._get_level_metrics(self._CURRENT_LOGGING_LEVEL - 1)['output_dir']
        output_dir = Path(osp.join(parent, output_dir)).resolve().__str__()
        assert not osp.exists(output_dir), \
            f"Output directory {output_dir} already exists."

        os.makedirs(output_dir, exist_ok=True)
        self._all_metrics[self._CURRENT_LOGGING_LEVEL] = {
            'output_dir': output_dir
        }

    def add_metrics_writer(self, output_path: str="metrics.json", level: int=-1) -> MetricWriter:
        """Adds a metrics writer to the metric storage and returns the metrics writer object."""
        if level == -1:
            # return the metrics writer for the deepest logging level
            level = self.get_level_count() - 1

        self._set_logging_level(level)
        metric = self._get_level_metrics(level)
        output_path = (Path(osp.join(metric['output_dir'], output_path))).resolve().__str__()

        if 'metrics' not in metric.keys():
            _new_writer = MetricWriter(output_path)
            metric['metrics'] = [_new_writer]
            return _new_writer

        writer: MetricWriter
        for writer in metric['metrics']:
            assert writer.output_path != output_path, \
                f"Metrics writer with output path {output_path} already exists."

        _new_writer = MetricWriter(output_path)
        metric['metrics'].append(_new_writer)
        return _new_writer

    def add_video_writer(self, output_path: str="video.mp4", fps: int=30, level: int=-1) -> VideoWriter:
        """Adds a video writer to the metric storage and returns the video writer object."""
        if level == -1:
            # return the metrics writer for the deepest logging level
            level = self.get_level_count() - 1

        self._set_logging_level(level)
        metric = self._get_level_metrics(level)
        output_path = (Path(osp.join(metric['output_dir'], output_path))).resolve().__str__()

        if 'videos' not in metric.keys():
            _new_writer = VideoWriter(output_path, fps)
            metric['videos'] = [_new_writer]
            return _new_writer

        writer: VideoWriter
        for writer in metric['videos']:
            assert writer.output_path != output_path, \
                f"Video writer with output path {output_path} already exists."

        _new_writer = VideoWriter(output_path, fps)
        metric['videos'].append(_new_writer)
        return _new_writer

    def get_level_count(self) -> int:
        """Returns the number of logging levels."""
        return self._LOGGING_LEVELS

    def _set_logging_level(self, level: int):
        """Set the current logging level."""
        assert level < self._LOGGING_LEVELS, \
            f"Logging level must be less than {self._LOGGING_LEVELS} but got {level}."
        self._CURRENT_LOGGING_LEVEL = level

    def _get_level_metrics(self, level: int) -> dict:
        """Returns the metrics dictionary at the specified level."""
        return self._all_metrics[self._CURRENT_LOGGING_LEVEL]

    def _close(self):
        """Close all writers for the current logging level. Should only be called by _close_all."""
        metric = self._get_level_metrics(self._CURRENT_LOGGING_LEVEL)
        if 'metrics' in metric.keys():
            writer: MetricWriter
            for writer in metric['metrics']:
                writer.close()

        if 'videos' in metric.keys():
            writer: VideoWriter
            for writer in metric['videos']:
                writer.close()

    def _close_all(self):
        """Close all writers for all logging levels. Should only be called by __exit__."""
        if not self._active:
            return
        for i in range(self._LOGGING_LEVELS):
            self._set_logging_level(i)
            self._close()
        self._active = False

    def __enter__(self):
        assert len(_METRIC_STORAGE_STACK) == 0, \
            "Cannot use MetricStorage as a context manager when it is already in use."
        _METRIC_STORAGE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _METRIC_STORAGE_STACK.pop()
        self._close_all()


def main():
    # with MetricStorage('test') as m:
    #     writer = m.add_metrics_writer()
    #     # breakpoint()
    #     m.add_logging_level('inner_loop')
    #     m.add_logging_level('whoop')
    #     for i in range(10):
    #         writer.write(val=i, val_squared=i**2)
    #         writer2 = m.add_metrics_writer(f'inner_*{i}.txt', level=1)
    #         for j in range(5):
    #             writer2.write(val=j, val_squared=j**2)
    #             writer2.step()
    #             writer3 = m.add_metrics_writer(f'inner_{j}.json', level=2)
    #             for k in range(3):
    #                 writer3.write(val=k, val_squared=k**2)
    #                 writer3.step()
    #         writer.write(exited_loop=i+i)
    #         writer.step()
    with MetricStorage('test') as m:
        writer = m.add_metrics_writer()
        m.add_logging_level('inner_loop')
        for i in range(10):
            writer.write(val=i, val_squared=i**2)
            writer.step()
            for j in range(5):
                writer2 = m.add_metrics_writer(f'inner_{j}.txt', level=1)
                writer2.write(val=j, val_squared=j**2)
                writer2.step()



def main2():
    from fabric.utils.event import EventStorage
    with EventStorage('whc') as storage:
        for i in range(10):
            storage.put_scalars(val=i)
            # storage.step()
            with EventStorage('first_inner_loop') as storage2:
                for j in range(3):
                    storage2.put_scalars(val=j)
                    # storage2.step()
                    with EventStorage('second_inner_loop') as storage3:
                        for k in range(3):
                            storage3.put_scalars(val=k)
                            storage3.step()
                    storage2.step()
            storage.step()


if __name__ == '__main__':
    # main()
    main2()
