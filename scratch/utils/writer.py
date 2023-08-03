import imageio.v2 as iio
import json
import os
import os.path as osp
import threading
import time
from itertools import chain
from typing import Any
from pathlib import Path
from datetime import datetime, timedelta


class MetricWriter():
    def __init__(self, output_path: str='metrics.json', start_iter: int=0, sync_period: int=1):
        """Writes metrics to a file in a pretty and readable format. Supports any file extension."""
        self.output_path = osp.abspath(output_path)

        self._iter = start_iter
        self._init_writer()

        self._sync_period = timedelta(seconds=sync_period)
        self._init_ticker()

    def _init_writer(self):
        self._init_curr_buffer()
        self._writer = Path(self.output_path).open('a', encoding='utf8')
        self._history = []
        self._history_lock = threading.Lock()   # ensure any edits to this list are lock-protected

    def _init_ticker(self):
        self._join = False
        self._last_sync = datetime.now()

        self._ticker = threading.Thread(target=self._sync, daemon=True)
        self._ticker.start()

    def _init_curr_buffer(self):
        self._curr_buffer = {'empty': True}

    def write(self, **kwargs):
        """Add a metric to the current history."""
        metric: Any # support any type of metric logging
        for label, metric in kwargs.items():
            self._curr_buffer[label] = metric

    def step(self):
        """Increment the current iteration and reset the current buffer. Protect the
        history by acquiring/releasing a lock."""
        self._curr_buffer.pop('empty')

        if len(self._curr_buffer) > 0:
            self.write(iter=self._iter)
            self._history_lock.acquire()
            self._history.append(self._curr_buffer)
            self._history_lock.release()

        self._iter += 1
        self._init_curr_buffer()

    def _write_to_disk(self):
        all_lines = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in self._history]
        all_lines = '\n'.join(all_lines) + '\n'
        self._writer.write(all_lines)
        self._writer.flush()

    def _flush_history(self):
        """Flush the current history to the disk. Ensure proper thread safety
        by acquiring/releasing a lock."""
        self._history_lock.acquire()
        if len(self._history):
            self._write_to_disk()
            self._history = []
        self._history_lock.release()

    def _sync(self):
        """Synchronizes the current metrics history with the disk. Managed
        by a daemon ticker thread; should never be called directly."""
        while not self._join:
            if (datetime.now() - self._last_sync) >= self._sync_period:
                self._flush_history()
                self._last_sync = datetime.now()

    def close(self):
        self._join = True
        self._ticker.join()
        self._flush_history()
        self._writer.close()

    def _establish_database(self):
        """Experimenting with sqlite3. For now, stick with json."""
        import sqlite3
        self._conn = sqlite3.connect(self.output_path)
        self._cursor = self._conn.cursor()
        self._cursor.execute(
            'CREATE TABLE test_run (data NUMERIC)'
        )


class VideoWriter():
    def __init__(self, output_path: str='video.mp4', fps: int=30):
        assert output_path.endswith('.mp4'), \
            f"VideoWriter only supports .mp4 files but got {output_path}."
        self.output_path = osp.abspath(output_path)
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


if __name__ == '__main__':
    writers = []
    for i in range(1):
        writers.append(MetricWriter(f'metric{i}.json', seq=False))

    w: MetricWriter
    for i in range(1000):
        for w in writers:
            if i % 30 == 0:
                w.write(loss=i*i)
                w.step(_iter=i+30)

    for w in writers:
        w.close()
