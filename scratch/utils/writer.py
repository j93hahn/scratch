import imageio.v2 as iio
import json
import os
import threading
import time
import os.path as osp
import sqlite3
from typing import Any
from pathlib import Path
from datetime import datetime, timedelta


class MetricWriter():
    def __init__(self, output_path: str='history.json', sync_period: int=5):
        # assert output_path.endswith('.json'), \
        #     f"MetricWriter only supports .json files but got {output_path}."
        self.output_path = osp.abspath(output_path)
        self.curr_history = []
        self._init_writer()

        self.sync_period = timedelta(seconds=sync_period)
        self._init_ticker()

        # self._establish_database()

    def _establish_database(self):
        self._conn = sqlite3.connect(self.output_path)
        self._cursor = self._conn.cursor()
        self._cursor.execute(
            'CREATE TABLE test_run (data NUMERIC)'
        )

    def _init_ticker(self):
        self._join = False
        self.last_sync = datetime.now()
        self.now = self.last_sync
        self.ticker = threading.Thread(target=self.sync, daemon=True)
        self.ticker.start()

    def _init_writer(self):
        self._writer = open(self.output_path, 'a+', encoding='utf-8')

    def add_metric(self, metric: Any):
        """Add a metric to the current history."""
        self.curr_history.append(metric)

    def flush_history(self):
        self._cursor.execute(
            'INSERT INTO test_run (data) self.curr_history'
        )
        self._conn.commit()
        self.curr_history = []

    def flush(self):
        self._writer.write(json.dumps(self.curr_history))
        self._writer.flush()
        self.curr_history = []

    def sync(self):
        """Synchronizes the current metrics history with the disk."""
        while not self._join:
            self.now = datetime.now()
            if (self.now - self.last_sync) >= self.sync_period:
                print(self.now - self.last_sync)
                self.flush()
                # self.flush_history()
                self.last_sync = self.now

    def close(self):
        if self.curr_history != []:
            self.flush()
        self._join = True
        self.ticker.join()
        self._writer.close()
        # self._conn.close()


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
    x = MetricWriter()
    for i in range(10):
        time.sleep(1)
        x.add_metric(i)
    print("Closing")
    x.close()
