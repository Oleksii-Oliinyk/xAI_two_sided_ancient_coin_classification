import sys
from contextlib import contextmanager
from pathlib import Path


class Tee:
    """Writes everything to several streams (console + log file)"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def log_stdout_to(log_path):
    """Inside the block print() goes to the console and to log_path"""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(sys.__stdout__, log_file)
        try:
            yield
        finally:
            sys.stdout = sys.__stdout__
