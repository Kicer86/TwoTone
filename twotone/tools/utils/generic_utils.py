
import itertools
import logging
import re
import signal
import sys
import threading
import time

from fractions import Fraction
from tqdm import tqdm


DISABLE_PROGRESSBARS = False


def hide_progressbar() -> bool:
    return not sys.stdout.isatty() or DISABLE_PROGRESSBARS


def get_tqdm_defaults():
    return {
    'leave': False,
    'smoothing': 0.1,
    'mininterval':.2,
    'disable': hide_progressbar()
}


def time_to_ms(time_str: str) -> int:
    """ Convert time string 'HH:MM:SS,SSS' to milliseconds """
    h, m, s, ms = re.split(r'[:.,]', time_str)
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms[:3])


def time_to_s(time: str):
    return time_to_ms(time) / 1000


def ms_to_time(ms: int) -> str:
    """ Convert milliseconds to time string 'HH:MM:SS,SSS' """
    h, remainder = divmod(ms, 60*60*1000)
    m, remainder = divmod(remainder, 60*1000)
    s, ms = divmod(remainder, 1000)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{int(ms):03}"


def fps_str_to_float(fps: str) -> float:
    try:
        return float(Fraction(fps))
    except (ZeroDivisionError, ValueError) as exc:
        raise ValueError(f"Invalid fps value: {fps}") from exc


def get_key_position(d: dict, key) -> int:
    for i, k in enumerate(d):
        if k == key:
            return i
    raise KeyError(f"Key {key} not found")


class InterruptibleProcess:
    def __init__(self):
        self._work = True
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        logging.info(f"Got signal #{signum}. Exiting soon.")
        self._work = False

    def _check_for_stop(self):
        if not self._work:
            logging.warning("Exiting now due to received signal.")
            sys.exit(1)


class TqdmBouncingBar:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._animate)
        self._pbar = None

    def _animate(self):
        width = self._pbar.total or 40
        positions = list(range(width)) + list(range(width - 2, -1, -1))
        for pos in itertools.cycle(positions):
            if self._stop_event.is_set():
                break
            self._pbar.n = pos
            self._pbar.refresh()
            time.sleep(self._kwargs.get("delay", 0.05))  # delay can be passed as kwarg

    def __enter__(self):
        self._pbar = tqdm(
            total=self._kwargs.pop("total", 40),
            bar_format=self._kwargs.pop("bar_format", "{desc} |{bar}|"),
            **self._kwargs
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        self._thread.join()
        self._pbar.close()

    def update(self, n=1):
        pass

    def close(self):
        self.__exit__(None, None, None)
