
import sys
import re


def hide_progressbar() -> bool:
    return not sys.stdout.isatty() or 'unittest' in sys.modules


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
    return eval(fps)
