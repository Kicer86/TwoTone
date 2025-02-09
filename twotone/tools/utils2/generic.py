
import sys


def hide_progressbar() -> bool:
    return not sys.stdout.isatty() or 'unittest' in sys.modules


def get_tqdm_defaults():
    return {
    'leave': False,
    'smoothing': 0.1,
    'mininterval':.2,
    'disable': hide_progressbar()
}
