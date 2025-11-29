
from dataclasses import dataclass


@dataclass(kw_only=True)
class SubtitleCommonData:
    name: str | None = None
    language: str | None = None
    default: int | bool = False
    format: str | None = None

# for subtitle tracks in files
@dataclass(kw_only=True)
class Subtitle(SubtitleCommonData):
    length: int | None = None
    tid: int | None = None
    format: str | None = None

# for files
@dataclass(kw_only=True)
class SubtitleFile(SubtitleCommonData):
    path: str | None = None
    encoding: str | None = None
