
import platform
import os

FramesInfo = dict[int, dict[str, str]]


class DebugRoutines:
    def __init__(self, debug_dir: str, lhs_all_frames: FramesInfo, rhs_all_frames: FramesInfo) -> None:
        self.it = 0
        self.debug_dir = debug_dir
        self.lhs_all_frames = lhs_all_frames
        self.rhs_all_frames = rhs_all_frames
        self.work = platform.system() != "Windows"

    def dump_frames(self, matches: FramesInfo, phase: str) -> None:
        if not self.work:
            return

        target_dir = os.path.join(self.debug_dir, f"#{self.it} {phase}")
        self.it += 1

        os.makedirs(target_dir)

        for i, (ts, info) in enumerate(matches.items()):
            path = info.get("path")
            if path:
                os.symlink(path, os.path.join(target_dir, f"{i:06d}_lhs_{ts:08d}"))

    def dump_matches(self, matches: list[tuple[int, int]], phase: str) -> None:
        if not self.work:
            return

        target_dir = os.path.join(self.debug_dir, f"#{self.it} {phase}")
        self.it += 1

        os.makedirs(target_dir)

        for i, (lhs_ts, rhs_ts) in enumerate(matches):
            lhs_info = self.lhs_all_frames.get(lhs_ts)
            rhs_info = self.rhs_all_frames.get(rhs_ts)
            lhs_path = lhs_info.get("path") if lhs_info else None
            rhs_path = rhs_info.get("path") if rhs_info else None
            if lhs_path:
                os.symlink(lhs_path, os.path.join(target_dir, f"{i:06d}_lhs_{lhs_ts:08d}"))
            if rhs_path:
                os.symlink(rhs_path, os.path.join(target_dir, f"{i:06d}_rhs_{rhs_ts:08d}"))

    def dump_pairs(self, matches: list[tuple[int, int, int, int]]) -> None:
        if not self.work:
            return

        target_dir = os.path.join(self.debug_dir, f"#{self.it} subsegments")
        self.it += 1

        os.makedirs(target_dir)

        for i, (lhs_ts_b, lhs_ts_e, rhs_ts_b, rhs_ts_e) in enumerate(matches):
            for ts, side, tag in [
                (lhs_ts_b, self.lhs_all_frames, f"lhs_b_{lhs_ts_b:08d}"),
                (lhs_ts_e, self.lhs_all_frames, f"lhs_e_{lhs_ts_e:08d}"),
                (rhs_ts_b, self.rhs_all_frames, f"rhs_b_{rhs_ts_b:08d}"),
                (rhs_ts_e, self.rhs_all_frames, f"rhs_e_{rhs_ts_e:08d}"),
            ]:
                info = side.get(ts)
                path = info.get("path") if info else None
                if path:
                    os.symlink(path, os.path.join(target_dir, f"{i:06d}_{tag}"))
