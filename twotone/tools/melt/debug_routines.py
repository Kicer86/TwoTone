
import os

type FramesInfo = Dict[int, Dict[str, str]]


class DebugRoutines:
    def __init__(self, debug_dir: str, lhs_all_frames: FramesInfo, rhs_all_frames: FramesInfo):
        self.it = 0
        self.debug_dir = debug_dir
        self.lhs_all_frames = lhs_all_frames
        self.rhs_all_frames = rhs_all_frames

    def dump_frames(self, matches, phase):
        target_dir = os.path.join(self.debug_dir, f"#{self.it} {phase}")
        self.it += 1

        os.makedirs(target_dir)

        for i, (ts, info) in enumerate(matches.items()):
            path = info["path"]
            os.symlink(path, os.path.join(target_dir, f"{i:06d}_lhs_{ts:08d}"))

    def dump_matches(self, matches, phase):
        target_dir = os.path.join(self.debug_dir, f"#{self.it} {phase}")
        self.it += 1

        os.makedirs(target_dir)

        for i, (lhs_ts, rhs_ts) in enumerate(matches):
            lhs_path = self.lhs_all_frames[lhs_ts]["path"]
            rhs_path = self.rhs_all_frames[rhs_ts]["path"]
            os.symlink(lhs_path, os.path.join(target_dir, f"{i:06d}_lhs_{lhs_ts:08d}"))
            os.symlink(rhs_path, os.path.join(target_dir, f"{i:06d}_rhs_{rhs_ts:08d}"))

    def dump_pairs(self, matches):
        target_dir = os.path.join(self.debug_dir, f"#{self.it} subsegments")
        self.it += 1

        os.makedirs(target_dir)

        for i, (lhs_ts_b, lhs_ts_e, rhs_ts_b, rhs_ts_e) in enumerate(matches):
            lhs_b_path = self.lhs_all_frames[lhs_ts_b]["path"]
            lhs_e_path = self.lhs_all_frames[lhs_ts_e]["path"]
            rhs_b_path = self.rhs_all_frames[rhs_ts_b]["path"]
            rhs_e_path = self.rhs_all_frames[rhs_ts_e]["path"]
            os.symlink(lhs_b_path, os.path.join(target_dir, f"{i:06d}_lhs_b_{lhs_ts_b:08d}"))
            os.symlink(lhs_e_path, os.path.join(target_dir, f"{i:06d}_lhs_e_{lhs_ts_e:08d}"))
            os.symlink(rhs_b_path, os.path.join(target_dir, f"{i:06d}_rhs_b_{rhs_ts_b:08d}"))
            os.symlink(rhs_e_path, os.path.join(target_dir, f"{i:06d}_rhs_e_{rhs_ts_e:08d}"))

