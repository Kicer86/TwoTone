
import os
import platform

from collections.abc import Callable
from typing import Literal

from .melt_common import FramesInfo, format_timestamp_range
Discontinuity = tuple[int, int, int, int, int]
DiscontinuityGroup = list[Discontinuity]


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

    def dump_content_discontinuities(
        self,
        groups: list[DiscontinuityGroup],
        *,
        lhs_label: str,
        rhs_label: str,
        lhs_fps: float,
        rhs_fps: float,
        prepare_frames: Callable[[], None],
        compensation_tolerance_frames: float = 4.0,
    ) -> str | None:
        """Write final error details and complete frame spans for every region."""
        return self._dump_discontinuities(
            groups,
            lhs_label=lhs_label,
            rhs_label=rhs_label,
            lhs_fps=lhs_fps,
            rhs_fps=rhs_fps,
            prepare_frames=prepare_frames,
            diagnostic_type="content",
            compensation_tolerance_frames=compensation_tolerance_frames,
        )

    def dump_timing_discontinuities(
        self,
        groups: list[DiscontinuityGroup],
        *,
        lhs_label: str,
        rhs_label: str,
        lhs_fps: float,
        rhs_fps: float,
        prepare_frames: Callable[[], None],
    ) -> str | None:
        """Write complete frame spans for unsupported matched-timeline steps."""
        return self._dump_discontinuities(
            groups,
            lhs_label=lhs_label,
            rhs_label=rhs_label,
            lhs_fps=lhs_fps,
            rhs_fps=rhs_fps,
            prepare_frames=prepare_frames,
            diagnostic_type="timing",
        )

    def _dump_discontinuities(
        self,
        groups: list[DiscontinuityGroup],
        *,
        lhs_label: str,
        rhs_label: str,
        lhs_fps: float,
        rhs_fps: float,
        prepare_frames: Callable[[], None],
        diagnostic_type: Literal["content", "timing"],
        compensation_tolerance_frames: float = 4.0,
    ) -> str | None:
        if not self.work:
            return None

        prepare_frames()
        target_dir = os.path.join(
            self.debug_dir,
            f"#{self.it} {diagnostic_type} discontinuity error",
        )
        self.it += 1
        os.makedirs(target_dir)

        if diagnostic_type == "content":
            details = [
                "Content discontinuity diagnostics",
                "",
                "Every decoded frame in each inclusive region is linked below.",
                "A compensating region restores the outer alignment but still means",
                "that the videos contain different local frame sequences.",
            ]
        else:
            details = [
                "Timing discontinuity diagnostics",
                "",
                "Every decoded frame in each inclusive region is linked below.",
                "These matched regions change the observed timeline relation and",
                "cannot be represented by one supported audio transform.",
            ]
        details.extend([
            f"{lhs_label} nominal frame rate: {lhs_fps:.6f} fps",
            f"{rhs_label} nominal frame rate: {rhs_fps:.6f} fps",
            "",
        ])
        for region_index, group in enumerate(groups):
            lhs_from, _, rhs_from, _, _ = group[0]
            _, lhs_to, _, rhs_to, _ = group[-1]
            frame_errors = [
                deficit_ms * rhs_fps / 1000
                for *_, deficit_ms in group
            ]
            compensating = (
                len(group) > 1
                and min(frame_errors) < 0 < max(frame_errors)
                and abs(sum(frame_errors)) <= compensation_tolerance_frames
            )
            if diagnostic_type == "timing":
                kind = "unresolved matched-timeline step"
            else:
                kind = (
                    "compensating adjacent frame-step errors"
                    if compensating
                    else "unresolved frame-step error"
                )
            lhs_first_id = int(self.lhs_all_frames[lhs_from]["frame_id"])
            lhs_last_id = int(self.lhs_all_frames[lhs_to]["frame_id"])
            rhs_first_id = int(self.rhs_all_frames[rhs_from]["frame_id"])
            rhs_last_id = int(self.rhs_all_frames[rhs_to]["frame_id"])

            details.extend([
                f"Region {region_index}: {kind}",
                (
                    f"  {lhs_label}: {format_timestamp_range(lhs_from, lhs_to)}; "
                    "decoded frames "
                    f"{lhs_first_id} -> {lhs_last_id}"
                ),
                (
                    f"  {rhs_label}: {format_timestamp_range(rhs_from, rhs_to)}; "
                    "decoded frames "
                    f"{rhs_first_id} -> {rhs_last_id}"
                ),
                "  Steps:",
            ])
            for step_index, (
                step_lhs_from,
                step_lhs_to,
                step_rhs_from,
                step_rhs_to,
                deficit_ms,
            ) in enumerate(group):
                step_lhs_first = int(self.lhs_all_frames[step_lhs_from]["frame_id"])
                step_lhs_last = int(self.lhs_all_frames[step_lhs_to]["frame_id"])
                step_rhs_first = int(self.rhs_all_frames[step_rhs_from]["frame_id"])
                step_rhs_last = int(self.rhs_all_frames[step_rhs_to]["frame_id"])
                details.append(
                    f"    {step_index + 1}: {lhs_label} {step_lhs_first}->{step_lhs_last} "
                    f"at {format_timestamp_range(step_lhs_from, step_lhs_to)} <-> "
                    f"{rhs_label} {step_rhs_first}->{step_rhs_last} "
                    f"at {format_timestamp_range(step_rhs_from, step_rhs_to)}; "
                    f"error {deficit_ms:+d} ms "
                    f"({deficit_ms * rhs_fps / 1000:+.1f} rhs frames)"
                )
            details.append("")

            region_dir = os.path.join(target_dir, f"region_{region_index:03d}")
            self._link_frame_span(
                os.path.join(region_dir, "lhs"),
                self.lhs_all_frames,
                lhs_from,
                lhs_to,
            )
            self._link_frame_span(
                os.path.join(region_dir, "rhs"),
                self.rhs_all_frames,
                rhs_from,
                rhs_to,
            )

        with open(os.path.join(target_dir, "details.txt"), "w", encoding="utf-8") as details_file:
            details_file.write("\n".join(details))
        return target_dir

    @staticmethod
    def _link_frame_span(
        target_dir: str,
        frames: FramesInfo,
        start_ms: int,
        end_ms: int,
    ) -> None:
        os.makedirs(target_dir)
        for timestamp, info in sorted(frames.items()):
            if not start_ms <= timestamp <= end_ms:
                continue
            path = info.get("path")
            if path is None:
                continue
            extension = os.path.splitext(path)[1]
            name = (
                f"frame_{int(info['frame_id']):06d}_"
                f"{timestamp:010d}{extension}"
            )
            os.symlink(path, os.path.join(target_dir, name))
