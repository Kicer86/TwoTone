
import logging

from typing import Dict


class AttachmentsPicker:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def pick_attachments(self, files_details: Dict, best_video_file: str):
        """Pick attachments for output.

        If multiple attachments of the same content type are available they are
        considered incomparable. In that case one attachment from
        ``best_video_file`` is chosen and a warning is logged.
        """

        attachments_by_type = {}
        for file, attachments in files_details.items():
            for attachment in attachments:
                ctype = attachment.get("content_type")
                attachments_by_type.setdefault(ctype, []).append((file, attachment))

        picked = []
        for ctype, items in attachments_by_type.items():
            if len(items) == 1:
                file, att = items[0]
                picked.append((file, att["tid"]))
                continue

            # prefer attachment from best_video_file if available
            chosen = None
            for file, att in items:
                if file == best_video_file:
                    chosen = (file, att)
                    break
            if not chosen:
                chosen = items[0]

            self.logger.warning(
                f"Multiple attachments of type {ctype} found; using {chosen[1]['file_name']} from {chosen[0]}"
            )
            picked.append((chosen[0], chosen[1]["tid"]))

        return picked

    def pick_chapter(self, chapters: Dict[str, bool], best_video_file: str):
        """Pick chapter source file."""
        chapter_files = [f for f, has in chapters.items() if has]
        if not chapter_files:
            return None
        chosen = None
        if len(chapter_files) > 1:
            chosen = best_video_file if best_video_file in chapter_files else chapter_files[0]
            self.logger.warning(
                f"Multiple chapter sources found; using chapters from {chosen}"
            )
        else:
            chosen = chapter_files[0]
        return chosen
