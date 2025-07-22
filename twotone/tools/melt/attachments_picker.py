
import logging

from typing import Dict, List, Tuple, Any


class AttachmentsPicker:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def pick_attachments(self, files_details: Dict[str, List[Dict[str, Any]]]) -> List[Tuple[str, int]]:
        picked_attachments: List[Tuple[str, int]] = []
        for file, attachments in files_details.items():
            for attachment in attachments:
                picked_attachments.append((file, attachment["tid"]))

        if picked_attachments:
            return [picked_attachments[0]]
        else:
            return []
