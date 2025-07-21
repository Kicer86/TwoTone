
import logging

from typing import Dict


class AttachmentsPicker:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def pick_attachments(self, files_details: Dict):
        picked_attachments = []
        for file, attachments in files_details.items():
            for attachment in attachments:
                picked_attachments.append((file, attachment["tid"]))

        if picked_attachments:
            return [picked_attachments[0]]
        else:
            return []
