
import logging

from .melt_common import AttachmentRef


class AttachmentsPicker:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def pick_attachments(self, files_details: dict) -> list[AttachmentRef]:
        picked_attachments: list[AttachmentRef] = []
        for file, attachments in files_details.items():
            for attachment in attachments:
                picked_attachments.append(AttachmentRef(file, attachment["tid"]))

        if picked_attachments:
            return [picked_attachments[0]]
        else:
            return []
