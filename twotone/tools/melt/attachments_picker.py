
from .melt_common import AttachmentRef


class AttachmentsPicker:
    @staticmethod
    def pick_attachments(files_details: dict) -> list[AttachmentRef]:
        for file, attachments in files_details.items():
            if attachments:
                return [AttachmentRef(file, attachments[0]["tid"])]
        return []
