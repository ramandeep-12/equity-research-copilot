"""Upload staging independent of the file uploader's changing selection."""

import hashlib

from report_metadata import company_key


def file_key(content):
    return hashlib.sha256(content).hexdigest()


def merge_pending(pending, additions):
    merged = {item["detail"]["report_id"]: item for item in pending}
    for item in additions:
        merged.setdefault(item["detail"]["report_id"], item)
    return list(merged.values())


def validate_batch(pending, edits=None, active_company=None):
    if not pending:
        raise ValueError("Detect at least one financial report first.")
    detected = {company_key(item["detail"]["company_name"]) for item in pending}
    if len(detected) != 1:
        names = sorted({item["detail"]["company_name"] for item in pending})
        raise ValueError(
            "These reports belong to different companies: "
            + ", ".join(names)
            + ". Remove the unrelated reports. Comparisons require reports from the same issuer."
        )
    if active_company and company_key(active_company) not in detected:
        raise ValueError(
            f"These reports do not belong to the active company, {active_company}. "
            "Choose a new company library or upload reports from the active issuer."
        )
    if edits:
        for item, edit in zip(pending, edits):
            if edit is not None and company_key(edit["company_name"]) != company_key(
                item["detail"]["company_name"]
            ):
                raise ValueError(
                    "The company cannot be changed to a different issuer. Use the detected company identity."
                )
    return next(iter(detected))
