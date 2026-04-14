from __future__ import annotations

import logging
from typing import Any


def emit_event(event: str, **fields: Any) -> None:
    logger = logging.getLogger("shaft.event")
    if fields:
        details = " ".join(f"{k}={v}" for k, v in sorted(fields.items()))
        logger.info("%s %s", event, details, extra={"event": event, "event_fields": fields})
    else:
        logger.info("%s", event, extra={"event": event, "event_fields": {}})
