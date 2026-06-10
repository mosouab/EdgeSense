"""Vendor-agnostic CMMS integration layer.

EdgeSense produces diagnostic tickets; this package translates each ticket
into a structured work request and routes it to the customer's CMMS via a
pluggable adapter. A `MockCmmsClient` is included for local demo / dev
use; real adapters (MaintainX, Fiix, Maximo, SAP PM, etc.) plug into the
same `CmmsClient` protocol.
"""

from .base import CmmsClient, WorkRequest, WorkRequestResult
from .builder import build_work_request
from .mock import MockCmmsClient

__all__ = [
    "CmmsClient",
    "WorkRequest",
    "WorkRequestResult",
    "build_work_request",
    "MockCmmsClient",
]
