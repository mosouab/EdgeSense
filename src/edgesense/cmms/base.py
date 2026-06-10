"""Core data types and adapter protocol for the CMMS layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class WorkRequest:
    """A vendor-neutral CMMS work request derived from a diagnostic ticket.

    Fields chosen to be the intersection of common CMMS schemas (MaintainX,
    Fiix, Maximo, eMaint, SAP PM). Per-customer adapters do the final
    mapping from this neutral shape into the vendor's exact fields.
    """

    external_id: str        # our reference, also used as filename when persisted
    title: str              # short summary — usually the root cause
    description: str        # long-form text including evidence + recommended action
    urgency: str            # "critical" | "high" | "medium" | "low" | "info"
    priority: int           # 1 (highest) to 5 (lowest); standard CMMS scale
    work_type: str          # "corrective" | "investigation" | "inspection" | "informational"
    asset_id: str           # asset identifier (matches the customer's CMMS hierarchy)
    asset_label: str        # human-readable asset name
    requested_by: str       # who initiated; defaults to the edge device
    requested_at: str       # ISO 8601 UTC timestamp
    category: str           # work-request category ("predictive maintenance")
    recommended_action: str # operator-facing maintenance prompt
    evidence: list[str] = field(default_factory=list)
    failure_mode: str = ""  # matched diagnosis rule name, if any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkRequestResult:
    """Outcome of submitting a work request to a CMMS adapter."""

    request: WorkRequest
    cmms_ref: str           # vendor-side identifier returned after submission
    submitted_at: str       # ISO 8601 UTC timestamp
    cmms_url: str | None = None
    storage_path: str | None = None  # for adapters that persist locally (e.g. mock)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "cmms_ref": self.cmms_ref,
            "cmms_url": self.cmms_url,
            "submitted_at": self.submitted_at,
            "storage_path": self.storage_path,
        }


@runtime_checkable
class CmmsClient(Protocol):
    """Vendor-agnostic CMMS adapter interface.

    Each concrete adapter implements `submit()` to translate the neutral
    `WorkRequest` into the vendor's API call (REST, SOAP, OData, etc.)
    and return a `WorkRequestResult` with the vendor's identifier.
    """

    name: str  # human-readable adapter name, e.g. "MaintainX", "Mock"

    def submit(self, request: WorkRequest) -> WorkRequestResult: ...
