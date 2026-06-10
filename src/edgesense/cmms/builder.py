"""Translate an EdgeSense diagnostic ticket into a neutral CMMS work request."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .base import WorkRequest

_PRIORITY_BY_URGENCY = {
    "critical": 1,
    "high": 2,
    "medium": 3,
    "low": 4,
    "info": 5,
}
_WORK_TYPE_BY_URGENCY = {
    "critical": "corrective",
    "high": "investigation",
    "medium": "inspection",
    "low": "informational",
    "info": "informational",
}


def build_work_request(
    *,
    diagnosis: dict[str, Any],
    contributors: list[dict[str, Any]] | None,
    forecast: dict[str, Any] | None,
    asset_id: str,
    asset_label: str,
    score: float | None,
    threshold: float | None,
    elapsed_simulated_seconds: float | None,
    requested_by: str = "EdgeSense Edge Device",
) -> WorkRequest:
    """Build a CMMS-ready work request from the current diagnosis context."""

    contributors = contributors or []

    urgency = str(diagnosis.get("urgency") or "info").lower()
    priority = _PRIORITY_BY_URGENCY.get(urgency, 5)
    work_type = _WORK_TYPE_BY_URGENCY.get(urgency, "informational")

    title = str(diagnosis.get("root_cause") or "Anomaly detected by EdgeSense")
    recommended_action = str(
        diagnosis.get("recommended_action") or "Investigate the flagged channels."
    )
    evidence: list[str] = [str(e) for e in (diagnosis.get("evidence") or [])]
    failure_mode = str(diagnosis.get("matched_rule") or "")

    description_parts: list[str] = [
        f"Root cause: {title}",
        f"Urgency: {diagnosis.get('urgency_label') or urgency}",
        f"Asset: {asset_label} [{asset_id}]",
        "",
        "Evidence:",
    ]
    if evidence:
        description_parts.extend(f"  • {item}" for item in evidence)
    else:
        description_parts.append("  (no specific channels singled out)")

    description_parts.extend(["", "Recommended action:", f"  {recommended_action}"])

    if failure_mode:
        description_parts.extend(["", f"Matched diagnostic rule: {failure_mode}"])

    if forecast and forecast.get("status") == "trending_up":
        ttt = forecast.get("time_to_alert_seconds")
        if isinstance(ttt, (int, float)):
            description_parts.extend(
                ["", f"Forecast: alert projected in {_humanise_seconds(ttt)}."]
            )
            lo = forecast.get("time_to_alert_low_seconds")
            hi = forecast.get("time_to_alert_high_seconds")
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                description_parts.append(
                    f"95% confidence band: {_humanise_seconds(lo)} → {_humanise_seconds(hi)}."
                )
    elif forecast and forecast.get("status") == "above_threshold":
        description_parts.extend(
            ["", "Forecast: score is already above the deployable threshold."]
        )

    description = "\n".join(description_parts)

    now = datetime.now(timezone.utc)
    external_id = f"EDGE-{now.strftime('%Y%m%dT%H%M%SZ')}"

    metadata: dict[str, Any] = {
        "edgesense_source": asset_id,
        "edgesense_score": score,
        "edgesense_threshold": threshold,
        "edgesense_elapsed_simulated_seconds": elapsed_simulated_seconds,
        "edgesense_forecast": forecast,
        "top_contributors": [
            {
                "name": c.get("name"),
                "label": c.get("label"),
                "delta_pct": c.get("delta_pct"),
                "current_pct": c.get("current_pct"),
                "baseline_pct": c.get("baseline_pct"),
            }
            for c in contributors[:5]
        ],
    }

    return WorkRequest(
        external_id=external_id,
        title=title,
        description=description,
        urgency=urgency,
        priority=priority,
        work_type=work_type,
        asset_id=asset_id,
        asset_label=asset_label,
        requested_by=requested_by,
        requested_at=now.isoformat(),
        category="predictive maintenance",
        recommended_action=recommended_action,
        evidence=evidence,
        failure_mode=failure_mode,
        metadata=metadata,
    )


def _humanise_seconds(seconds: float) -> str:
    """Render a seconds duration as the most legible single-unit string."""

    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    if seconds < 3600:
        return f"{seconds / 60:.0f} minutes"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} hours"
    if seconds < 60 * 86400:
        return f"{seconds / 86400:.1f} days"
    return f"{seconds / (30 * 86400):.1f} months"
