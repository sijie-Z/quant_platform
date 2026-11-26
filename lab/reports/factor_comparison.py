"""Cross-run factor comparison report — knowledge emergence from Registry.

Per Builder-mode discipline (v0.1, no new abstraction, no Runner, no Engine):
this is a minimal script that reads runs from the Registry, flattens their
evaluation + trust metadata, and writes a single Markdown comparison table.
Zero new framework code — just SQL → Markdown.

Trust handling (audit R2/R24/R26/R27): every success-status run is listed,
but the table carries an explicit Validity / affected_by / Trusted column so a
run marked invalidated, affected or superseded is never silently presented as
a clean success. The comparison summary counts every status value, derives the
observed PIT declarations from the rows instead of hardcoding "pit=false", and
stamps the artifact with a generation timestamp and row counts.

Output: data/reports/factor_comparison.md
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from quant_platform.lab.registry import DEFAULT_DB, RunStore


def _json_list(raw) -> list[str]:
    try:
        value = json.loads(raw or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(v) for v in value] if isinstance(value, list) else []


def generate(context: str | None = None) -> str:
    """Generate cross-run comparison and return report path."""
    store = RunStore(DEFAULT_DB)
    runs = store.list_runs(limit=200)

    out = Path("data/reports/factor_comparison.md")
    if not runs:
        out.parent.mkdir(parents=True, exist_ok=True)
        generated_at = datetime.now(UTC).isoformat(timespec="seconds")
        out.write_text(
            "# Factor Comparison\n\n"
            f"_Generated at {generated_at} from Registry `{DEFAULT_DB}`._\n\n"
            "_(No runs in Registry yet.)_\n",
            encoding="utf-8",
        )
        return str(out)

    # --- Aggregate facts purely from Registry records ------------------------
    status_counts = Counter((r.get("status") or "unknown") for r in runs)
    success_runs = [r for r in runs if r.get("status") == "success"]
    invalid_runs = [
        r for r in runs if (r.get("validity") or "valid") != "valid"
    ]

    data_sources = list({r.get("data_source") for r in success_runs if r.get("data_source")})
    universe_providers = list({r.get("universe_provider") for r in success_runs if r.get("universe_provider")})
    adjust_kinds = set()
    pit_declarations: Counter = Counter()
    all_warnings: list[str] = []

    # Build per-run rows
    rows: list[dict] = []
    trusted_count = 0
    for r in success_runs:
        try:
            ev = json.loads(r.get("evaluation") or "{}")
            um = json.loads(r.get("universe_meta") or "{}")
            dm = json.loads(r.get("data_meta") or "{}")
        except json.JSONDecodeError:
            ev, um, dm = {}, {}, {}

        pit = um.get("pit")
        pit_declarations[str(pit)] += 1
        bias_list = um.get("bias_warning") or []
        biases = ", ".join(str(b) for b in bias_list) if isinstance(bias_list, list) else str(bias_list)
        adjust = dm.get("adjust") or "?"

        adjust_kinds.add(adjust)

        # Warnings from the record verbatim
        all_warnings.extend(_json_list(r.get("warnings")))

        validity = r.get("validity") or "valid"
        causes = _json_list(r.get("affected_by"))
        # Synthetic runs are traceable, but they are simulation artifacts, not
        # market evidence: keep them out of the trusted count.
        is_synthetic = str(r.get("data_source") or "").lower() == "synthetic"
        trusted = (
            validity == "valid"
            and store.has_complete_lineage(r)
            and not is_synthetic
        )
        if trusted:
            trusted_count += 1

        rows.append({
            # Legacy / lineage-less rows can have NULL factor or provider;
            # formatting must not crash on exactly the case this report is
            # supposed to flag as untrusted.
            "factor": r.get("factor") or "?",
            "ic": ev.get("ic_mean"),
            "icir": ev.get("icir"),
            "ic_pos": ev.get("ic_positive_ratio"),
            "n_ic_obs": ev.get("n_ic_obs"),
            "sharpe_proxy": ev.get("long_short_sharpe_proxy"),
            "pit": pit,
            "biases": biases,
            "provider": dm.get("provider") or r.get("data_source") or "?",
            "adjust": adjust,
            "validity": validity,
            "affected_by": ", ".join(causes),
            "trusted": (
                "yes" if trusted else ("synthetic" if is_synthetic else "NO")
            ),
        })

    # --- Sort by ICIR descending for the table ---------------------------------
    rows.sort(key=lambda x: x.get("icir") or 0, reverse=True)

    # --- Write Markdown --------------------------------------------------------
    out = Path("data/reports/factor_comparison.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")

    status_line = ", ".join(
        f"{name}={count}" for name, count in sorted(status_counts.items())
    )
    lines = [
        "# Factor Comparison Report",
        "",
        f"**Generated at {generated_at}** from Registry `{DEFAULT_DB}`. "
        f"{len(success_runs)} success-status runs "
        f"({trusted_count} trusted), {len(runs)} total rows.",
        "",
        "## Summary statistics (from Registry)",
        "",
        f"- Total runs: {len(runs)}",
        f"- Status counts: {status_line}",
        f"- Trusted runs (success + validity=valid + complete lineage): {trusted_count}",
        f"- Runs with non-valid validity: {len(invalid_runs)}",
        f"- Data sources: {', '.join(sorted(data_sources)) if data_sources else 'none'}",
        f"- Universe providers: {', '.join(sorted(universe_providers)) if universe_providers else 'none'}",
        f"- Adjustment methods used: {', '.join(sorted(adjust_kinds)) if adjust_kinds else 'none'}",
        "",
    ]

    if invalid_runs:
        lines += [
            "## Validity warnings",
            "",
            "> These runs produced a success status but are no longer trusted. "
            "Do not use them as evidence.",
            "",
        ]
        for r in invalid_runs:
            causes = _json_list(r.get("affected_by"))
            detail = f" — affected_by: {', '.join(causes)}" if causes else ""
            lines.append(
                f"- `{r.get('run_id', '?')}` ({r.get('factor') or '?'}): "
                f"validity={r.get('validity') or 'valid'}{detail}"
            )
        lines.append("")

    lines += [
        "## Per-factor comparison (success-status runs; validity shown per row)",
        "",
        "| Factor          | IC     | ICIR  | IC>0%  | Sharpe proxy | PIT  | Adjust | Provider | Validity | Affected by | Trusted |",
        "|-----------------|--------|-------|--------|--------------|------|--------|----------|----------|-------------|---------|",
    ]
    for r in rows:
        ic = f"{r['ic']:.4f}" if isinstance(r["ic"], (int, float)) else str(r["ic"])
        icir = f"{r['icir']:.4f}" if isinstance(r["icir"], (int, float)) else str(r["icir"])
        ic_pos = f"{r['ic_pos']:.0%}" if isinstance(r["ic_pos"], (int, float)) else str(r["ic_pos"])
        sharpe = f"{r['sharpe_proxy']:.2f}" if isinstance(r["sharpe_proxy"], (int, float)) else str(r["sharpe_proxy"])
        pit = r["pit"]
        trusted = r["trusted"]
        lines.append(
            f"| {r['factor']:<15s} | {ic:<6s} | {icir:<6s} | {ic_pos:<6s} | {sharpe:<12s} | {pit}  | {r['adjust']:<6s} | {r['provider']:<8s} | {r['validity']:<10s} | {r['affected_by']:<11s} | {trusted} |"
        )

    lines += [
        "",
        "## Trust warnings summary",
        "",
    ]
    if all_warnings:
        for i, w in enumerate(sorted(set(all_warnings)), 1):
            lines.append(f"{i}. {w}")
    else:
        lines.append("_(no warnings recorded)_")

    pit_text = ", ".join(
        f"{name}={count}" for name, count in sorted(pit_declarations.items())
    ) or "none"
    lines += [
        "",
        "## Known limitations (from Registry trust fields)",
        "",
        f"- PIT declarations observed across success runs: {pit_text}. "
        "Runs declaring pit=false apply the current constituent snapshot to "
        "history — survivorship bias possible.",
        f"- Adjustment methods vary: {', '.join(sorted(adjust_kinds)) if adjust_kinds else 'none'}.",
        "- DSR marked insufficient_trials for single-factor runs — not formal validation.",
        "- BH-FDR not applicable until multiple factors tested simultaneously.",
        "",
        f"_Generated by lab.reports.factor_comparison at {generated_at} — "
        f"{len(runs)} registry rows. All facts from Registry._",
        "",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    path = generate()
    print(f"Factor comparison: {path}", flush=True)
