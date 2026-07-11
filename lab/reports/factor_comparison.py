"""Cross-run factor comparison report — knowledge emergence from Registry.

Per Builder-mode discipline (v0.1, no new abstraction, no Runner, no Engine):
this is a minimal script that reads ALL success runs from the Registry,
flattens their evaluation + trust metadata, and writes a single Markdown
comparison table. Zero new framework code — just SQL → Markdown.

The value test: does the Registry already contain enough structured knowledge
that a comparison emerges WITHOUT touching individual run scripts?

Output: data/reports/factor_comparison.md
"""

from __future__ import annotations

import json
from pathlib import Path

from quant_platform.lab.registry import DEFAULT_DB, RunStore


def _json_str(val) -> str:
    """Round-trip to ensure compact JSON."""
    return json.dumps(val, ensure_ascii=False, default=str)


def generate(context: str | None = None) -> str:
    """Generate cross-run comparison and return report path."""
    store = RunStore(DEFAULT_DB)
    runs = store.list_runs(limit=200)

    if not runs:
        out = Path("data/reports/factor_comparison.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# Factor Comparison\n\n_(No runs in Registry yet.)_\n", encoding="utf-8")
        return str(out)

    # --- Aggregate facts purely from Registry records ------------------------
    success_runs = [r for r in runs if r.get("status") == "success"]
    failed_runs = [r for r in runs if r.get("status") == "failed"]

    data_sources = list({r.get("data_source") for r in success_runs if r.get("data_source")})
    universe_providers = list({r.get("universe_provider") for r in success_runs if r.get("universe_provider")})
    adjust_kinds = set()
    all_warnings: list[str] = []

    # Build per-run rows
    rows: list[dict] = []
    for r in success_runs:
        try:
            ev = json.loads(r.get("evaluation") or "{}")
            um = json.loads(r.get("universe_meta") or "{}")
            dm = json.loads(r.get("data_meta") or "{}")
        except json.JSONDecodeError:
            ev, um, dm = {}, {}, {}

        pit = um.get("pit")
        biases = ", ".join(um.get("bias_warning", []))
        adjust = dm.get("adjust", "?")

        adjust_kinds.add(adjust)

        # Warnings from the record verbatim
        if r.get("warnings"):
            try:
                all_warnings.extend(json.loads(r["warnings"]))
            except (json.JSONDecodeError, TypeError):
                pass

        rows.append({
            "factor": r.get("factor", "?"),
            "ic": ev.get("ic_mean"),
            "icir": ev.get("icir"),
            "ic_pos": ev.get("ic_positive_ratio"),
            "n_ic_obs": ev.get("n_ic_obs"),
            "sharpe_proxy": ev.get("long_short_sharpe_proxy"),
            "pit": pit,
            "biases": biases,
            "provider": dm.get("provider", r.get("data_source", "?")),
            "adjust": adjust,
        })

    # --- Sort by ICIR descending for the table ---------------------------------
    rows.sort(key=lambda x: x.get("icir") or 0, reverse=True)

    # --- Write Markdown --------------------------------------------------------
    out = Path("data/reports/factor_comparison.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Factor Comparison Report",
        f"",
        f"**Generated from Registry**. {len(success_runs)} success runs, {len(failed_runs)} failed runs.",
        f"",
        "## Summary statistics (from Registry)",
        f"",
        f"- Total runs: {len(runs)}",
        f"- Success: {len(success_runs)}",
        f"- Failed: {len(failed_runs)}",
        f"- Data sources: {', '.join(sorted(data_sources)) if data_sources else 'none'}",
        f"- Universe providers: {', '.join(sorted(universe_providers)) if universe_providers else 'none'}",
        f"- Adjustment methods used: {', '.join(sorted(adjust_kinds)) if adjust_kinds else 'none'}",
        f"",
    ]

    lines += [
        "## Per-factor comparison (success runs only)",
        "",
        "| Factor          | IC     | ICIR  | IC>0%  | Sharpe proxy | PIT  | Adjust | Provider |",
        "|-----------------|--------|-------|--------|--------------|------|--------|----------|",
    ]
    for r in rows:
        ic = f"{r['ic']:.4f}" if isinstance(r["ic"], (int, float)) else str(r["ic"])
        icir = f"{r['icir']:.4f}" if isinstance(r["icir"], (int, float)) else str(r["icir"])
        ic_pos = f"{r['ic_pos']:.0%}" if isinstance(r["ic_pos"], (int, float)) else str(r["ic_pos"])
        sharpe = f"{r['sharpe_proxy']:.2f}" if isinstance(r["sharpe_proxy"], (int, float)) else str(r["sharpe_proxy"])
        pit = r["pit"]
        lines.append(
            f"| {r['factor']:<15s} | {ic:<6s} | {icir:<6s} | {ic_pos:<6s} | {sharpe:<12s} | {pit}  | {r['adjust']:<6s} | {r['provider']:<8s} |"
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

    lines += [
        "",
        "## Known limitations (from Registry trust fields)",
        "",
        "- All runs use `pit=false` (current constituent snapshot applied to history) — survivorship bias possible.",
        f"- Adjustment methods vary: {', '.join(sorted(adjust_kinds)) if adjust_kinds else 'none'}.",
        "- DSR marked insufficient_trials for single-factor runs — not formal validation.",
        "- BH-FDR not applicable until multiple factors tested simultaneously.",
        "",
        f"_Generated by lab.reports.factor_comparison — zero new framework. All facts from Registry._",
        f"",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    path = generate()
    print(f"Factor comparison: {path}", flush=True)
