"""Daily snapshot for the user's personal holdings (.scratch/holdings).

Reads holdings.json (positions with broker codes; funds by stored value until
fund codes are confirmed), fetches Sina spot tables, and writes:
- .scratch/holdings/latest.json  (values, weights, alerts)
- .scratch/holdings/history.csv  (one row per day, appended)

Alerts: greenland-style face-value delisting proximity, single-day moves
>=5%, concentrated tech bucket weight > 85%, single position > 20%.

Read-only with respect to the broker; nothing is traded.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HOLD_DIR = ROOT / ".scratch" / "holdings"
HOLDINGS = HOLD_DIR / "holdings.json"
LATEST = HOLD_DIR / "latest.json"
HISTORY = HOLD_DIR / "history.csv"

BUCKETS = {
    "sz159516": "芯片/半导体",
    "sz159801": "芯片/半导体",
    "sh512480": "芯片/半导体",
    "sh516920": "芯片/半导体",
    "sz159819": "AI/机器人",
    "sh562500": "AI/机器人",
    "sz159994": "5G/通信",
    "sh562970": "光伏",
    "sh513110": "纳指100",
    "sh600606": "地产个股",
}
TECH_BUCKETS = {"芯片/半导体", "AI/机器人", "5G/通信", "光伏", "纳指100"}


def fetch_prices() -> dict[str, dict]:
    import akshare as ak

    prices: dict[str, dict] = {}
    etf = ak.fund_etf_category_sina(symbol="ETF基金")
    etf["code"] = etf["代码"].astype(str)
    for _, row in etf.iterrows():
        prices[row["code"]] = {
            "price": pd.to_numeric(row.get("最新价"), errors="coerce"),
            "pct": pd.to_numeric(row.get("涨跌幅"), errors="coerce"),
            "prev": pd.to_numeric(row.get("昨收"), errors="coerce"),
        }
    stock = ak.stock_zh_a_spot()
    stock["code"] = stock["代码"].astype(str)
    for _, row in stock.iterrows():
        prices[row["code"]] = {
            "price": pd.to_numeric(row.get("最新价"), errors="coerce"),
            "pct": pd.to_numeric(row.get("涨跌幅"), errors="coerce"),
            "prev": pd.to_numeric(row.get("昨收"), errors="coerce"),
        }
    return prices


def build_snapshot(data: dict, prices: dict[str, dict],
                   today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    rows = []
    total_value = 0.0
    total_cost = 0.0
    quoted_pnl = 0.0
    missing = []
    missing_cost = 0.0
    for position in data["positions"]:
        quote = prices.get(position["code"])
        price = quote.get("price") if quote else None
        is_quoted = price is not None and pd.notna(price) and price > 0
        cost_value = float(position["cost"]) * position["shares"]
        total_cost += cost_value
        if is_quoted:
            price = float(price)
            value = price * position["shares"]
            pnl = value - cost_value
            total_value += value
            quoted_pnl += pnl
            weight_placeholder = value
        else:
            # A missing quote is not evidence that price == cost. Exclude it
            # from totals rather than manufacturing a zero-PnL position.
            missing.append({"name": position["name"], "code": position["code"]})
            missing_cost += cost_value
            price = None
            value = None
            pnl = None
            weight_placeholder = 0.0
        rows.append({
            "name": position["name"], "code": position["code"],
            "shares": position["shares"], "cost": position["cost"],
            "price": price,
            "value": None if value is None else round(value, 2),
            "pnl": None if pnl is None else round(pnl, 2),
            "pct": None if pnl is None or not cost_value else round(pnl / cost_value * 100, 2),
            "day_pct": (float(quote["pct"]) if is_quoted and pd.notna(quote.get("pct")) else None),
            "bucket": BUCKETS.get(position["code"], "其他"),
            "confirmed": position.get("confirmed", True),
            "quote_status": "quoted" if is_quoted else "missing",
            "weight_placeholder": weight_placeholder,
        })
    fund_value = sum(f["value"] for f in data.get("funds", []))
    grand_total = total_value + float(data.get("cash_cny", 0.0)) + fund_value
    for row in rows:
        row["weight"] = (round(row["weight_placeholder"] / total_value, 4)
                         if row["quote_status"] == "quoted" and total_value else None)
        row.pop("weight_placeholder")

    alerts = []
    for row in rows:
        if row["quote_status"] != "quoted":
            continue
        if row["code"] == "sh600606" and row["price"] < 1.15:
            alerts.append(
                f"绿地控股 {row['price']:.3f} 元，距 1 元面值退市线不足 "
                f"{(row['price']-1.0)*100:.0f} 分"
            )
        if row["day_pct"] is not None and abs(row["day_pct"]) >= 5:
            alerts.append(f"{row['name']} 单日 {row['day_pct']:+.2f}%")
        if row["weight"] is not None and row["weight"] > 0.2:
            alerts.append(f"{row['name']} 占有报价证券市值 {row['weight']*100:.1f}%")
    if missing:
        alerts.append(
            "缺行情、未计入总市值（不是按成本零盈亏）："
            + "、".join(f"{row['name']}({row['code']})" for row in missing)
        )
    funds_as_of = data.get("as_of")
    if data.get("funds") and funds_as_of:
        try:
            age = (today - dt.date.fromisoformat(funds_as_of)).days
            if age > 0:
                alerts.append(f"基金金额仍是 {funds_as_of} 的存档值，未接入净值更新")
        except ValueError:
            alerts.append("基金存档日期格式异常，无法确认净值新鲜度")
    bucket_value: dict[str, float] = {}
    for row in rows:
        if row["quote_status"] == "quoted":
            bucket_value[row["bucket"]] = bucket_value.get(row["bucket"], 0.0) + row["value"]
    tech_value = sum(v for b, v in bucket_value.items() if b in TECH_BUCKETS)
    tech_share = tech_value / total_value if total_value else 0.0
    if tech_share > 0.85:
        alerts.append(f"科技类（芯片+AI+通信+光伏+纳指）占有报价证券市值 {tech_share*100:.1f}%")

    return {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "securities_value": round(total_value, 2),
        "securities_value_complete": not missing,
        "securities_cost": round(total_cost, 2),
        "securities_pnl": round(quoted_pnl, 2),
        "securities_pnl_complete": not missing,
        "cash": data.get("cash_cny", 0.0),
        "funds_value": round(fund_value, 2),
        "funds_as_of": funds_as_of,
        "grand_total": round(grand_total, 2),
        "grand_total_complete": not missing,
        "missing_cost_cny": round(missing_cost, 2),
        "tech_share": round(tech_share, 4),
        "buckets": {b: round(v, 2) for b, v in sorted(bucket_value.items())},
        "alerts": alerts,
        "missing_quotes": missing,
        "positions": rows,
    }


def main() -> int:
    data = json.loads(HOLDINGS.read_text(encoding="utf-8"))
    payload = build_snapshot(data, fetch_prices())
    LATEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    today = dt.date.today().isoformat()
    record = {
        "date": today,
        "securities_value": payload["securities_value"],
        "securities_pnl": payload["securities_pnl"],
        "grand_total": payload["grand_total"],
        "tech_share": payload["tech_share"],
        "n_alerts": len(payload["alerts"]),
    }
    frame = pd.DataFrame([record])
    if HISTORY.is_file():
        history = pd.read_csv(HISTORY)
        history = history[history["date"] != today]
        frame = pd.concat([history, frame], ignore_index=True)
    frame.to_csv(HISTORY, index=False)

    print(json.dumps({
        "securities_value": payload["securities_value"],
        "securities_value_complete": payload["securities_value_complete"],
        "securities_pnl": payload["securities_pnl"],
        "grand_total": payload["grand_total"],
        "grand_total_complete": payload["grand_total_complete"],
        "tech_share": payload["tech_share"],
        "alerts": payload["alerts"],
        "missing": payload["missing_quotes"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
