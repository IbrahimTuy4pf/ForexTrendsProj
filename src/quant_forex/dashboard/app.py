"""Streamlit research dashboard. Read-only. No order ticket."""

from __future__ import annotations

import streamlit as st

from quant_forex.dashboard.catalog import (
    backtest_metrics_table,
    catalog,
    load_equity,
    load_json,
    load_ohlcv_preview,
    sibling_markdown,
)
from quant_forex.execution.broker import LiveTradingNotImplemented, submit_live_order
from quant_forex.utils.paths import ProjectPaths


def main() -> None:
    st.set_page_config(page_title="Quant Forex research", layout="wide")
    paths = ProjectPaths.find()
    snap = catalog(paths)
    st.title("Quant Forex research dashboard")
    st.warning(snap["disclaimer"])
    st.caption("Live trading: disabled. Broker orders: not implemented.")

    overview, market, backtest, models, diagnostics, paper, reports = st.tabs(
        ["Overview", "Market", "Backtest", "Models", "Diagnostics", "Paper", "Reports"]
    )

    with overview:
        st.subheader("What this is")
        st.markdown(
            "A viewer for **local** pipeline outputs (processed bars, costed backtests, "
            "ML/walk-forward JSON, paper ledger snapshots). It does not stream a broker feed."
        )
        st.markdown(
            f"- Processed CSV: `{snap['processed_csv']}`\n"
            f"- Equity CSV: `{snap['equity_csv']}`\n"
            f"- Live trading allowed: **{snap['live_trading_allowed']}**"
        )
        if st.button("Try to send a live order (should fail)"):
            try:
                submit_live_order()
                st.error("Unexpected: live order did not raise.")
            except LiveTradingNotImplemented as exc:
                st.success(str(exc))

    with market:
        frame = load_ohlcv_preview(snap["processed_csv"])
        if frame.empty:
            st.info("No processed OHLCV. Run `python -m quant_forex ingest` first.")
        else:
            st.line_chart(frame.set_index("timestamp")["close"], height=360)
            st.dataframe(frame.tail(20), width="stretch")

    with backtest:
        metrics = backtest_metrics_table(load_json(snap["reports"]["backtest"]))
        if metrics.empty:
            st.info("No backtest JSON yet. Run `python -m quant_forex backtest`.")
        else:
            st.dataframe(metrics, width="stretch")
            st.caption("These numbers include costs in the research engine. They are not live PnL.")
        equity = load_equity(snap["equity_csv"])
        if not equity.empty and "strategy" in equity.columns:
            choice = st.selectbox("Equity curve", sorted(equity["strategy"].unique()))
            subset = equity.loc[equity["strategy"] == choice]
            st.line_chart(subset.set_index("timestamp")["equity"], height=320)
            st.line_chart(subset.set_index("timestamp")["drawdown"], height=220)

    with models:
        ml = load_json(snap["reports"]["ml"])
        wf = load_json(snap["reports"]["walkforward"])
        if ml is None and wf is None:
            st.info("Run `python -m quant_forex train` and/or `walk-forward`.")
        if ml:
            st.markdown(f"Selected model: `{ml.get('selected_model')}`")
            st.json({"split": ml.get("split_method"), "metrics": ml.get("metrics")})
        if wf:
            st.markdown(f"Walk-forward folds: `{wf.get('n_folds')}` mode `{wf.get('mode')}`")
            st.dataframe(wf.get("folds") or [], width="stretch")

    with diagnostics:
        diag = load_json(snap["reports"]["diagnostics"])
        if diag is None:
            st.info("Run `python -m quant_forex diagnose`.")
        else:
            for flag in diag.get("flags") or []:
                st.warning(flag)
            if not diag.get("flags"):
                st.write("No flags recorded (that is not a green light to trade).")
            st.json({"caveat": diag.get("caveat"), "n_strategies": len(diag.get("strategies") or [])})

    with paper:
        paper_payload = load_json(snap["reports"]["paper"])
        if paper_payload is None:
            st.info("Run `python -m quant_forex paper --enable-paper`. Still not a broker.")
        else:
            st.markdown(paper_payload.get("note", ""))
            cols = st.columns(3)
            cols[0].metric("Simulated equity", f"{paper_payload.get('equity', 0):.2f}")
            cols[1].metric("Open units", f"{paper_payload.get('open_units', 0):.4f}")
            cols[2].metric("Closed trades", paper_payload.get("n_closed_trades", 0))
            st.json(paper_payload.get("pending") or {"pending": None})

    with reports:
        kind = st.selectbox("Report", [k for k, p in snap["reports"].items() if p is not None] or ["(none)"])
        path = snap["reports"].get(kind)
        text = sibling_markdown(path)
        if text:
            st.markdown(text)
        elif path:
            st.json(load_json(path))
        else:
            st.info("No JSON reports in `reports/` yet.")


if __name__ == "__main__":
    main()
