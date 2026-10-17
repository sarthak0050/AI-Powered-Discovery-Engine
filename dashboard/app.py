"""Streamlit dashboard for the discovery engine.

    python run.py dashboard          (or: streamlit run dashboard/app.py)

Reads only from data/exports/ — no database, no API keys, no LLM calls — so the same
file can be hosted on Streamlit Community Cloud unchanged.

Tabs:
  Overview                    funnel of counts + journey-stage breakdown
  Pre-purchase opportunities  A1 scorecard (CI bars, per-factor breakdown) + external sources
  Trust & retention           Track B scorecard + bridge-by-stage (circularity note)
  Evidence explorer           filter posts by opportunity / code / stage / source
  Engine quality              the quality report with pass / below badges
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

EXPORTS = Path(__file__).resolve().parent.parent / "data" / "exports"

DECISION_STAGES = ["browsing", "saved_or_wishlisted", "in_cart", "comparing",
                   "postponed", "abandoned"]

FACTOR_LABELS = {
    "frequency_contrib": "frequency",
    "severity_contrib": "severity",
    "proximity_to_purchase_contrib": "proximity",
    "non_monetary_addressability_contrib": "non-monetary",
}

st.set_page_config(page_title="Myntra Discovery Engine", layout="wide")


# --------------------------------------------------------------------------- loading

@st.cache_data(show_spinner=False)
def load(name: str) -> pd.DataFrame:
    """One CSV from data/exports/, or an empty frame if it is not there yet."""
    path = EXPORTS / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _num(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _parse_json_list(value) -> list[str]:
    if not value:
        return []
    value = str(value).strip()
    if value.startswith("["):
        try:
            return [str(x) for x in json.loads(value)]
        except Exception:
            pass
    return [x.strip().strip("\"'") for x in value.strip("[]").split(",") if x.strip()]


def _parse_pipe(value) -> list[str]:
    return [x.strip() for x in str(value or "").split("|") if x.strip()]


def _missing(name: str) -> None:
    st.warning(f"data/exports/{name} is missing — run the earlier stages "
               f"(`python run.py analyze`, `python run.py score`, "
               f"`python run.py quality`).")


def _fmt_pct(value) -> str:
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "—"


# --------------------------------------------------------------------------- tab 1

def tab_overview() -> None:
    clean = load("clean_posts.csv")
    tagged = load("tagged_posts.csv")
    bridge = load("bridge_check.csv")
    if tagged.empty:
        _missing("tagged_posts.csv")
        return

    in_scope = tagged["in_scope"] == "1"
    prepurchase = in_scope & tagged["journey_stage"].isin(DECISION_STAGES)
    n_raw = len(clean)
    n_relevant = len(tagged)
    n_scope = int(in_scope.sum())
    n_pre = int(prepurchase.sum())
    n_a1 = max(0, n_pre - len(bridge))

    st.subheader("Corpus at a glance")
    cards = [
        ("Raw collected", n_raw, "kept after near-duplicate / spam removal"),
        ("Relevant", n_relevant, "about a fashion buying decision"),
        ("In scope", n_scope, "pass the locked rubric"),
        ("Pre-purchase", n_pre, "Track A journey stages"),
        ("A1 (item-level)", n_a1, "pre-purchase without the bridge set"),
    ]
    cols = st.columns(len(cards))
    prev = None
    for col, (label, value, note) in zip(cols, cards):
        with col:
            delta = None if not prev else f"{value / prev * 100:.0f}% of prior"
            st.metric(label, f"{value:,}", delta=delta, delta_color="off", help=note)
        prev = value

    st.divider()

    st.subheader("Funnel — raw posts to item-level pre-purchase friction")
    stages = ["Raw (deduplicated)", "Relevant", "In scope", "Pre-purchase", "A1 (item-level)"]
    counts = [n_raw, n_relevant, n_scope, n_pre, n_a1]
    fig = go.Figure(go.Funnel(
        y=stages, x=counts, textposition="inside", textinfo="value+percent initial",
        marker={"color": ["#cdd9ea", "#a3bcdc", "#7295c9", "#4270b4", "#1f4a8f"]},
        connector={"line": {"color": "#c3ccd8"}},
        hovertemplate="%{y}<br>%{x:,} posts · %{percentInitial:.1%} of raw<extra></extra>"))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360,
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Raw = posts after near-duplicate/spam removal (the pool that entered the "
               "relevance filter). A1 = pre-purchase posts that do not cite a past "
               "bad experience (the bridge set is subtracted).")

    st.divider()

    st.subheader("Journey-stage breakdown (pre-purchase, Track A)")
    sb = _num(load("stage_breakdown.csv"),
              ["posts", "share_of_track_a_pct", "top1_posts", "top2_posts", "top3_posts"])
    if sb.empty:
        _missing("stage_breakdown.csv")
        return
    order = {s: i for i, s in enumerate(DECISION_STAGES)}
    sb["_order"] = sb["journey_stage"].map(order).fillna(len(order))
    sb = sb.sort_values("_order")

    left, right = st.columns([1, 1], gap="large")
    with left:
        fig = px.bar(sb, x="posts", y="journey_stage", orientation="h",
                     color="posts", color_continuous_scale="Blues",
                     custom_data=["share_of_track_a_pct"],
                     labels={"posts": "posts", "journey_stage": ""})
        fig.update_traces(
            texttemplate="%{x:,}", textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>%{x:,} posts · %{customdata[0]:.1f}% of Track A<extra></extra>")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360,
                          coloraxis_showscale=False, showlegend=False,
                          xaxis_range=[0, float(sb["posts"].max()) * 1.22])
        st.plotly_chart(fig, width="stretch")
        st.caption("Journey order, earliest to latest; % of Track A on hover.")
    with right:
        show = sb[["journey_stage", "posts", "share_of_track_a_pct",
                   "top1_opportunity", "top1_pct_of_stage"]].copy()
        show["posts"] = show["posts"].map(lambda v: f"{int(v):,}" if pd.notna(v) else "—")
        show["share_of_track_a_pct"] = show["share_of_track_a_pct"].map(_fmt_pct)
        show["top1_pct_of_stage"] = show["top1_pct_of_stage"].map(_fmt_pct)
        show.columns = ["stage", "posts", "share of Track A", "top opportunity", "% of stage"]
        st.dataframe(show, hide_index=True, width="stretch", height=360)


# --------------------------------------------------------------------------- tab 2

def _scorecard_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["rank", "opportunity", "posts", "share_track_pct",
              "freq_ci95_low_pct", "freq_ci95_high_pct", "n_sources",
              "member_codes", "total_score"]].copy()
    out["95% CI"] = out.apply(
        lambda r: f"{r['freq_ci95_low_pct']:.1f}–{r['freq_ci95_high_pct']:.1f}", axis=1)
    out = out[["rank", "opportunity", "posts", "share_track_pct", "95% CI",
               "n_sources", "member_codes", "total_score"]]
    out.columns = ["#", "opportunity", "posts", "share %", "95% CI", "sources",
                   "member codes", "score"]
    return out


def _factor_chart(df: pd.DataFrame, title: str) -> None:
    contribs = [c for c in FACTOR_LABELS if c in df.columns]
    long = df.melt(id_vars="opportunity", value_vars=contribs,
                   var_name="factor", value_name="contribution")
    long["factor"] = long["factor"].map(FACTOR_LABELS)
    order = df.sort_values("total_score")["opportunity"].tolist()
    fig = px.bar(long, x="contribution", y="opportunity", color="factor",
                 orientation="h", category_orders={"opportunity": order},
                 title=title, labels={"opportunity": "", "contribution": "weighted score"})
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380,
                      legend_title_text="factor")
    st.plotly_chart(fig, width="stretch")


def tab_prepurchase() -> None:
    a1 = _num(load("opportunity_scorecard_prepurchase_a1.csv"),
              ["posts", "share_track_pct", "freq_ci95_low_pct", "freq_ci95_high_pct",
               "n_sources", "total_score", *FACTOR_LABELS])
    if a1.empty:
        _missing("opportunity_scorecard_prepurchase_a1.csv")
        return
    a1 = a1.sort_values("total_score", ascending=False)

    st.subheader("A1 — item-level pre-purchase friction")
    st.caption("Pre-purchase posts that do not cite a past bad experience, re-scored on "
               "the existing Track A opportunity definitions (no new synthesis).")
    st.dataframe(_scorecard_table(a1), hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=a1["share_track_pct"], y=a1["opportunity"], orientation="h",
            error_x=dict(type="data",
                         array=a1["freq_ci95_high_pct"] - a1["share_track_pct"],
                         arrayminus=a1["share_track_pct"] - a1["freq_ci95_low_pct"]),
            marker_color="#3f6bb0"))
        fig.update_layout(title="Share of A1 posts (95% Wilson CI)",
                          xaxis_title="% of A1", yaxis_title="",
                          margin=dict(l=10, r=10, t=40, b=10), height=380)
        st.plotly_chart(fig, width="stretch")
    with right:
        _factor_chart(a1, "Score by factor (weighted contribution)")

    st.subheader("External sources shoppers check")
    ext = _num(load("external_sources_decision.csv"), ["count", "pct_of_relevant_posts"])
    if ext.empty:
        _missing("external_sources_decision.csv")
        return
    ext["key"] = ext["source"].str.strip().str.lower()
    agg = (ext.groupby("key", as_index=False)["count"].sum()
           .sort_values("count", ascending=False).head(15))
    fig = px.bar(agg.sort_values("count"), x="count", y="key", orientation="h",
                 labels={"count": "mentions", "key": ""},
                 title="Where pre-purchase posts say they cross-check (top 15)")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420, showlegend=False)
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- tab 3

def tab_trust() -> None:
    tb = _num(load("opportunity_scorecard_trust.csv"),
              ["posts", "share_track_pct", "freq_ci95_low_pct", "freq_ci95_high_pct",
               "n_sources", "total_score", *FACTOR_LABELS])
    if tb.empty:
        _missing("opportunity_scorecard_trust.csv")
        return
    tb = tb.sort_values("total_score", ascending=False)

    st.subheader("Track B — Trust & retention (purchased / returned)")
    st.caption("Past the buying decision, so proximity_to_purchase is dropped and the "
               "remaining factor weights are renormalised.")
    st.dataframe(_scorecard_table(tb), hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=tb["share_track_pct"], y=tb["opportunity"], orientation="h",
            error_x=dict(type="data",
                         array=tb["freq_ci95_high_pct"] - tb["share_track_pct"],
                         arrayminus=tb["share_track_pct"] - tb["freq_ci95_low_pct"]),
            marker_color="#1f4a8f"))
        fig.update_layout(title="Share of Track B posts (95% Wilson CI)",
                          xaxis_title="% of Track B", yaxis_title="",
                          margin=dict(l=10, r=10, t=40, b=10), height=420)
        st.plotly_chart(fig, width="stretch")
    with right:
        _factor_chart(tb, "Score by factor (weighted contribution)")

    st.subheader("Bridge by stage — do pre-purchase posts cite a past bad experience?")
    b = _num(load("bridge_by_stage.csv"),
             ["posts", "code_pct", "keyword_pct", "union_pct",
              "code_ci_low", "code_ci_high", "keyword_ci_low", "keyword_ci_high",
              "union_ci_low", "union_ci_high"])
    if b.empty:
        _missing("bridge_by_stage.csv")
        return
    long = b.melt(id_vars=["group", "posts"],
                  value_vars=["code_pct", "keyword_pct", "union_pct"],
                  var_name="signal", value_name="pct")
    long["signal"] = long["signal"].map(
        {"code_pct": "bad-experience code", "keyword_pct": "category keyword",
         "union_pct": "union"})
    fig = px.bar(long, x="group", y="pct", color="signal", barmode="group",
                 text="pct", labels={"group": "", "pct": "% of group"})
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380,
                      legend_title_text="signal")
    st.plotly_chart(fig, width="stretch")
    st.info(
        "Circularity note: `abandoned` posts are, by construction, ones that did not "
        "convert, so the bridge concentrates there. The abandoned slice cannot be read "
        "as independent evidence that past bad experiences cause abandonment — that is "
        "why Track A is split into A1 (item-level) and A2 (trust-driven).")


# --------------------------------------------------------------------------- tab 4

def tab_evidence() -> None:
    tagged = load("tagged_posts.csv")
    posts = load("relevant_posts.csv")
    if tagged.empty or posts.empty:
        _missing("tagged_posts.csv / relevant_posts.csv")
        return
    tagged = tagged[tagged["in_scope"] == "1"].copy()
    tagged["codes_list"] = tagged["codes"].map(_parse_json_list)

    merge_cols = [c for c in ("post_id", "text_original", "text_en", "url", "source",
                              "date") if c in posts.columns]
    df = tagged.merge(posts[merge_cols], on="post_id", how="left")
    df["text"] = df.get("text_original", df.get("text_en", "")).fillna("")

    # opportunity -> member codes, from both scorecards
    opp_map: dict[str, set[str]] = {}
    for name in ("opportunity_scorecard_prepurchase_a1.csv",
                 "opportunity_scorecard_trust.csv"):
        sc = load(name)
        for _, r in sc.iterrows():
            opp_map[r["opportunity"]] = set(_parse_pipe(r.get("member_codes", "")))

    st.subheader("Evidence explorer")
    c1, c2 = st.columns(2)
    with c1:
        opp = st.multiselect("Opportunity", sorted(opp_map))
        code = st.multiselect("Code", sorted({c for cs in df["codes_list"] for c in cs}))
    with c2:
        stage = st.multiselect("Journey stage", sorted(df["journey_stage"].unique()))
        source = st.multiselect("Source", sorted(df["source"].dropna().unique()))

    sel = df
    if opp:
        wanted = set().union(*(opp_map[o] for o in opp))
        sel = sel[sel["codes_list"].map(lambda cs: bool(wanted & set(cs)))]
    if code:
        sel = sel[sel["codes_list"].map(lambda cs: bool(set(code) & set(cs)))]
    if stage:
        sel = sel[sel["journey_stage"].isin(stage)]
    if source:
        sel = sel[sel["source"].isin(source)]

    st.caption(f"{len(sel):,} post(s) match. Showing up to 200.")
    show = sel.head(200)[["post_id", "source", "journey_stage", "primary_code",
                          "evidence_quote", "url"]].copy()
    show.columns = ["post_id", "source", "stage", "primary code", "quote", "link"]
    st.dataframe(
        show, hide_index=True, width="stretch", height=520,
        column_config={"link": st.column_config.LinkColumn("link", display_text="open")})


# --------------------------------------------------------------------------- tab 5

BADGE = {
    "pass": ("#e6f4ea", "#1e7e34", "PASS"),
    "below": ("#fdecea", "#b3261e", "BELOW"),
    "info": ("#eef1f5", "#44546a", "INFO"),
}


def tab_quality() -> None:
    q = load("engine_quality.csv")
    if q.empty:
        _missing("engine_quality.csv")
        return
    st.subheader("Engine quality report")
    pass_n = int((q["status"] == "pass").sum())
    below_n = int((q["status"] == "below").sum())
    info_n = int((q["status"] == "info").sum())
    st.caption(f"{len(q)} metrics · {pass_n} pass · {below_n} below · {info_n} info "
               f"(from data/exports/engine_quality.csv)")

    rows = []
    for _, r in q.iterrows():
        bg, fg, label = BADGE.get(r["status"], BADGE["info"])
        badge = (f'<span style="background:{bg};color:{fg};padding:2px 8px;'
                 f'border-radius:10px;font-weight:600;font-size:12px">{label}</span>')
        rows.append(
            f"<tr><td style='padding:6px 10px'>{r['metric']}</td>"
            f"<td style='padding:6px 10px;text-align:right;font-weight:600'>{r['value']}</td>"
            f"<td style='padding:6px 10px'>{r['target']}</td>"
            f"<td style='padding:6px 10px'>{badge}</td>"
            f"<td style='padding:6px 10px;color:#5a6472;font-size:12px'>{r['method']}</td>"
            f"<td style='padding:6px 10px;text-align:right'>{r['n']}</td></tr>")
    html = (
        "<table style='width:100%;border-collapse:collapse'>"
        "<thead><tr style='text-align:left;border-bottom:1px solid #ddd'>"
        "<th style='padding:6px 10px'>metric</th><th style='padding:6px 10px'>value</th>"
        "<th style='padding:6px 10px'>target</th><th style='padding:6px 10px'>status</th>"
        "<th style='padding:6px 10px'>method</th><th style='padding:6px 10px'>n</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")
    st.markdown(html, unsafe_allow_html=True)


# --------------------------------------------------------------------------- main

def main() -> None:
    st.title("Myntra Discovery Engine")
    st.caption("How people decide what to buy — pre-purchase friction and trust & "
               "retention, built from public conversations.")
    tabs = st.tabs(["Overview", "Pre-purchase opportunities", "Trust & retention",
                    "Evidence explorer", "Engine quality"])
    with tabs[0]:
        tab_overview()
    with tabs[1]:
        tab_prepurchase()
    with tabs[2]:
        tab_trust()
    with tabs[3]:
        tab_evidence()
    with tabs[4]:
        tab_quality()


main()
