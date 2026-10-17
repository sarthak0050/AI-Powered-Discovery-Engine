"""Streamlit dashboard for the discovery engine.

    python run.py dashboard          (or: streamlit run dashboard/app.py)

Reads only from data/exports/ — no database, no API keys, no LLM calls — so the same
file can be hosted on Streamlit Community Cloud unchanged.

Six tabs, written so a non-technical reader understands the story in 30 seconds:

  Summary                     why Myntra shoppers hesitate, in one screen
  What stops shoppers         pre-purchase blockers (item-level)
  Where shoppers look elsewhere   the other sites people check
  After-purchase trust        returns / repeat-buy issues + the past-experience link
  Real voices                 the actual quotes behind every number
  How it works & how reliable the pipeline, the funnel, and the quality checks

Internal names (codebook codes, "in_scope", factor weights, kappa, CI) never appear
outside the "Technical details" expanders.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

EXPORTS = Path(__file__).resolve().parent.parent / "data" / "exports"

ACCENT = "#E91E63"       # Myntra-like pink, used sparingly for the highlight
NEUTRAL = "#B7BEC9"      # grey bars / context
INK = "#1F2430"
MUTED = "#6B7280"

DECISION_STAGES = ["browsing", "saved_or_wishlisted", "in_cart", "comparing",
                   "postponed", "abandoned"]

STAGE_LABELS = {
    "browsing": "Browsing",
    "saved_or_wishlisted": "Saved to wishlist",
    "in_cart": "In cart",
    "comparing": "Comparing",
    "postponed": "Postponed",
    "abandoned": "Gave up",
    "purchased": "Purchased",
    "returned": "Returned",
    "other": "Other",
}

SOURCE_LABELS = {
    "playstore": "Google Play",
    "appstore": "App Store",
    "reddit": "Reddit",
    "youtube": "YouTube",
    "manual": "Manual",
}

# Internal opportunity name -> plain-English label. Same label everywhere it appears.
BLOCKER_LABELS = {
    "Styling Advice and Store Recommendations": "Styling advice",
    "Direct Product Link and Discovery Sharing": "Finding items seen elsewhere",
    "Cross-Platform Price and Deal Comparison": "Comparing prices elsewhere",
    "App Navigation and Checkout Methods": "Finding & checkout in the app",
    "Garment Fit and Size Availability": "Fit & size",
    "Returns, Policy Rules, and Authenticity": "Returns & authenticity worries",
    # after-purchase
    "Delivery, Fulfillment, and Refund Logistics": "Delivery & refunds",
    "Customer Support and Platform Retention": "Customer support",
    "Product Quality and Representation Accuracy": "Product not as shown",
    "Pricing, Discounts, and Promotional Value": "Pricing & offers",
    "Returns and Exchange Policy Execution": "Returns & exchanges",
    "Cross-Platform Shopping and App Usability": "App usability",
    "Seller Authenticity and Brand Trust": "Trusting the seller",
    "Garment Sizing and Fit Evaluation": "Sizing & fit",
    "Fashion Discovery and Style Guidance": "Style discovery",
}

# Curated highlight quotes for the Summary (real, verbatim, from the exports).
FINDINGS = [
    {
        "headline": "Shoppers seek styling validation before they buy",
        "number": "32% of decision-moment posts",
        "quote": "it help me to select perfectly everything it also help me and my "
                 "bestie to select for a party wear",
        "source": "Google Play",
    },
    {
        "headline": "They can’t find the products they saw elsewhere",
        "number": "30% of decision-moment posts",
        "quote": "Where is the dress from?",
        "source": "Reddit",
    },
    {
        "headline": "They check prices on other platforms before buying",
        "number": "19% of decision-moment posts",
        "quote": "prices are higher than other apps",
        "source": "Google Play",
    },
]

st.set_page_config(page_title="Why Myntra shoppers hesitate",
                   layout="wide", initial_sidebar_state="collapsed")


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


def _label(value: str) -> str:
    return BLOCKER_LABELS.get(value, value)


def _source(value: str) -> str:
    return SOURCE_LABELS.get(str(value).lower(), str(value).title())


def _people(n: float) -> str:
    return f"{int(round(n)):,}"


# --------------------------------------------------------------------------- chrome

def _howto(text: str) -> None:
    st.markdown(
        "<div style='background:#FCE4EC;border-left:4px solid " + ACCENT + ";"
        "border-radius:8px;padding:12px 16px;margin:2px 0 18px;font-size:14px;"
        "line-height:1.55;color:" + INK + "'>"
        "<b>How to read this</b><br>" + text + "</div>",
        unsafe_allow_html=True)


def _badge(text: str, bg: str, fg: str) -> str:
    return (f"<span style='background:{bg};color:{fg};padding:2px 9px;border-radius:11px;"
            f"font-weight:600;font-size:11.5px;margin-right:6px'>{text}</span>")


def _metric_row(cards) -> None:
    cols = st.columns(len(cards))
    for col, (label, value, note) in zip(cols, cards):
        with col:
            st.metric(label, value, help=note or None)
            if note:
                st.caption(note)


def _clean_bar(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=44, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, title=None),
        yaxis=dict(showgrid=False, zeroline=False, title=None),
        font=dict(color=INK, size=13), title=dict(x=0, xanchor="left", font=dict(size=15)))
    return fig


# --------------------------------------------------------------------------- summary

def tab_summary() -> None:
    clean = load("clean_posts.csv")
    tagged = load("tagged_posts.csv")
    bridge = load("bridge_check.csv")
    a1 = _num(load("opportunity_scorecard_prepurchase_a1.csv"), ["share_track_pct", "posts"])
    sb = _num(load("stage_breakdown.csv"), ["posts", "share_of_track_a_pct"])

    n_raw = len(clean)
    n_pre = int((sb["posts"].sum())) if not sb.empty else 0
    top = a1.sort_values("share_track_pct", ascending=False).iloc[0] if not a1.empty else None
    wishlist = (sb.loc[sb["journey_stage"] == "saved_or_wishlisted", "share_of_track_a_pct"]
                if not sb.empty else pd.Series(dtype=float))
    wish_pct = float(wishlist.iloc[0]) if len(wishlist) else 1.1

    st.title("Why Myntra shoppers hesitate before buying")
    st.markdown(
        "<div style='font-size:17px;color:" + MUTED + ";margin-top:-6px'>"
        f"Insights from {_people(n_raw)} public reviews and conversations, "
        "analysed with AI.</div>", unsafe_allow_html=True)
    st.write("")
    st.markdown(
        "This dashboard turns thousands of public Myntra reviews and conversations into "
        "a clear picture of why shoppers hesitate before buying — and what puts them off "
        "coming back. Every number links back to a real post you can read for yourself.")

    _howto(
        "This is the whole story on one screen: how many posts we analysed, the single "
        "biggest blocker, and three findings worth remembering. Start with the four "
        "numbers, then read the three findings for the detail. Key takeaway: most "
        "hesitation is about styling advice, finding items seen elsewhere, and price.")

    st.write("")
    cards = [
        ("Posts analysed", _people(n_raw), "public reviews and conversations"),
        ("Decision-moment posts", _people(n_pre), "posts where someone is choosing or giving up"),
        ("Top blocker", _label(top["opportunity"]) if top is not None else "—",
         (f"{top['share_track_pct']:.0f}% of decision-moment posts" if top is not None else "")),
        ("Wishlist mentions", f"{wish_pct:.1f}%", "wishlist behaviour is rarely discussed"),
    ]
    _metric_row(cards)

    st.divider()
    st.subheader("3 key findings")
    st.caption("Each is a plain-English takeaway, the share of decision-moment posts it "
               "covers, and a real quote.")
    cols = st.columns(3)
    for col, f in zip(cols, FINDINGS):
        with col:
            with st.container(border=True):
                st.markdown(
                    f"<div style='font-weight:700;font-size:16px;color:{INK}'>{f['headline']}</div>"
                    f"<div style='color:{ACCENT};font-weight:700;font-size:15px;margin:6px 0'>"
                    f"{f['number']}</div>"
                    f"<div style='font-style:italic;color:{INK}'>“{f['quote']}”</div>"
                    f"<div style='color:{MUTED};font-size:12.5px;margin-top:6px'>"
                    f"— {f['source']}</div>",
                    unsafe_allow_html=True)

    st.write("")
    st.info("**What we couldn’t see.** Wishlist and saved-item behaviour is almost never "
            "described in public posts (only 1.1% of decision-moment posts), so this "
            "dashboard can say very little about it.")


# --------------------------------------------------------------------------- blockers

def tab_blockers() -> None:
    _howto(
        "This tab shows what stops shoppers before they buy, as a share of the 1,998 "
        "decision-moment posts. Longer bars are more common; the thin line is the likely "
        "range. Key takeaway: styling advice, finding items people saw elsewhere, and "
        "price checks account for most of it.")
    a1 = _num(load("opportunity_scorecard_prepurchase_a1.csv"),
              ["posts", "share_track_pct", "freq_ci95_low_pct", "freq_ci95_high_pct",
               "n_sources", "total_score"])
    if a1.empty:
        _missing("opportunity_scorecard_prepurchase_a1.csv")
        return
    a1 = a1.sort_values("share_track_pct", ascending=True)
    a1["label"] = a1["opportunity"].map(_label)

    fig = go.Figure(go.Bar(
        x=a1["share_track_pct"], y=a1["label"], orientation="h",
        marker_color=[ACCENT if v >= a1["share_track_pct"].nlargest(3).min() else NEUTRAL
                      for v in a1["share_track_pct"]],
        error_x=dict(type="data",
                     array=a1["freq_ci95_high_pct"] - a1["share_track_pct"],
                     arrayminus=a1["share_track_pct"] - a1["freq_ci95_low_pct"],
                     color=MUTED, thickness=1.2, width=4),
        text=[f"{v:.0f}%" for v in a1["share_track_pct"]],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{x:.1f}% of decision-moment posts<extra></extra>"))
    fig.update_layout(title="Styling advice, finding items seen elsewhere, and price "
                            "checks are the most common blockers",
                      xaxis_range=[0, float(a1["share_track_pct"].max()) * 1.25])
    _clean_bar(fig)
    st.plotly_chart(fig, width="stretch")
    st.caption("Thin line = likely range (95%). A post can mention more than one blocker, "
               "so the bars do not add up to 100%.")

    with st.expander("Technical details"):
        show = a1.sort_values("total_score", ascending=False)[
            ["label", "posts", "share_track_pct", "freq_ci95_low_pct", "freq_ci95_high_pct",
             "n_sources", "total_score", "opportunity", "member_codes"]].copy()
        show.columns = ["blocker", "posts", "share %", "range low", "range high", "sources",
                        "weighted score", "internal name", "member codes"]
        st.dataframe(show, hide_index=True, width="stretch")
        st.caption("Weighted score blends how common a blocker is, how severe it is, how "
                   "late in the journey it bites, and how fixable it is without spending "
                   "money. Member codes are the internal themes grouped into each blocker.")


# --------------------------------------------------------------------------- elsewhere

def tab_elsewhere() -> None:
    _howto(
        "This tab shows the apps and sites shoppers say they check outside Myntra before "
        "buying. Longer bars mean more mentions. Key takeaway: Amazon, Ajio and Flipkart "
        "are the most-checked alternatives.")
    ext = _num(load("external_sources_decision.csv"), ["count"])
    if ext.empty:
        _missing("external_sources_decision.csv")
        return
    ext["key"] = ext["source"].str.strip().str.lower()
    agg = (ext.groupby("key", as_index=False)["count"].sum()
           .sort_values("count", ascending=False).head(12))
    agg = agg.sort_values("count", ascending=True)
    agg["label"] = agg["key"].str.title()
    top3 = agg["count"].nlargest(3).min()

    fig = go.Figure(go.Bar(
        x=agg["count"], y=agg["label"], orientation="h",
        marker_color=[ACCENT if v >= top3 else NEUTRAL for v in agg["count"]],
        text=[_people(v) for v in agg["count"]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{x:,} mentions<extra></extra>"))
    fig.update_layout(title="Amazon, Ajio and Flipkart are the most-checked alternatives",
                      xaxis_range=[0, float(agg["count"].max()) * 1.2])
    _clean_bar(fig, height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption("Counts are mentions inside decision-moment posts, not unique shoppers.")


# --------------------------------------------------------------------------- trust

def tab_trust() -> None:
    _howto(
        "This tab shows the issues that follow a purchase — the things that drive returns "
        "and stop people coming back. The second chart asks how often posts about giving "
        "up mention a past bad experience. Key takeaway: delivery, customer support and "
        "product accuracy dominate after-purchase complaints.")
    tb = _num(load("opportunity_scorecard_trust.csv"),
              ["posts", "share_track_pct", "freq_ci95_low_pct", "freq_ci95_high_pct",
               "n_sources", "total_score"])
    if tb.empty:
        _missing("opportunity_scorecard_trust.csv")
        return
    tb = tb.sort_values("share_track_pct", ascending=True)
    tb["label"] = tb["opportunity"].map(_label)
    top3 = tb["share_track_pct"].nlargest(3).min()

    fig = go.Figure(go.Bar(
        x=tb["share_track_pct"], y=tb["label"], orientation="h",
        marker_color=[ACCENT if v >= top3 else NEUTRAL for v in tb["share_track_pct"]],
        error_x=dict(type="data",
                     array=tb["freq_ci95_high_pct"] - tb["share_track_pct"],
                     arrayminus=tb["share_track_pct"] - tb["freq_ci95_low_pct"],
                     color=MUTED, thickness=1.2, width=4),
        text=[f"{v:.0f}%" for v in tb["share_track_pct"]], textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>%{x:.1f}% of after-purchase posts<extra></extra>"))
    fig.update_layout(title="Delivery, customer support and product accuracy are the "
                            "biggest after-purchase issues",
                      xaxis_range=[0, float(tb["share_track_pct"].max()) * 1.25])
    _clean_bar(fig, height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption("Share of the 4,435 after-purchase posts (purchased or returned). A post "
               "can mention more than one issue.")

    st.write("")
    b = _num(load("bridge_by_stage.csv"), ["posts", "union_pct", "union_ci_low", "union_ci_high"])
    if not b.empty:
        b = b.copy()
        b["label"] = b["group"].map({"abandoned": "Gave up on the purchase",
                                     "other_prepurchase": "Still deciding"})
        fig2 = go.Figure(go.Bar(
            x=b["label"], y=b["union_pct"],
            marker_color=[ACCENT if g == "abandoned" else NEUTRAL for g in b["group"]],
            error_y=dict(type="data",
                         array=b["union_ci_high"] - b["union_pct"],
                         arrayminus=b["union_pct"] - b["union_ci_low"],
                         color=MUTED, thickness=1.2, width=4),
            text=[f"{v:.0f}%" for v in b["union_pct"]], textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>%{y:.1f}% mention a past bad experience<extra></extra>"))
        fig2.update_layout(title="Posts about giving up are far more likely to mention a "
                                 "past bad experience",
                           yaxis_range=[0, float(b["union_pct"].max()) * 1.25])
        _clean_bar(fig2, height=360)
        st.plotly_chart(fig2, width="stretch")
        st.info("Careful: “gave up” posts are, by definition, ones that did not convert, so "
                "this link is expected and is **not** proof that a past bad experience "
                "*caused* it. That is why the “What stops shoppers” tab sets these posts "
                "aside and looks only at item-level blockers.")

    with st.expander("Technical details"):
        show = tb.sort_values("total_score", ascending=False)[
            ["label", "posts", "share_track_pct", "total_score", "opportunity",
             "member_codes"]].copy()
        show.columns = ["issue", "posts", "share %", "weighted score", "internal name",
                        "member codes"]
        st.dataframe(show, hide_index=True, width="stretch")


# --------------------------------------------------------------------------- voices

def _blocker_by_code() -> dict[str, str]:
    """primary_code -> plain-English blocker label, from both scorecards."""
    mapping: dict[str, str] = {}
    for name in ("opportunity_scorecard_prepurchase_a1.csv",
                 "opportunity_scorecard_trust.csv"):
        sc = load(name)
        for _, r in sc.iterrows():
            label = _label(r["opportunity"])
            for code in _parse_pipe(r.get("member_codes", "")):
                mapping.setdefault(code, label)
    return mapping


def tab_voices() -> None:
    _howto(
        "This tab shows the actual posts behind every number, so you can read them "
        "yourself. Use the filters to narrow by blocker, stage or source, and click "
        "“View original” to open the post. Key takeaway: every finding here is traceable "
        "to a real quote — spot-check any of them before acting on it.")
    tagged = load("tagged_posts.csv")
    posts = load("relevant_posts.csv")
    if tagged.empty or posts.empty:
        _missing("tagged_posts.csv / relevant_posts.csv")
        return
    tagged = tagged[tagged["in_scope"] == "1"].copy()
    tagged["codes_list"] = tagged["codes"].map(_parse_json_list)

    merge_cols = [c for c in ("post_id", "url", "source") if c in posts.columns]
    df = tagged.merge(posts[merge_cols], on="post_id", how="left")
    code_map = _blocker_by_code()
    df["blocker"] = df["primary_code"].map(lambda c: code_map.get(c, "Other"))
    df["stage"] = df["journey_stage"].map(lambda s: STAGE_LABELS.get(s, s))
    df["source_label"] = df["source"].map(_source)

    c1, c2, c3 = st.columns(3)
    with c1:
        blockers = st.multiselect("Blocker", sorted(df["blocker"].unique()))
    with c2:
        stages = st.multiselect("Stage", [STAGE_LABELS[s] for s in DECISION_STAGES
                                          if s in set(df["journey_stage"])])
    with c3:
        sources = st.multiselect("Source", sorted(df["source_label"].dropna().unique()))

    sel = df
    if blockers:
        sel = sel[sel["blocker"].isin(blockers)]
    if stages:
        sel = sel[sel["stage"].isin(stages)]
    if sources:
        sel = sel[sel["source_label"].isin(sources)]

    n = len(sel)
    if "voices_n" not in st.session_state:
        st.session_state.voices_n = 10
    show_n = st.session_state.voices_n
    st.caption(f"{_people(n)} post(s) match. Showing {min(show_n, n)}.")

    for _, r in sel.head(show_n).iterrows():
        quote = html.escape(str(r.get("evidence_quote") or "").strip()) or "(no short quote)"
        url = str(r.get("url") or "").strip()
        link = (f" <a href='{html.escape(url)}' target='_blank' "
                f"style='color:{ACCENT};font-weight:600'>View original ↗</a>") if url else ""
        st.markdown(
            "<div style='border:1px solid #ECEFF3;border-left:4px solid " + ACCENT + ";"
            "border-radius:8px;padding:13px 16px;margin-bottom:11px;background:#FFFFFF'>"
            f"<div style='font-size:15px;color:{INK};margin-bottom:9px'>“{quote}”</div>"
            + _badge(html.escape(str(r['blocker'])), "#FCE4EC", "#AD1457")
            + _badge(html.escape(str(r['stage'])), "#EEF2F7", "#44506B")
            + _badge(html.escape(str(r['source_label'])), "#EEF2F7", "#44506B")
            + link + "</div>",
            unsafe_allow_html=True)

    if show_n < n:
        if st.button("Show more"):
            st.session_state.voices_n += 10
            st.rerun()


# --------------------------------------------------------------------------- how it works

def tab_how() -> None:
    _howto(
        "This tab explains how the numbers were produced and how reliable they are. Read "
        "the five steps first, then the funnel to see how many posts survived each check, "
        "and the reliability panel for how often the AI agreed with humans. Key takeaway: "
        "the pipeline is honest about its limits, and AI tagging matches human judgement "
        "well for barrier types but only moderately for topics.")
    clean = load("clean_posts.csv")
    tagged = load("tagged_posts.csv")
    bridge = load("bridge_check.csv")
    quality = load("engine_quality.csv")

    st.subheader("How it works")
    steps = [
        ("1 · Collect", "Gather public reviews and conversations from Google Play, the "
                        "App Store, Reddit and YouTube."),
        ("2 · Filter", "Remove duplicates and spam, then keep only posts about buying "
                       "clothes online."),
        ("3 · Tag", "The AI labels each post: journey stage, reason, blocker and a "
                    "verbatim quote."),
        ("4 · Validate", "Check the AI's labels against human labels and an independent "
                         "second AI."),
        ("5 · Score", "Group themes into blockers and rank them by how common, severe and "
                      "fixable they are."),
    ]
    cols = st.columns(5)
    for col, (title, body) in zip(cols, steps):
        with col:
            with st.container(border=True):
                st.markdown(f"<b>{title}</b>", unsafe_allow_html=True)
                st.caption(body)

    st.write("")
    st.subheader("The funnel")
    st.caption("Each step removes posts that don’t belong. Read top to bottom.")
    in_scope = int((tagged["in_scope"] == "1").sum()) if not tagged.empty else 0
    n_pre = int(((tagged["in_scope"] == "1") &
                 tagged["journey_stage"].isin(DECISION_STAGES)).sum()) if not tagged.empty else 0
    rows = [
        ("Collected & cleaned", len(clean),
         "public posts after removing duplicates and spam"),
        ("Relevant", len(tagged), "kept only talk about a fashion buying decision"),
        ("On topic", in_scope, "removed pure complaints with no buying decision"),
        ("Decision-moment", n_pre,
         "someone choosing, saving, comparing, postponing or giving up"),
        ("Item-level blockers", max(0, n_pre - len(bridge)),
         "removed posts citing a past bad experience, to isolate item-level blockers"),
    ]
    maxn = max(r[1] for r in rows) or 1
    blocks = []
    for name, count, cap in rows:
        width = count / maxn * 100
        blocks.append(
            "<div style='margin-bottom:14px'>"
            "<div style='display:flex;justify-content:space-between;font-size:14px;"
            f"color:{INK}'><b>{name}</b><b>{_people(count)}</b></div>"
            "<div style='background:#EEF2F7;border-radius:6px;height:20px;margin:4px 0'>"
            f"<div style='width:{width:.1f}%;background:{ACCENT};height:20px;"
            "border-radius:6px'></div></div>"
            f"<div style='color:{MUTED};font-size:12.5px'>{cap}</div></div>")
    st.markdown("".join(blocks), unsafe_allow_html=True)

    st.write("")
    st.subheader("How reliable")
    r1, r2, r3 = st.columns(3)
    with r1:
        with st.container(border=True):
            st.markdown("**Barrier types**")
            st.markdown(_badge("87% match · meets target", "#E6F4EA", "#1E7E34"),
                        unsafe_allow_html=True)
            st.caption("AI agreed with human labels on the type of barrier.")
    with r2:
        with st.container(border=True):
            st.markdown("**Topics**")
            st.markdown(_badge("72% match · below target", "#FDE7E9", "#B3261E"),
                        unsafe_allow_html=True)
            st.caption("AI agreed with human labels on the main topic. Treat topic "
                       "splits as indicative.")
    with r3:
        with st.container(border=True):
            st.markdown("**Coverage**")
            st.markdown(_badge("Google Play, Reddit, YouTube", "#E6F4EA", "#1E7E34"),
                        unsafe_allow_html=True)
            st.caption("App Store data is too thin to rely on.")

    st.write("")
    st.warning("**Limitations.** Public posts skew towards complaints, so happy customers "
               "are under-represented. Wishlist and saved-item behaviour is rarely "
               "discussed. App Store data is thin. And scope decisions (which posts count "
               "as a buying decision) vary somewhat between AI models.")

    with st.expander("Technical details"):
        if quality.empty:
            _missing("engine_quality.csv")
        else:
            q = quality.copy()
            st.dataframe(q[["metric", "value", "target", "status", "method", "n"]],
                         hide_index=True, width="stretch")
            st.caption("Full quality report: engine_quality.csv. Kappa (agreement beyond "
                       "chance) was 0.81 for barrier types and 0.70 for topics.")


# --------------------------------------------------------------------------- main

def main() -> None:
    tabs = st.tabs([
        "Summary",
        "What stops shoppers",
        "Where shoppers look elsewhere",
        "After-purchase trust",
        "Real voices",
        "How it works & how reliable",
    ])
    with tabs[0]:
        tab_summary()
    with tabs[1]:
        tab_blockers()
    with tabs[2]:
        tab_elsewhere()
    with tabs[3]:
        tab_trust()
    with tabs[4]:
        tab_voices()
    with tabs[5]:
        tab_how()


main()
