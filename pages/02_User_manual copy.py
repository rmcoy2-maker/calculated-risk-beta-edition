from __future__ import annotations

import streamlit as st
from pathlib import Path

# ------------------------------------------------------------
# Page config (exactly one)
# ------------------------------------------------------------
st.set_page_config(
    page_title="User Manual • Calculated Risk",
    page_icon="📘",
    layout="wide",
)

# ------------------------------------------------------------
# Static assets discovery (logo, QR, manual files)
# Put these under any of the probed /static/ folders:
#   - serving_ui/app/static
#   - serving_ui/static
#   - <project_root>/static
# Files:
#   - logo.png
#   - qr_calculatedrisk.png
#   - CalculatedRisk_User_Manual_Branded_Full_WithTOC.docx  (editable)
#   - CalculatedRisk_User_Manual_Brand_Full.pdf             (optional)
# ------------------------------------------------------------
HERE = Path(__file__).resolve()
STATIC_CANDIDATES = [
    HERE.parent / "static",
    HERE.parent.parent / "static",
    HERE.parents[2] / "static",
    Path.cwd() / "static",
]

def find_static_file(name: str) -> Path | None:
    for root in STATIC_CANDIDATES:
        p = root / name
        if p.exists():
            return p
    return None

DOCX_NAME = "CalculatedRisk_User_Manual_Branded_Full_WithTOC.docx"
PDF_NAME  = "CalculatedRisk_User_Manual_Brand_Full.pdf"

DOCX_PATH = find_static_file(DOCX_NAME)
PDF_PATH  = find_static_file(PDF_NAME)
LOGO_PATH = find_static_file("logo.png")
QR_PATH   = find_static_file("qr_calculatedrisk.png")

# ------------------------------------------------------------
# Header (brand + tagline)
# ------------------------------------------------------------
left, mid, right = st.columns([1, 4, 1])

with left:
    if LOGO_PATH:
        st.image(str(LOGO_PATH), use_column_width=True)

with mid:
    st.markdown(
        """
        # **Calculated Risk**
        **AI-powered Sports Analytics**  
        _“I didn’t just beat the odds — I rearranged them.”_
        """,
        help="Calculated Risk user manual (public preview)"
    )

with right:
    if QR_PATH:
        st.image(str(QR_PATH), caption="Scan to learn more", use_column_width=True)

st.divider()

# ------------------------------------------------------------
# “What you get” teaser + downloads
# ------------------------------------------------------------
st.subheader("What’s inside the full manual")
st.markdown(
    """
- Platform overview: **Line Shop, Edge Finder, Backtests, Parlays, Bankroll, Analytics**  
- Clear tiers: **Rookie**, **Pro**, **HOF** — what each level unlocks  
- The math made simple: **EV, CLV, Parlay EV, Kelly**, risk of ruin  
- Step-by-step **how-to** for every page (what to click, how to read results)
    """
)

dl1, dl2, dl3 = st.columns(3)
with dl1:
    if DOCX_PATH:
        st.download_button(
            "📥 Download Manual (Word, editable)",
            data=DOCX_PATH.read_bytes(),
            file_name=DOCX_NAME,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    else:
        st.info("Add the Word manual to a `static/` folder to enable this button.")

with dl2:
    if PDF_PATH:
        st.download_button(
            "📄 Download Manual (PDF)",
            data=PDF_PATH.read_bytes(),
            file_name=PDF_NAME,
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.caption("Optional PDF: place in `static/` to enable.")

with dl3:
    if PDF_PATH:
        b64 = base64.b64encode(PDF_PATH.read_bytes()).decode("utf-8")
        st.markdown(
            f'<a href="data:application/pdf;base64,{b64}" target="_blank">Open PDF in a new tab</a>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Add a PDF to enable web preview.")

st.divider()

# ------------------------------------------------------------
# Tiers — one tab per tier (clean & scannable)
# ------------------------------------------------------------
tab_rookie, tab_pro, tab_hof = st.tabs(["🟢 Rookie", "🔵 Pro", "🟣 HOF"])

with tab_rookie:
    st.markdown("### Rookie — $20/month")
    st.markdown(
        """
| **Feature** | **Details** |
|---|---|
| All Picks Explorer | Basic filters; 50 rows/session |
| Lines — Explorer | Open vs live lines (daily), simplified |
| Backtest — Scores | Last 7 days |
| Parlay Builder | Up to 3 legs; EV preview |
| Parlay Scored Exp. | Settled legs & P/L |
| Micro Calculations | Top EV singles |
| History | 30-day view |
| Alerts | — |
| API / Exports | Manual CSV |
| Support | KB + email |
        """.strip()
    )

with tab_pro:
    st.markdown("### Pro — $55/month")
    st.markdown(
        """
| **Feature** | **Details** |
|---|---|
| All Picks Explorer | Advanced filters (props, books) |
| Lines — Explorer | More details; open/close comparison |
| Backtest — Scores | 90-day horizon; hit-rate by filter |
| Parlay Builder | Up to 8 legs; correlated-leg detection |
| Parlay Scored Exp. | ROI by leg count/market/book |
| Micro Calculations | Singles + simple combos; EV insights |
| History | 180-day view; per-book split |
| Alerts | Email alerts (price move / edge) |
| API / Exports | CSV + JSON; schema docs |
| Support | Priority email |
        """.strip()
    )

with tab_hof:
    st.markdown("### HOF (Elite) — $80/month")
    st.markdown(
        """
| **Feature** | **Details** |
|---|---|
| All Picks Explorer | Full library; 50k+ rows; unlimited exports |
| Lines — Explorer | Line-move history, CLV tracking |
| Backtest — Scores | Multi-season, cohort analysis, confidence bands |
| Parlay Builder | Unlimited legs; correlated guards; stash & share |
| Parlay Scored Exp. | Advanced analytics; Kelly suggestions |
| Micro Calculations | Multileg; sensitivity sliders; batch exports |
| History | Unlimited; tax-year export |
| Alerts | Email + webhook; custom rules |
| API / Exports | Full API + webhooks |
| Support | Priority + office hours |
        """.strip()
    )

st.divider()

# ------------------------------------------------------------
# The Math (teaser)
# ------------------------------------------------------------
st.subheader("The math that powers it")
st.markdown(
    r"""
- **Expected Value (EV)**: \( EV = p_{\text{win}}\cdot \text{odds}_{\text{dec}} - (1 - p_{\text{win}}) \)  
- **Closing Line Value (CLV)**: \( CLV = \frac{\text{odds placed}}{\text{closing odds}} - 1 \)  
- **Parlay EV**: \( P_{\text{parlay}} = \prod_i p_i \), then \( EV = P_{\text{parlay}}\cdot O_{\text{parlay}} - (1 - P_{\text{parlay}}) \)  
- **Kelly (stake sizing)**: \( f^* = \frac{bp - q}{b} \), where \( b=\text{odds}_{\text{dec}}-1,\ q=1-p \)

*The full manual includes worked examples and guidance on when to use (or avoid) Kelly.*
    """
)

# ------------------------------------------------------------
# Advanced features (teaser)
# ------------------------------------------------------------
st.subheader("Advanced features (high-level)")
st.markdown(
    """
- **Line Shop** — compare prices across books; find best available number fast  
- **Edge Finder** — filter by EV, market, and book; export candidate plays  
- **Backtest** — validate strategy over time; view ROI, hit rate, CLV distribution  
- **Parlay Builder & Scored Explorer** — build/simulate parlays; analyze settled performance  
- **Bankroll Tracker** — P/L, unit growth, and risk metrics  
- **Analytics Hub (Pro/HOF)** — cohort analysis, calibration and confidence bands  
    """
)

st.divider()

# ------------------------------------------------------------
# CTA
# ------------------------------------------------------------
st.markdown(
    """
### Ready to go deeper?
Download the **editable Word manual** above, or explore the app with a **Guest Pass**.  
Questions? **cs@calculatedrisk.group**
    """
)

# ------------------------------------------------------------
# Expanded Manual Sections (UNGATED)
# ------------------------------------------------------------
st.subheader("Executive Summary")
st.markdown("""
- **Calculated Risk** is an analytics platform for sports fans and sharps. It is **informational only** — not a sportsbook.  
- This manual is a clear on-ramp for **Rookie** users, with a growth path to **Pro** and **HOF** tiers as you gain confidence.
""")

st.subheader("Getting Started")
st.markdown("""
- **Eligibility:** Full app enforces age **21+**, U.S. only, excluding **ID, NV, WA**. The **User Manual** can run in demo mode (no sensitive actions).  
- **Units:** Choose a small fixed unit (e.g., **1–2% of bankroll**). Examples in this manual are framed in units.
""")

st.subheader("Feature Roadmap: Rookie → Pro → HOF")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
**Rookie**
- Line Shopping 101
- Flat unit sizing
- Break-even % & EV basics
- Odds ↔ Probability cheatsheet
""")
with c2:
    st.markdown("""
**Pro**
- CLV tracking
- Standard Deviation & Std. Error
- Edge Score (model vs implied)
- Backtesting overview
""")
with c3:
    st.markdown("""
**HOF**
- Kelly & fractional Kelly
- Model vs Market diagnostics
- Ghost Parlays planning
- Season-long portfolio metrics
""")
st.info("Start as **Rookie** (better prices, unit sizing) → move to **Pro** for CLV & confidence → **HOF** for Kelly and advanced diagnostics.")
st.divider()

st.subheader("Line Shopping 101 (Rookie)")
st.markdown(r"""
- **Goal:** Incrementally increase EV by taking the **best available price** across books.  
- **Break-even probability:** \( p_{\text{BE}} = \frac{1}{\text{odds}_{\text{dec}}} \)  
  −110 ⇒ 1.909 ⇒ 52.38% • +120 ⇒ 2.20 ⇒ 45.45%  
- **EV (single):** \( EV = p_{\text{win}} \times \text{odds}_{\text{dec}} - (1 - p_{\text{win}}) \)  
- **Example (p_win = 0.55):** Book A −110 (1.909) → EV = 0.55×1.909 − 0.45 = **+0.105** u;  
  Book B −105 (1.952) → EV = 0.55×1.952 − 0.45 = **+0.122** u. → Take the better price.
""")

st.subheader("Bankroll & Sizing (Rookie → HOF)")
st.markdown(r"""
- **Flat Units (Rookie):** Pick a small fixed unit (**1–2% bankroll**) and use it consistently.  
- **Kelly (HOF-ready):** \( f^* = \frac{bp - q}{b} \), where \( b = \text{dec} - 1 \), \( p = p_{\text{win}} \), \( q = 1 - p \).  
  Example: +150 (2.5), \(p=0.45\) ⇒ \(b=1.5\), \(q=0.55\) ⇒ \( f^* = \frac{1.5\times0.45 - 0.55}{1.5} = 0.116 \) ⇒ **11.6%** bankroll.  
  Many use **½-Kelly** or **¼-Kelly** to reduce drawdowns.  
- **Guardrails:** Daily/weekly **profit-lock** and **stop-loss** caps (e.g., **+5u lock**, **−3u stop**).
""")

st.subheader("Backtesting & Confidence (Pro)")
st.markdown(r"""
- **Win rate** \( \hat{p} \) over \( n \) historical picks.  
- **Std. Dev (per pick):** \( \sigma = \sqrt{\hat{p}(1-\hat{p})} \) • **Std. Error:** \( \text{SE} = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} \)  
- **95% CI:** \( \hat{p} \pm 1.96 \times \text{SE} \)  
- **Example:** \( n=400 \), \( \hat{p}=0.54 \) ⇒ \( \text{SE}=\sqrt{0.54\times0.46/400}=0.0249 \) → CI ≈ **[0.491, 0.589]**.  
- **Interpretation:** Edge is plausible if CI stays **above** break-even for your typical odds.
""")

st.subheader("Parlays & Ghost Parlays (HOF)")
st.markdown(r"""
- **Hit probability:** \( P_{\text{parlay}} = \prod_i p_i \) • **Decimal odds:** \( O_{\text{parlay}} = \prod_i O_i \)  
- **EV:** \( EV_{\text{parlay}} = P_{\text{parlay}} \times O_{\text{parlay}} - (1 - P_{\text{parlay}}) \)  
- **Example:** Two legs, each \(p=0.60\), both −110 (1.909): \( P=0.36 \), \( O=3.64 \),  
  \( EV = 0.36\times3.64 - 0.64 = \mathbf{+0.71} \) units.  
- **Caution:** Variance is **much higher** — size parlays **smaller** than singles. Use **Ghost Parlays** to plan.
""")

st.subheader("Edge Score & Alerts (Pro)")
st.markdown(r"""
- **Implied probability from American odds (A):**  
  If \(A>0: p = \frac{100}{A+100}\); if \(A<0: p = \frac{|A|}{|A|+100}\).  
- **Edge Score (simple):** \( \text{Edge} = p_{\text{model}} - p_{\text{implied}} \).  
- **Example:** Model \( p=0.58 \); best market −110 ⇒ implied ≈ **0.5238** ⇒ **Edge = 5.62%**.  
  Alerts trigger when Edge > your threshold (e.g., **3–5%**).
""")

st.subheader("Closing Line Value (CLV) (Pro)")
st.markdown(r"""
- **Formula:** \( \text{CLV} = \frac{\text{odds placed}}{\text{closing odds}} - 1 \)  
- **Example:** Bet at +120 (2.20); closed +100 (2.00) ⇒ \( \text{CLV} = \frac{2.20}{2.00} - 1 = \mathbf{+10\%} \).  
- **Meaning:** Sustained **+CLV** suggests long-term edge.
""")

st.subheader("Odds & Probability Conversions (Rookie)")
st.markdown(r"""
- **American → Decimal:** \( \text{dec} = 1 + \frac{A}{100} \) if \(A>0\); \( \text{dec} = 1 + \frac{100}{|A|} \) if \(A<0\).  
- **Implied Probability:** \( p = \frac{100}{A+100} \) if \(A>0\); \( p = \frac{|A|}{|A|+100} \) if \(A<0\).  
- **Break-even:** \( p_{\text{BE}} = \frac{1}{\text{dec}} \).
""")

st.subheader("Safeguards & Compliance")
st.markdown("""
- Calculated Risk is **informational only**; we **do not accept wagers**.  
- Use **units**, set **guardrails**, and consider **fractional Kelly** to reduce drawdowns.  
- Age/location rules apply in the full app.
""")

st.subheader("Appendix A — Formulas & Examples")
st.markdown(r"""
- **EV (single):** \( EV = p_{\text{win}} \times \text{dec} - (1 - p_{\text{win}}) \).  
  Example: \( p=0.60, \text{dec}=2.5 \Rightarrow EV = 1.5 - 0.4 = \mathbf{+1.1} \) units.  
- **CLV:** \( \text{CLV} = \frac{\text{odds placed}}{\text{closing odds}} - 1 \).  
  Example: \( 2.20/2.00 - 1 = \mathbf{+10\%} \).  
- **Parlay:** \( P_{\text{parlay}} = \prod_i p_i \); \( EV_{\text{parlay}} = P_{\text{parlay}} \times O_{\text{parlay}} - (1 - P_{\text{parlay}}) \).  
- **Kelly:** \( f^* = \frac{bp - q}{b} \) with \( b = \text{dec} - 1 \), \( q = 1 - p \).
""")

st.subheader("Glossary")
st.markdown("""
- **Unit:** Fixed fraction of bankroll (e.g., 1–2%) used to standardize stake sizing.  
- **Break-even %:** Win rate required to be long-run neutral at given odds.  
- **CLV:** Closing Line Value; price you got vs. market close.  
- **Edge Score:** Model probability minus implied probability from the best available odds.
""")

st.divider()
