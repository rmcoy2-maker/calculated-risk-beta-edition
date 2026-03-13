from __future__ import annotations

# ---- recovered app shims ----
try:
    from lib.compliance_gate import require_eligibility
except Exception:
    def require_eligibility(*args, **kwargs):
        return True

try:
    from lib.access import require_allowed_page, beta_banner, live_enabled, premium_enabled
except Exception:
    def require_allowed_page(*args, **kwargs):
        return None
    def beta_banner(*args, **kwargs):
        return None
    def live_enabled(*args, **kwargs):
        return False
    def premium_enabled(*args, **kwargs):
        return True

def do_expensive_refresh():
    return None

try:
    from app.lib.auth import login, show_logout
except Exception:
    def login(required: bool = False):
        class _Auth:
            ok = True
            authenticated = True
        return _Auth()
    def show_logout():
        return None

try:
    from app.utils.diagnostics import mount_in_sidebar
except Exception:
    def mount_in_sidebar(page_name: str):
        return None

try:
    from app.utils.parlay_ui import selectable_odds_table
except Exception:
    def selectable_odds_table(*args, **kwargs):
        return None

try:
    from app.utils.parlay_cart import read_cart, add_to_cart, clear_cart
except Exception:
    import pandas as _shim_pd
    def read_cart():
        return _shim_pd.DataFrame()
    def add_to_cart(*args, **kwargs):
        return None
    def clear_cart():
        return None

try:
    from app.utils.nudge import begin_session, touch_session, session_duration_str, bump_usage, show_nudge
except Exception:
    def begin_session(): return None
    def touch_session(): return None
    def session_duration_str(): return ""
    def bump_usage(*args, **kwargs): return None
    def show_nudge(*args, **kwargs): return None
# ---- /recovered app shims ----

import time

import sys
from pathlib import Path
import streamlit as st

_here = Path(__file__).resolve()
for up in [_here] + list(_here.parents):
    cand = up / 'serving_ui' / 'app' / '__init__.py'
    if cand.exists():
        base = str((up / 'serving_ui').resolve())
        if base not in sys.path:
            sys.path.insert(0, base)
        break
PAGE_PROTECTED = False
auth = login(required=PAGE_PROTECTED)
if not auth.ok:
    st.stop()
show_logout()

auth = login(required=False)
if not auth.authenticated:
    st.info('You are in read-only mode.')
show_logout()
import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
_SERVING_UI = _HERE.parents[2]
if str(_SERVING_UI) not in sys.path:
    sys.path.insert(0, str(_SERVING_UI))
st.set_page_config(page_title='94 Legal Terms Privacy', page_icon='📈', layout='wide')

# === Nudge+Session (auto-injected) ===
try:
    from app.utils.nudge import begin_session, touch_session, session_duration_str, bump_usage, show_nudge  # type: ignore
except Exception:
    def begin_session(): pass
    def touch_session(): pass
    def session_duration_str(): return ""
    bump_usage = lambda *a, **k: None
    def show_nudge(*a, **k): pass

# Initialize/refresh session and show live duration
begin_session()
touch_session()
if hasattr(st, "sidebar"):
    st.sidebar.caption(f"🕒 Session: {session_duration_str()}")

# Count a lightweight interaction per page load
bump_usage("page_visit")

# Optional upsell banner after threshold interactions in last 24h
show_nudge(feature="analytics", metric="page_visit", threshold=10, period="1D", demo_unlock=True, location="inline")
# === /Nudge+Session (auto-injected) ===

# === Nudge (auto-injected) ===
try:
    from app.utils.nudge import bump_usage, show_nudge  # type: ignore
except Exception:
    bump_usage = lambda *a, **k: None
    def show_nudge(*a, **k): pass

# Count a lightweight interaction per page load
bump_usage("page_visit")

# Show a nudge once usage crosses threshold in the last 24h
show_nudge(feature="analytics", metric="page_visit", threshold=10, period="1D", demo_unlock=True, location="inline")
# === /Nudge (auto-injected) ===

try:
    from app.utils.diagnostics import mount_in_sidebar
except ModuleNotFoundError:
    try:
        import sys
        from pathlib import Path as _efP
        sys.path.append(str(_efP(__file__).resolve().parents[3]))
        from app.utils.diagnostics import mount_in_sidebar
    except Exception:
        try:
            from utils.diagnostics import mount_in_sidebar
        except Exception:

            def mount_in_sidebar(page_name: str):
                return None
import io
from textwrap import dedent
from pathlib import Path
import datetime as _dt
APP_NAME = 'Edge Finder'
ORG_NAME = 'Calculated Risk'
CONTACT_EMAIL = 'cs.calculatedrisk.com'
CONTACT_ADDRESS = 'Lexington, KY USA'
st.title('⚖️ Legal')
TODAY = _dt.date.today().strftime('%B %d, %Y')

def md(s: str):
    st.markdown(dedent(s), unsafe_allow_html=False)

def make_pdf_from_markdown(md_text: str) -> bytes | None:
    """Create a lightweight PDF (if reportlab is installed)."""
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=LETTER)
        W, H = LETTER
        L = 0.9 * inch
        T = H - 0.9 * inch
        lh = 12
        max_chars = 95
        c.setFont('Helvetica-Bold', 14)
        c.drawString(L, T, f'{APP_NAME} — Legal')
        y = T - 20
        c.setFont('Helvetica', 10)
        for line in md_text.splitlines():
            if not line.strip():
                y -= lh
            else:
                while len(line) > max_chars:
                    c.drawString(L, y, line[:max_chars])
                    y -= lh
                    line = line[max_chars:]
                    if y < 0.9 * inch:
                        c.showPage()
                        c.setFont('Helvetica', 10)
                        y = T
                c.drawString(L, y, line)
                y -= lh
            if y < 0.9 * inch:
                c.showPage()
                c.setFont('Helvetica', 10)
                y = T
        c.showPage()
        c.save()
        buf.seek(0)
        return buf.read()
    except Exception:
        return None
TERMS_MD = dedent(f'''
# Terms of Service
**Effective Date:** {TODAY}  
**Applies to:** {APP_NAME} by {ORG_NAME}

## 1. Acceptance of Terms
By accessing or using {APP_NAME} (the “Service”), you agree to be bound by these Terms of Service (“Terms”). If you do not agree, do not use the Service.

## 2. Eligibility & Responsible Use
You must be at least 18 years old (or 21 years old where required by law) to use this Service. The Service is not available in all states. Specifically, access is restricted in **Washington, Idaho, and Nevada** due to state laws. We may expand or reduce availability at our discretion. You are responsible for complying with all applicable laws and regulations in your location.

## 3. No Sportsbook, No Money Transmission
We are **not a sportsbook**. The Service never processes or facilitates real money bets, wagers, deposits, or payouts. No money passes through the Service other than subscription or access fees. All features are provided for **analytics, educational, entertainment, and informational purposes only**. You acknowledge that the Service is a **third-party analytics platform** and not affiliated with any sportsbook or gambling operator.

## 4. No Financial or Betting Advice
{APP_NAME} provides **informational and educational** analytics only. Nothing in the Service constitutes financial, investment, or betting advice. Past performance does not guarantee future results. You assume full responsibility for decisions you make.

## 5. Accounts & Security
If the Service allows user accounts, you are responsible for safeguarding credentials and for all activity under your account. Notify us promptly at {CONTACT_EMAIL} of any unauthorized use.

## 6. Data Sources & Availability
The Service may rely on third-party data feeds and public or proprietary datasets. We do not warrant the accuracy, completeness, timeliness, or availability of any data or the Service itself and may change or discontinue features at any time.

## 7. Acceptable Use
You agree not to:
- Reverse engineer, scrape at scale without permission, or circumvent security controls.
- Upload malware or content that is unlawful, infringing, or harmful.
- Use the Service in violation of applicable law or third-party rights.

## 8. Intellectual Property
All content, features, and functionality of the Service are owned by {ORG_NAME} or its licensors and are protected by applicable IP laws. You receive a limited, non-exclusive, non-transferable license to use the Service for personal, non-commercial purposes (unless otherwise agreed in writing).

## 9. Paid Features (if any)
Fees (if charged) are due as stated at purchase. Except where required by law, payments are non-refundable. We may change pricing with notice for future billing cycles.

## 10. Third-Party Services
The Service may link to third-party sites or integrate third-party services (e.g., sportsbooks, analytics). We are not responsible for their content, policies, or practices.

## 11. Disclaimers
THE SERVICE IS PROVIDED “AS IS” AND “AS AVAILABLE.” TO THE MAXIMUM EXTENT PERMITTED BY LAW, {ORG_NAME} DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, INCLUDING MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.

## 12. Limitation of Liability
TO THE MAXIMUM EXTENT PERMITTED BY LAW, {ORG_NAME} AND ITS AFFILIATES, OFFICERS, EMPLOYEES, AND AGENTS SHALL NOT BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, EXEMPLARY, OR PUNITIVE DAMAGES, OR ANY LOSS OF PROFITS, REVENUE, DATA, OR GOODWILL, ARISING FROM OR RELATED TO YOUR USE OF THE SERVICE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. OUR TOTAL LIABILITY SHALL NOT EXCEED THE AMOUNT YOU PAID FOR THE SERVICE IN THE 3 MONTHS PRECEDING THE CLAIM (OR USD $100 IF NO FEES WERE PAID).

## 13. Indemnification
You agree to indemnify and hold harmless {ORG_NAME} from any claims, losses, or expenses (including reasonable attorneys’ fees) arising out of your use of the Service or violation of these Terms.

## 14. Changes to the Service or Terms
We may modify or discontinue the Service or update these Terms at any time. Material changes will be posted in-app or via reasonable notice. Continued use after changes constitutes acceptance.

## 15. Termination
We may suspend or terminate your access at any time, with or without notice, for any reason, including breach of these Terms.

## 16. Governing Law & Dispute Resolution
These Terms are governed by the laws of the Commonwealth of Kentucky and applicable U.S. federal law, without regard to conflict-of-laws rules. You agree to the exclusive jurisdiction and venue of state and federal courts located in Fayette County, Kentucky.

## 17. Contact
Questions about these Terms? Contact us at **{CONTACT_EMAIL}** or mail **{CONTACT_ADDRESS}**.
''').strip()
PRIVACY_MD = dedent(f'''
# Privacy Policy
**Effective Date:** {TODAY}  
**Applies to:** {APP_NAME} by {ORG_NAME}

## 1. Overview
This Privacy Policy explains what information we collect, how we use it, and your choices. By using {APP_NAME}, you agree to the practices described here.

## 2. Information We Collect
- **Account & Contact Data:** name, email, and any details you provide when contacting support.
- **Usage Data:** app interactions, page views, diagnostic logs, device and browser information.
- **Files & Inputs:** CSV uploads or data you load into the app.
- **Cookies & Local Storage:** used to maintain sessions, preferences, and improve functionality.

## 3. How We Use Information
- To operate, maintain, and improve the Service.
- To provide support and respond to inquiries.
- To analyze performance and develop new features.
- To enforce Terms, prevent abuse, and ensure security.

## 4. Processing Bases
Where required, we rely on one or more legal bases: your consent, contractual necessity, legitimate interests (e.g., security, analytics), or compliance with legal obligations.

## 5. Sharing & Disclosure
We do not sell your personal information. We may share information with:
- **Service Providers** who help us operate the Service (e.g., hosting, analytics) under appropriate confidentiality and security commitments.
- **Legal & Safety** when required by law, regulation, or to protect rights, safety, and integrity.
- **Business Transfers** in connection with a merger, acquisition, or asset sale.

## 6. Data Retention
We retain information as long as necessary for the purposes described above or as required by law. We may anonymize or aggregate data for longer-term analytics.

## 7. Security
We use reasonable administrative, technical, and physical safeguards to protect information. No method of transmission or storage is 100% secure.

## 8. Your Choices & Rights
- **Access/Correction/Deletion:** contact **{CONTACT_EMAIL}** to request.
- **Email Preferences:** you can opt out of non-essential emails.
- **Cookies:** you can control cookies via browser settings; some features may not function without them.
- **Do Not Track:** we do not respond to DNT signals at this time.

## 9. Children’s Privacy
The Service is not directed to children under 13 (or under 16 in some regions). We do not knowingly collect personal data from children.

## 10. International Users
If you access the Service from outside the U.S., you consent to processing in the U.S., where laws may differ from those in your country.

## 11. Changes to this Policy
We may update this Policy. Material changes will be posted in-app or via reasonable notice. Your continued use constitutes acceptance.

## 12. Contact
For privacy questions or requests, contact **{CONTACT_EMAIL}** or mail **{CONTACT_ADDRESS}**.
''').strip()
st.sidebar.link_button('Terms of Service', '#terms-of-service', width='stretch')
st.sidebar.link_button('Privacy Policy', '#privacy-policy', width='stretch')
t_terms, t_priv = st.tabs(['Terms of Service', 'Privacy Policy'])
with t_terms:
    md(TERMS_MD)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.download_button('⬇️ Download Terms (Markdown)', data=TERMS_MD.encode('utf-8'), file_name=f"{APP_NAME.replace(' ', '_')}_Terms_of_Service.md", mime='text/markdown', width='stretch')
    with c2:
        pdf_bytes = make_pdf_from_markdown(TERMS_MD)
        if pdf_bytes:
            st.download_button('⬇️ Download Terms (PDF)', data=pdf_bytes, file_name=f"{APP_NAME.replace(' ', '_')}_Terms_of_Service.pdf", mime='application/pdf', width='stretch')
        else:
            st.info('Install **reportlab** for PDF export: `pip install reportlab`', icon='ℹ️')
with t_priv:
    md(PRIVACY_MD)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.download_button('⬇️ Download Privacy (Markdown)', data=PRIVACY_MD.encode('utf-8'), file_name=f"{APP_NAME.replace(' ', '_')}_Privacy_Policy.md", mime='text/markdown', width='stretch')
    with c2:
        pdf_bytes = make_pdf_from_markdown(PRIVACY_MD)
        if pdf_bytes:
            st.download_button('⬇️ Download Privacy (PDF)', data=pdf_bytes, file_name=f"{APP_NAME.replace(' ', '_')}_Privacy_Policy.pdf", mime='application/pdf', width='stretch')
        else:
            st.info('Install **reportlab** for PDF export: `pip install reportlab`', icon='ℹ️')
st.divider()
st.caption(f'© {ORG_NAME} — {APP_NAME}. This page is generic boilerplate and not legal advice.')
