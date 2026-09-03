"""Professional light-blue visual system for AviMaint-DSS.

Presentation-only module. It intentionally does not alter retrieval, extraction,
calibration, or frozen research artifacts.
"""
from __future__ import annotations

import streamlit as st

# Product palette ----------------------------------------------------------- #
PLANE = "#f6f8fc"
SURFACE = "#ffffff"
SURFACE2 = "#f8fafc"
INK = "#0f172a"
INK2 = "#334155"
MUTED = "#64748b"
HAIR = "#e2e8f0"
PRIMARY = "#2563eb"
PRIMARY_SOFT = "#eff6ff"

SERIES = ["#2563eb", "#0ea5e9", "#10b981", "#f59e0b", "#8b5cf6", "#14b8a6", "#6366f1", "#ef4444"]
STATUS = {"good": "#16a34a", "warning": "#d97706", "serious": "#ea580c", "critical": "#dc2626"}

BADGE = {
    "strong": ("#16a34a", "Strong historical evidence"),
    "moderate": ("#d97706", "Moderate historical evidence"),
    "exploratory": ("#64748b", "Exploratory evidence"),
    "abstain": ("#dc2626", "Nearest cases only"),
}


def setup_page():
    st.set_page_config(
        page_title="AviMaint-DSS",
        page_icon="🛩️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        f"""
        <style>
        :root {{
          --av-primary:{PRIMARY}; --av-primary-soft:{PRIMARY_SOFT};
          --av-bg:{PLANE}; --av-surface:{SURFACE}; --av-surface-2:{SURFACE2};
          --av-text:{INK}; --av-text-2:{INK2}; --av-muted:{MUTED}; --av-hair:{HAIR};
          --av-shadow:0 1px 2px rgba(15,23,42,.04), 0 8px 28px rgba(15,23,42,.055);
        }}

        html, body, [class*="css"] {{ font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
          "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
        .stApp {{ background:var(--av-bg); color:var(--av-text); -webkit-font-smoothing:antialiased; }}
        section[data-testid="stSidebar"] {{ display:none !important; }}
        #MainMenu, footer, [data-testid="stDecoration"] {{ visibility:hidden !important; }}
        header[data-testid="stHeader"] {{ background:transparent !important; height:.25rem !important; }}
        [data-testid="stToolbar"] {{ visibility:hidden !important; height:0 !important; }}
        .block-container {{
          padding-top:1rem !important; padding-bottom:3rem !important;
          padding-left:clamp(18px,3vw,54px) !important;
          padding-right:clamp(18px,3vw,54px) !important;
          max-width:none !important; width:100% !important;
        }}

        h1,h2,h3,h4,h5,h6 {{ color:var(--av-text) !important; letter-spacing:-.025em; font-weight:700; }}
        h3 {{ font-size:1.55rem !important; margin:.2rem 0 .2rem !important; }}
        p, label, .stMarkdown {{ color:var(--av-text-2); }}
        a {{ color:var(--av-primary); }}

        /* Product shell */
        .brand-shell {{
          display:flex; align-items:center; gap:12px; min-height:46px;
          padding:2px 0 6px 0;
        }}
        .brand-mark {{
          width:36px; height:36px; border-radius:10px; display:inline-flex; align-items:center;
          justify-content:center; color:#fff; font-size:19px; flex:0 0 36px;
          background:linear-gradient(145deg,#1d4ed8,#3b82f6); box-shadow:0 6px 18px rgba(37,99,235,.22);
        }}
        .brand-copy {{ display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }}
        .brand-name {{ font-size:19px; line-height:1; font-weight:800; color:var(--av-text); letter-spacing:-.02em; }}
        .brandsub {{ font-size:12.5px; color:var(--av-muted); font-weight:500; }}
        .chiprow {{ display:flex; gap:8px; justify-content:flex-end; flex-wrap:wrap; align-items:center;
          min-height:46px; }}
        .chip {{ display:inline-flex; align-items:center; gap:7px; font-size:12px; font-weight:600;
          color:var(--av-text-2); background:#fff; border:1px solid var(--av-hair);
          padding:7px 11px; border-radius:999px; box-shadow:0 1px 2px rgba(15,23,42,.025); }}
        .chip.on {{ color:#166534; background:#f0fdf4; border-color:#bbf7d0; }}
        .chip.off {{ color:var(--av-muted); background:#f8fafc; }}
        .chip.mode {{ color:#1d4ed8; background:#eff6ff; border-color:#bfdbfe; }}
        .chip.data {{ color:var(--av-muted); font-weight:500; }}
        .sdot {{ display:inline-block; width:8px; height:8px; border-radius:50%; }}
        .sdot.on {{ background:{STATUS['good']}; box-shadow:0 0 0 3px rgba(22,163,74,.12); }}
        .sdot.off {{ background:#94a3b8; }}

        /* Horizontal product navigation: equal width, full available width */
        ul.nav.nav-pills {{
          display:flex !important; width:100% !important; gap:5px !important;
          flex-wrap:nowrap !important; align-items:center !important;
        }}
        ul.nav.nav-pills > li.nav-item {{ flex:1 1 0 !important; min-width:0 !important; }}
        ul.nav.nav-pills .nav-link {{
          width:100% !important; justify-content:center !important; text-align:center !important;
          overflow:hidden !important; text-overflow:ellipsis !important;
        }}

        /* KPI grid */
        .kpi-row {{ display:grid; grid-template-columns:repeat(6,minmax(145px,1fr)); gap:12px;
          margin:12px 0 24px; }}
        .kpi {{ position:relative; overflow:hidden; background:#fff; border:1px solid var(--av-hair);
          border-radius:14px; padding:17px 18px 16px; min-height:108px; box-shadow:var(--av-shadow); }}
        .kpi:before {{ content:""; position:absolute; top:0; left:0; right:0; height:3px;
          background:linear-gradient(90deg,#2563eb,#60a5fa); opacity:.9; }}
        .kpi .v {{ font-size:29px; font-weight:800; color:var(--av-text); line-height:1.12;
          font-variant-numeric:tabular-nums; letter-spacing:-.03em; }}
        .kpi .l {{ font-size:11px; font-weight:700; color:var(--av-muted); margin-top:8px;
          text-transform:uppercase; letter-spacing:.07em; }}
        .kpi .s {{ font-size:11.5px; color:var(--av-text-2); margin-top:3px; }}

        /* Content surfaces */
        .card, .strat, .relbox {{ background:#fff; border:1px solid var(--av-hair); border-radius:14px;
          box-shadow:0 1px 2px rgba(15,23,42,.025); }}
        .card {{ padding:16px 18px; margin-bottom:12px; }}
        .strat {{ padding:15px 17px; margin-bottom:11px; }}
        .strat.primary {{ border-color:#93c5fd; box-shadow:0 0 0 1px #dbeafe, var(--av-shadow); }}
        .strat.weak {{ opacity:.82; border-style:dashed; }}
        .muted {{ color:var(--av-muted); font-size:13px; line-height:1.55; }}
        .caseid {{ font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace; font-size:11.5px;
          font-weight:600; color:#1d4ed8; background:#eff6ff; border:1px solid #dbeafe;
          padding:2px 7px; border-radius:6px; }}
        .pill {{ display:inline-flex; align-items:center; font-size:11px; font-weight:600; padding:3px 8px;
          border-radius:999px; background:#f8fafc; color:#475569; border:1px solid var(--av-hair); margin-right:5px; }}
        .star {{ font-size:11px; font-weight:700; color:#1d4ed8; background:#eff6ff; border:1px solid #bfdbfe;
          padding:3px 8px; border-radius:999px; }}
        .reco {{ font-size:20px; font-weight:750; color:var(--av-text); line-height:1.4;
          background:linear-gradient(90deg,#eff6ff,#ffffff); border:1px solid #dbeafe;
          border-left:4px solid var(--av-primary); border-radius:12px; padding:15px 18px; margin:12px 0 5px; }}
        .reco2 {{ font-size:16px; font-weight:700; color:var(--av-text); margin:9px 0 4px; }}
        .badge {{ display:inline-flex; align-items:center; font-size:12px; font-weight:750; padding:7px 11px;
          border-radius:999px; color:#fff; box-shadow:0 1px 2px rgba(15,23,42,.06); }}
        .lens {{ display:inline-flex; font-size:11px; font-weight:700; padding:5px 9px; border-radius:999px;
          background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; text-transform:uppercase;
          letter-spacing:.045em; margin-left:7px; }}

        .entrow {{ display:flex; flex-wrap:wrap; gap:8px; margin:7px 0 10px; }}
        .ent {{ font-size:13px; padding:6px 10px; border-radius:9px; color:var(--av-text); background:#fff; }}
        .ent .etype {{ font-size:9.5px; color:var(--av-muted); text-transform:uppercase;
          letter-spacing:.06em; margin-left:5px; }}
        .relbox {{ padding:11px 14px; margin-bottom:10px; }}
        .rel {{ font-size:13px; margin:4px 0; color:var(--av-text-2); }}
        .rel .rt {{ color:var(--av-text); font-weight:650; }}
        .rel .rarrow {{ color:#0f766e; font-family:"SFMono-Regular",Consolas,monospace; font-size:11.5px; }}
        .kglegend {{ display:flex; flex-wrap:wrap; gap:18px; margin:8px 0 12px; font-size:12.5px; color:var(--av-text-2); }}
        .kglegend .dot,.kglegend .sq {{ display:inline-block; width:10px; height:10px; margin-right:6px; vertical-align:middle; }}
        .kglegend .dot {{ border-radius:50%; }} .kglegend .sq {{ border-radius:2px; }}

        /* Native Streamlit controls */
        div[data-testid="stMetric"] {{ background:#fff; border:1px solid var(--av-hair); border-radius:14px;
          padding:14px 16px; box-shadow:var(--av-shadow); }}
        div[data-testid="stMetricValue"] {{ color:var(--av-text); font-weight:800; font-size:25px; }}
        div[data-testid="stDataFrame"] {{ background:#fff; border:1px solid var(--av-hair); border-radius:12px;
          overflow:hidden; box-shadow:0 1px 2px rgba(15,23,42,.025); }}
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, textarea {{
          background:#fff !important; border-color:#cbd5e1 !important; color:var(--av-text) !important;
          border-radius:10px !important; }}
        div[data-baseweb="input"]:focus-within > div, div[data-baseweb="select"]:focus-within > div {{
          border-color:#60a5fa !important; box-shadow:0 0 0 3px rgba(37,99,235,.10) !important; }}
        .stButton > button {{ border-radius:10px !important; font-weight:650 !important; border:1px solid #cbd5e1 !important;
          box-shadow:0 1px 2px rgba(15,23,42,.035); }}
        .stButton > button[kind="primary"] {{ background:var(--av-primary) !important; color:white !important;
          border-color:var(--av-primary) !important; }}
        div[data-baseweb="tab-list"] {{ gap:4px; border-bottom:1px solid var(--av-hair); }}
        button[data-baseweb="tab"] {{ color:var(--av-muted) !important; font-weight:650 !important; }}
        button[data-baseweb="tab"][aria-selected="true"] {{ color:var(--av-primary) !important; }}
        details {{ background:#fff !important; border:1px solid var(--av-hair) !important; border-radius:12px !important; }}

        .research-note {{ background:#f8fafc; border:1px solid var(--av-hair); border-left:4px solid #2563eb;
          border-radius:10px; padding:12px 14px; color:#475569; font-size:12.5px; line-height:1.55; margin:10px 0 18px; }}
        .footer-note {{ margin-top:34px; padding-top:16px; border-top:1px solid var(--av-hair); color:var(--av-muted);
          font-size:11.5px; display:flex; gap:10px; justify-content:space-between; flex-wrap:wrap; }}

        @media (max-width:1250px) {{
          .kpi-row {{ grid-template-columns:repeat(3,minmax(150px,1fr)); }}
          ul.nav.nav-pills {{ flex-wrap:wrap !important; }}
          ul.nav.nav-pills > li.nav-item {{ flex:1 1 calc(25% - 5px) !important; }}
        }}
        @media (max-width:820px) {{
          .block-container {{ padding-left:16px !important; padding-right:16px !important; }}
          .kpi-row {{ grid-template-columns:repeat(2,minmax(130px,1fr)); }}
          .chiprow {{ justify-content:flex-start; }}
          ul.nav.nav-pills > li.nav-item {{ flex:1 1 calc(50% - 5px) !important; }}
          .brandsub {{ display:none; }}
        }}
        @media (max-width:520px) {{
          .kpi-row {{ grid-template-columns:1fr; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple[str, str, str]]):
    cells = "".join(
        f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div>'
        f'{f"<div class=s>{s}</div>" if s else ""}</div>'
        for (l, v, s) in items
    )
    st.markdown(f'<div class="kpi-row">{cells}</div>', unsafe_allow_html=True)


def badge_html(kind: str, lens: str | None = None) -> str:
    color, label = BADGE.get(kind, (MUTED, kind))
    lens_html = f'<span class="lens">{lens} lens</span>' if lens and lens != "none" else ""
    return f'<span class="badge" style="background:{color}">{label}</span>{lens_html}'


def section(title: str, subtitle: str = ""):
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(
            f'<div class="muted" style="margin-top:-7px;margin-bottom:12px">{subtitle}</div>',
            unsafe_allow_html=True,
        )


def footer():
    st.markdown(
        '<div class="footer-note"><span><b>AviMaint-DSS</b> · historical maintenance evidence & planning support</span>'
        '<span>Decision support only · verify against current approved maintenance data</span></div>',
        unsafe_allow_html=True,
    )
