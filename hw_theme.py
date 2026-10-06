"""
================================================================================
HEALTHWISE — DESIGN SYSTEM
================================================================================
One place for colour, type and chart chrome, so every surface in the app reads
as a single system in both light and dark mode.

WHY THESE COLOURS
-----------------
The palette is not chosen by eye. Every set below was checked with a
colour-vision validator against the surface it actually renders on, in BOTH
modes, on these gates: lightness band, chroma floor, colour-blind separation
(CVD Delta-E >= 8), normal-vision separation (>= 15) and contrast vs surface.

The previous palette FAILED that check: its mid-blue (#5FA8D3) and teal
(#2A9D8F) sat only 11.3 Delta-E apart in NORMAL vision — below the 15 floor —
so two series were genuinely hard to tell apart even for readers with full
colour vision. Both are replaced here.

  * RISK TIER is an ORDERED scale (Low -> Medium -> High), so it uses a
    single-hue ordinal ramp, light to dark. Validated: monotone lightness,
    step gaps >= 0.06, light end clears 2:1 on the surface, hue spread 3 degrees.
    A traffic-light green/amber/red ramp was rejected: it fails the ordinal
    gates (non-monotone lightness, 117-degree hue spread, amber at 1.83:1) and
    it would spend the reserved status colours on a data series.
  * IDENTITY series (smoker vs non-smoker, model A vs B) use categorical slots
    1-3: blue / orange / aqua. Validated all-pairs in both modes.
  * STATUS colours stay reserved for genuine state — a referral flag, a data
    warning — and never carry a data series. They always ship with an icon and
    a text label, never colour alone.
================================================================================
"""

# --------------------------------------------------------------- PALETTES ---
LIGHT = {
    "surface":      "#FFFFFF",
    "plane":        "#F6F7F9",
    "raised":       "#FFFFFF",
    "ink":          "#14161A",
    "ink_2":        "#4A4F57",
    "muted":        "#7A818B",
    "grid":         "#E7E9EC",
    "axis":         "#C8CCD2",
    "border":       "rgba(20,22,26,0.10)",
    "border_2":     "rgba(20,22,26,0.06)",
    # ordered risk-tier ramp (single hue, validated ordinal)
    "tier":         {"Low": "#86B6EF", "Medium": "#3987E5", "High": "#184F95"},
    # categorical identity slots (validated all-pairs)
    "cat":          ["#2A78D6", "#EB6834", "#1BAF7A"],
    "smoker":       {"no": "#2A78D6", "yes": "#EB6834"},
    # reserved status — never a data series
    "good":         "#0CA30C",
    "warning":      "#FAB219",
    "serious":      "#EC835A",
    "critical":     "#D03B3B",
    "accent_soft":  "#EEF4FD",
    "warn_soft":    "#FFF6E6",
    "crit_soft":    "#FDEDED",
    "good_soft":    "#EAF7EA",
}

DARK = {
    "surface":      "#16181C",
    "plane":        "#0E1013",
    "raised":       "#1C1F24",
    "ink":          "#F2F4F7",
    "ink_2":        "#B8BEC7",
    "muted":        "#8A919B",
    "grid":         "#262A30",
    "axis":         "#383D45",
    "border":       "rgba(255,255,255,0.11)",
    "border_2":     "rgba(255,255,255,0.06)",
    "tier":         {"Low": "#9EC5F4", "Medium": "#5598E7", "High": "#256ABF"},
    "cat":          ["#3987E5", "#D95926", "#199E70"],
    "smoker":       {"no": "#3987E5", "yes": "#D95926"},
    "good":         "#0CA30C",
    "warning":      "#FAB219",
    "serious":      "#EC835A",
    "critical":     "#D03B3B",
    "accent_soft":  "#15233A",
    "warn_soft":    "#2E2617",
    "crit_soft":    "#2E1A1A",
    "good_soft":    "#16291A",
}

TIERS = ["Low", "Medium", "High"]


def palette(dark: bool) -> dict:
    return DARK if dark else LIGHT


# ----------------------------------------------------------- PLOTLY CHROME --
def plotly_layout(P: dict) -> dict:
    """Recessive chrome: hairline grid, no chart-junk, generous padding."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'Inter', -apple-system, BlinkMacSystemFont, "
                         "'Segoe UI', system-ui, sans-serif",
                  size=12.5, color=P["ink_2"]),
        title=dict(font=dict(size=14.5, color=P["ink"], weight=600),
                   x=0, xanchor="left", y=0.97, pad=dict(b=10)),
        margin=dict(l=62, r=26, t=54, b=52),
        hoverlabel=dict(bgcolor=P["raised"], bordercolor=P["border"],
                        font=dict(size=12.5, color=P["ink"],
                                  family="'Inter', system-ui, sans-serif")),
        xaxis=dict(gridcolor=P["grid"], zeroline=False, linecolor=P["axis"],
                   tickfont=dict(color=P["muted"], size=11.5),
                   title=dict(font=dict(color=P["muted"], size=11.5)),
                   showgrid=False, ticks="outside", ticklen=4,
                   tickcolor=P["axis"]),
        yaxis=dict(gridcolor=P["grid"], zeroline=False, linecolor="rgba(0,0,0,0)",
                   tickfont=dict(color=P["muted"], size=11.5),
                   title=dict(font=dict(color=P["muted"], size=11.5)),
                   griddash="solid", ticks=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right",
                    x=1, font=dict(size=11.5, color=P["ink_2"]),
                    bgcolor="rgba(0,0,0,0)"),
        colorway=P["cat"],
        separators=".,",
    )



def _dark_widget_css(P: dict) -> str:
    """Restyle Streamlit's baseweb widgets for the dark surface."""
    return f"""
header[data-testid="stHeader"] {{ background:{P['plane']} !important; }}
header[data-testid="stHeader"] * {{ color:{P['ink_2']} !important; }}

/* text inputs, selects, multiselects */
div[data-baseweb="select"] > div,
div[data-baseweb="input"],
div[data-baseweb="base-input"],
.stTextInput input, .stNumberInput input {{
  background:{P['surface']} !important;
  border-color:{P['border']} !important;
  color:{P['ink']} !important;
}}
div[data-baseweb="select"] svg {{ fill:{P['muted']} !important; }}
div[data-baseweb="select"] div,
.stSelectbox div[role="button"] {{ color:{P['ink']} !important; }}

/* dropdown menus float in a portal, so they need their own rule */
div[data-baseweb="popover"] div,
div[data-baseweb="menu"], ul[role="listbox"] {{
  background:{P['raised']} !important; color:{P['ink']} !important;
}}
ul[role="listbox"] li:hover {{ background:{P['border_2']} !important; }}

/* multiselect tags */
span[data-baseweb="tag"] {{
  background:{P['cat'][0]} !important; color:#fff !important;
  border:none !important;
}}
span[data-baseweb="tag"] svg {{ fill:#fff !important; }}

/* file uploader, expanders, tables */
section[data-testid="stFileUploaderDropzone"] {{
  background:{P['surface']} !important; border-color:{P['border']} !important;
}}
section[data-testid="stFileUploaderDropzone"] * {{ color:{P['ink_2']} !important; }}
div[data-testid="stExpander"] details {{
  background:{P['surface']} !important; border-color:{P['border_2']} !important;
}}
div[data-testid="stExpander"] summary {{ color:{P['ink_2']} !important; }}
div[data-testid="stDataFrame"], div[data-testid="stDataFrame"] * {{
  color:{P['ink']} !important;
}}
div[data-testid="stDataFrame"] {{ background:{P['surface']} !important; }}

/* buttons */
.stDownloadButton button, .stButton button {{
  background:{P['surface']} !important; color:{P['ink']} !important;
  border-color:{P['border']} !important;
}}
.stSlider [data-baseweb="slider"] div[role="slider"] {{
  border-color:{P['surface']} !important;
}}
.stCaption, [data-testid="stCaptionContainer"] {{ color:{P['muted']} !important; }}
"""

# ------------------------------------------------------------------- CSS ----
def css(P: dict, dark: bool) -> str:
    dark_widgets = _dark_widget_css(P) if dark else ''
    shadow = ("0 1px 2px rgba(0,0,0,.35), 0 8px 24px -12px rgba(0,0,0,.55)"
              if dark else
              "0 1px 2px rgba(20,22,26,.04), 0 8px 24px -14px rgba(20,22,26,.16)")
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
  --surface:{P['surface']}; --plane:{P['plane']}; --raised:{P['raised']};
  --ink:{P['ink']}; --ink2:{P['ink_2']}; --muted:{P['muted']};
  --border:{P['border']}; --border2:{P['border_2']}; --grid:{P['grid']};
  --accent:{P['cat'][0]}; --accent-soft:{P['accent_soft']};
  --good:{P['good']}; --warning:{P['warning']}; --critical:{P['critical']};
  --warn-soft:{P['warn_soft']}; --crit-soft:{P['crit_soft']}; --good-soft:{P['good_soft']};
  --radius:12px; --shadow:{shadow};
}}

html, body, [class*="css"], .stApp {{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  font-feature-settings:'cv02','cv03','cv04','ss01';
}}
.stApp {{ background:var(--plane); color:var(--ink); }}
.main .block-container {{ padding:1.4rem 2.2rem 3rem; max-width:1480px; }}
h1,h2,h3,h4 {{ color:var(--ink); letter-spacing:-0.017em; font-weight:650; }}
p, li, span, label {{ color:var(--ink2); }}
hr {{ border-color:var(--border2); }}
a {{ color:var(--accent); }}

/* ---------- masthead ---------- */
.hw-top {{
  display:flex; align-items:center; justify-content:space-between;
  gap:1rem; padding:.55rem .2rem 1rem; border-bottom:1px solid var(--border2);
  margin-bottom:1.15rem; flex-wrap:wrap;
}}
.hw-brand {{ display:flex; align-items:center; gap:.7rem; }}
.hw-mark {{
  width:34px; height:34px; border-radius:9px; flex:0 0 34px;
  background:linear-gradient(145deg,{P['cat'][0]},{P['tier']['High']});
  display:grid; place-items:center; color:#fff; font-weight:700; font-size:15px;
  letter-spacing:-.5px;
}}
.hw-name {{ font-size:1.02rem; font-weight:650; color:var(--ink); line-height:1.2; }}
.hw-sub  {{ font-size:.73rem; color:var(--muted); line-height:1.3; }}
.hw-meta {{ display:flex; gap:1.5rem; align-items:center; flex-wrap:wrap; }}
.hw-meta-item .k {{
  font-size:.635rem; text-transform:uppercase; letter-spacing:.075em;
  color:var(--muted); font-weight:600; display:block;
}}
.hw-meta-item .v {{ font-size:.83rem; color:var(--ink); font-weight:600;
  font-variant-numeric:tabular-nums; }}

/* ---------- KPI ---------- */
.kpi-row {{ display:grid; gap:.85rem; margin-bottom:.4rem; }}
.kpi {{
  background:var(--raised); border:1px solid var(--border2);
  border-radius:var(--radius); padding:.95rem 1.05rem; box-shadow:var(--shadow);
  position:relative; overflow:hidden; height:100%;
}}
.kpi::before {{
  content:''; position:absolute; left:0; top:0; bottom:0; width:3px;
  background:var(--accent); opacity:.85;
}}
.kpi.is-good::before    {{ background:var(--good); }}
.kpi.is-warn::before    {{ background:var(--warning); }}
.kpi.is-crit::before    {{ background:var(--critical); }}
.kpi.is-neutral::before {{ background:var(--muted); opacity:.45; }}
.kpi .k {{
  font-size:.665rem; text-transform:uppercase; letter-spacing:.075em;
  color:var(--muted); font-weight:600; margin-bottom:.3rem;
}}
.kpi .v {{
  font-size:1.62rem; font-weight:680; color:var(--ink); line-height:1.15;
  letter-spacing:-.024em;
}}
.kpi .s {{ font-size:.735rem; color:var(--muted); margin-top:.2rem; line-height:1.35; }}

/* ---------- cards & notes ---------- */
.card {{
  background:var(--raised); border:1px solid var(--border2);
  border-radius:var(--radius); padding:1.05rem 1.15rem .85rem;
  box-shadow:var(--shadow); margin-bottom:.9rem;
}}
.card-h {{ font-size:.95rem; font-weight:640; color:var(--ink); margin-bottom:.1rem; }}
.card-s {{ font-size:.775rem; color:var(--muted); margin-bottom:.55rem; }}

.note {{
  border-left:3px solid var(--accent); background:var(--accent-soft);
  padding:.7rem .9rem; border-radius:0 8px 8px 0; font-size:.855rem;
  color:var(--ink2); margin:.55rem 0 .9rem; line-height:1.52;
}}
.note.warn {{ border-left-color:var(--warning); background:var(--warn-soft); }}
.note.crit {{ border-left-color:var(--critical); background:var(--crit-soft); }}
.note.good {{ border-left-color:var(--good);    background:var(--good-soft); }}
.note b {{ color:var(--ink); font-weight:620; }}

.pill {{
  display:inline-flex; align-items:center; gap:.34rem; padding:.2rem .6rem;
  border-radius:999px; font-size:.7rem; font-weight:620; line-height:1.5;
  border:1px solid var(--border);
}}
.pill.good {{ background:var(--good-soft); color:var(--good); border-color:var(--good); }}
.pill.warn {{ background:var(--warn-soft); color:{'#FAB219' if dark else '#8A6200'};
              border-color:var(--warning); }}
.pill.crit {{ background:var(--crit-soft); color:var(--critical); border-color:var(--critical); }}
.pill.mut  {{ background:transparent; color:var(--muted); }}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{
  gap:.15rem; border-bottom:1px solid var(--border2); padding-bottom:0;
}}
.stTabs [data-baseweb="tab"] {{
  background:transparent; border:none; border-radius:8px 8px 0 0;
  padding:.5rem .85rem; font-size:.83rem; font-weight:560; color:var(--muted);
}}
.stTabs [data-baseweb="tab"]:hover {{ color:var(--ink); background:var(--border2); }}
.stTabs [aria-selected="true"] {{
  color:var(--ink) !important; font-weight:650;
  box-shadow:inset 0 -2px 0 0 var(--accent);
}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none; }}

/* ---------- sidebar & controls ---------- */
section[data-testid="stSidebar"] {{
  background:var(--surface); border-right:1px solid var(--border2);
}}
section[data-testid="stSidebar"] .block-container {{ padding-top:1.4rem; }}
.side-h {{
  font-size:.665rem; text-transform:uppercase; letter-spacing:.08em;
  color:var(--muted); font-weight:640; margin:1.1rem 0 .3rem;
}}
.stSlider label, .stSelectbox label, .stMultiSelect label, .stRadio label {{
  font-size:.79rem !important; font-weight:560 !important; color:var(--ink2) !important;
}}
div[data-testid="stExpander"] {{
  border:1px solid var(--border2); border-radius:9px; background:var(--surface);
}}
div[data-testid="stExpander"] summary {{ font-size:.79rem; color:var(--muted); font-weight:560; }}
.stDownloadButton button, .stButton button {{
  border-radius:8px; border:1px solid var(--border); font-size:.8rem;
  font-weight:580; color:var(--ink); background:var(--surface);
}}
.stDownloadButton button:hover, .stButton button:hover {{
  border-color:var(--accent); color:var(--accent);
}}
/* keyboard focus must stay visible */
*:focus-visible {{ outline:2px solid var(--accent) !important; outline-offset:2px; }}

.foot {{
  font-size:.735rem; color:var(--muted); line-height:1.6;
  border-top:1px solid var(--border2); padding-top:.9rem; margin-top:1.6rem;
}}
#MainMenu, footer {{ visibility:hidden; }}

/* --------------------------------------------------------------------------
   Dark-mode overrides for Streamlit's own widgets.
   Streamlit's config.toml theme is static, so a runtime toggle has to restyle
   the framework's internals directly or the controls stay light against a dark
   page. Every value below comes from the same validated dark palette.
   -------------------------------------------------------------------------- */
{dark_widgets}
</style>
"""
