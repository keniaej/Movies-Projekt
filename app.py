"""
🎬 The Evolution of Cinema – BI Dashboard
Run: python app.py  →  open http://127.0.0.1:8050
"""

import gzip
import json
import sys
import traceback
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data…")

# Paths relative to this file, so the app works from any working directory
ROOT       = Path(__file__).resolve().parent
DATA_DIR   = ROOT / "data"
MODEL_DIR  = ROOT / "models"
META_PATH  = MODEL_DIR / "model_meta.json"

_required = [DATA_DIR / "movies_full.csv", DATA_DIR / "genre_trends.csv",
             DATA_DIR / "predictions_2026.csv", META_PATH]
_missing = [p.relative_to(ROOT).as_posix() for p in _required if not p.exists()]
if _missing:
    sys.exit("❌ Missing files: " + ", ".join(_missing) +
             "\n   Run notebooks/01_data_pipeline.ipynb and "
             "notebooks/02_revenue_model.ipynb first (see README).")

full_df      = pd.read_csv(DATA_DIR / "movies_full.csv").reset_index(drop=True)
genre_trends = pd.read_csv(DATA_DIR / "genre_trends.csv").reset_index(drop=True)
pred_df      = pd.read_csv(DATA_DIR / "predictions_2026.csv").reset_index(drop=True)

# Model: XGBoost is stored in its own version-independent JSON format;
# scikit-learn models (if one of them wins a re-run) as a gzipped pickle.
META = json.loads(META_PATH.read_text())
MODEL_PATH = MODEL_DIR / META["model_file"]
if MODEL_PATH.suffix == ".json":
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
else:
    with gzip.open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

MODEL_NAME   = META["model_name"]
FEATURE_COLS = META["feature_cols"]
KNOWN_GENRES = META["genres"]                 # list index = genre_encoded
GENRE_CODE   = {g: i for i, g in enumerate(KNOWN_GENRES)}
MEDIAN_RT    = float(META["median_runtime"])
INTERVAL_LOG = tuple(META["interval_log"])    # 80% prediction interval (log scale)

for col in ["budget_adj","revenue_adj","roi","runtime",
            "vote_average","vote_count","release_year","release_month"]:
    if col in full_df.columns:
        full_df[col] = pd.to_numeric(full_df[col], errors="coerce")

full_df["is_oscar_winner"] = full_df["is_oscar_winner"].fillna(False).astype(bool)
full_df["has_financials"]  = full_df["has_financials"].fillna(False).astype(bool)
full_df["primary_genre"]   = full_df["primary_genre"].fillna("Unknown")
full_df["primary_country"] = full_df["primary_country"].fillna("Unknown")
full_df["decade"]          = full_df["decade"].fillna("Unknown")

COUNTRY_FIX = {
    "United States of America": "United States",
    "Korea": "South Korea",
    "Czech Republic": "Czechia",
}
full_df["country_clean"] = full_df["primary_country"].replace(COUNTRY_FIX)
fin_df = full_df[full_df["has_financials"]].copy().reset_index(drop=True)
genre_trends["decade_year"] = pd.to_numeric(
    genre_trends["release_year"], errors="coerce"
)
TOP_GENRES = full_df["primary_genre"].value_counts().head(12).index.tolist()

print(f"✅ {len(full_df):,} movies  |  {len(fin_df):,} with financials")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────
DARK = {
    "bg":      "#0a0b10",
    "card":    "#12141c",
    "card2":   "#1a1d28",
    "card3":   "#20243a",
    "border":  "#2a2d3e",
    "text":    "#f0f0f0",
    "subtext": "#8087a8",
    "accent":  "#f5c518",
    "accent2": "#e07b39",
    "green":   "#2ecc71",
    "red":     "#e74c3c",
    "blue":    "#3b9edd",
    "grid":    "rgba(255,255,255,0.04)",
    "land":    "#1a1d28",
    "lake":    "#0a0b10",
    "coast":   "rgba(255,255,255,0.10)",
}
LIGHT = {
    "bg":      "#f2efe8",
    "card":    "#ffffff",
    "card2":   "#eeebe3",
    "card3":   "#e4e0d6",
    "border":  "#d0ccbf",
    "text":    "#16192b",
    "subtext": "#6b6f85",
    "accent":  "#b89200",
    "accent2": "#c04a20",
    "green":   "#27ae60",
    "red":     "#c0392b",
    "blue":    "#2980b9",
    "grid":    "rgba(0,0,0,0.05)",
    "land":    "#e4e0d6",
    "lake":    "#f2efe8",
    "coast":   "rgba(0,0,0,0.15)",
}

# 12 distinct, readable chart colors
CHART_COLORS = [
    "#f5c518",  # gold
    "#e07b39",  # orange
    "#3b9edd",  # blue
    "#2ecc71",  # green
    "#e74c3c",  # red
    "#9b59b6",  # purple
    "#1abc9c",  # teal
    "#e67e22",  # amber
    "#27ae60",  # dark green
    "#2980b9",  # dark blue
    "#d35400",  # burnt orange
    "#8e44ad",  # dark purple
]

GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap"
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRANSLATIONS
# ─────────────────────────────────────────────────────────────────────────────
T = {
    "en": {
        "title":           "THE EVOLUTION OF CINEMA",
        "subtitle":        "A Business Intelligence Deep-Dive · Film History 1920–2026",
        "total_movies":    "Total Movies",
        "box_office":      "Total Box Office",
        "med_budget":      "Median Budget",
        "top_genre":       "Top Genre",
        "oscars":          "Oscar Winners",
        "countries":       "Countries",
        "since":           "1920 – 2025",
        "infl_adj":        "inflation-adjusted",
        "by_volume":       "by volume",
        "best_pic":        "Best Picture",
        "producing":       "producing films",
        "decade_range":    "Decade Range",
        "genres":          "Genres",
        "tab_genre":       "📈 Genre Trends",
        "tab_map":         "🌍 World Map",
        "tab_scatter":     "💰 Budget vs Revenue",
        "tab_oscar":       "🏆 Oscar Winners",
        "tab_revenue":     "📊 Revenue by Decade",
        "tab_predict":     "🔮 2026 Predictions",
        "sub_genre":       "Line per genre — each line shows how many films of that genre were released each decade. Hover to compare.",
        "sub_map":         "Total films produced per country — hover for details. Darker = more films.",
        "sub_scatter":     "Each dot is one film. Above the dashed line = profitable. ⭐ = Oscar Best Picture winner.",
        "sub_oscar":       "Gold bars = inflation-adjusted revenue. Red line = audience rating on right axis. Hover a bar to see the film title.",
        "sub_revenue":     "Four views of the same story: how revenue, volume, ROI, and growth evolved decade by decade.",
        "sub_predict":     "{model} predictions for 2026 blockbusters. Error bars = 80% prediction interval, based on the model's errors on unseen test films. Use the tool below to predict any film.",
        "predictor_title": "🔮 Live Revenue Predictor",
        "predictor_sub":   "Enter a budget, genre, and release month — the model predicts worldwide box office revenue instantly.",
        "budget_label":    "Budget ($M)",
        "genre_label":     "Genre",
        "month_label":     "Release Month",
        "runtime_label":   "Runtime (min)",
        "predict_btn":     "▶  Predict Revenue",
        "pred_revenue":    "PREDICTED REVENUE",
        "est_roi":         "ESTIMATED ROI",
        "inputs":          "INPUTS",
        "range_80":        "80% prediction interval",
        "summer":          "☀️ Summer release",
        "holiday":         "🎄 Holiday release",
        "standard":        "📅 Standard release",
        "break_even":      "Break-even line",
        "chart_genre":     "Genre Popularity Over the Decades",
        "chart_map":       "Worldwide Film Production by Country",
        "chart_scatter":   "Production Budget vs Worldwide Revenue",
        "chart_oscar":     "Oscar Best Picture Winners — Revenue & Ratings",
        "chart_rev":       "Box Office Trends by Decade",
        "chart_pred":      "2026 Blockbuster Revenue Predictions",
        "panel_rev_bud":   "Median Revenue vs Budget ($M)",
        "panel_count":     "Films with Financial Data",
        "panel_roi":       "Median ROI (%)",
        "panel_index":     "Revenue Growth Index",
        "months":          ["Jan","Feb","Mar","Apr","May","Jun",
                            "Jul","Aug","Sep","Oct","Nov","Dec"],
        "revenue_label":   "Revenue ($M adj.)",
        "rating_label":    "Avg Rating / 10",
        "budget_x":        "Production Budget — inflation adjusted ($)",
        "revenue_y":       "Worldwide Revenue — inflation adjusted ($)",
        "decade_x":        "Decade",
        "films_y":         "Number of Films Released",
        "country_hover":   "Films",
        "avg_rev_hover":   "Avg Revenue ($M)",
        "no_data":         "No data available for the selected filters. Try adjusting the decade range or genres.",
        "error_msg":       "Chart error:\n",
        "fill_fields":     "Please fill in all fields.",
        "note_2020s":      "⚠️ Note: 2020s data is sparse — COVID disrupted releases and the TMDb API returned fewer records for recent years.",
    },
    "de": {
        "title":           "DIE EVOLUTION DES KINOS",
        "subtitle":        "Business Intelligence Analyse · Filmgeschichte 1920–2026",
        "total_movies":    "Filme gesamt",
        "box_office":      "Gesamteinnahmen",
        "med_budget":      "Median Budget",
        "top_genre":       "Top Genre",
        "oscars":          "Oscar-Gewinner",
        "countries":       "Länder",
        "since":           "1920 – 2025",
        "infl_adj":        "inflationsbereinigt",
        "by_volume":       "nach Anzahl",
        "best_pic":        "Bester Film",
        "producing":       "produzierende Länder",
        "decade_range":    "Jahrzehnt-Bereich",
        "genres":          "Genres",
        "tab_genre":       "📈 Genre-Trends",
        "tab_map":         "🌍 Weltkarte",
        "tab_scatter":     "💰 Budget vs. Einnahmen",
        "tab_oscar":       "🏆 Oscar-Gewinner",
        "tab_revenue":     "📊 Einnahmen pro Jahrzehnt",
        "tab_predict":     "🔮 Prognosen 2026",
        "sub_genre":       "Eine Linie pro Genre — zeigt wie viele Filme pro Genre pro Jahrzehnt erschienen sind. Hover zum Vergleichen.",
        "sub_map":         "Produzierte Filme pro Land — hover für Details. Dunkler = mehr Filme.",
        "sub_scatter":     "Jeder Punkt ist ein Film. Über der Linie = profitabel. ⭐ = Oscar Bester Film.",
        "sub_oscar":       "Goldene Balken = inflationsbereinigte Einnahmen. Rote Linie = Bewertung (rechte Achse). Hover für Filmtitel.",
        "sub_revenue":     "Vier Perspektiven: Einnahmen, Volumen, ROI und Wachstumsindex — Jahrzehnt für Jahrzehnt.",
        "sub_predict":     "{model}-Prognosen für 2026-Blockbuster. Fehlerbalken = 80% Prognoseintervall, berechnet aus den Fehlern des Modells auf ungesehenen Testfilmen.",
        "predictor_title": "🔮 Live-Einnahmenprognose",
        "predictor_sub":   "Budget, Genre und Erscheinungsmonat eingeben — das Modell prognostiziert sofort die weltweiten Einnahmen.",
        "budget_label":    "Budget (Mio. $)",
        "genre_label":     "Genre",
        "month_label":     "Erscheinungsmonat",
        "runtime_label":   "Laufzeit (Min.)",
        "predict_btn":     "▶  Einnahmen prognostizieren",
        "pred_revenue":    "PROGNOSTIZIERTE EINNAHMEN",
        "est_roi":         "GESCHÄTZTER ROI",
        "inputs":          "EINGABEN",
        "range_80":        "80% Prognoseintervall",
        "summer":          "☀️ Sommerveröffentlichung",
        "holiday":         "🎄 Feiertagsveröffentlichung",
        "standard":        "📅 Standardveröffentlichung",
        "break_even":      "Break-Even-Linie",
        "chart_genre":     "Genre-Popularität über die Jahrzehnte",
        "chart_map":       "Weltweite Filmproduktion nach Land",
        "chart_scatter":   "Produktionsbudget vs. weltweite Einnahmen",
        "chart_oscar":     "Oscar-Gewinner (Bester Film) — Einnahmen & Bewertungen",
        "chart_rev":       "Kasseneinnahmen nach Jahrzehnt",
        "chart_pred":      "Blockbuster-Prognosen 2026",
        "panel_rev_bud":   "Median Einnahmen vs. Budget (Mio. $)",
        "panel_count":     "Filme mit Finanzdaten",
        "panel_roi":       "Median ROI (%)",
        "panel_index":     "Einnahmen-Wachstumsindex",
        "months":          ["Jan","Feb","Mär","Apr","Mai","Jun",
                            "Jul","Aug","Sep","Okt","Nov","Dez"],
        "revenue_label":   "Einnahmen (Mio. $ bereinigt)",
        "rating_label":    "Ø Bewertung / 10",
        "budget_x":        "Produktionsbudget — inflationsbereinigt ($)",
        "revenue_y":       "Weltweite Einnahmen — inflationsbereinigt ($)",
        "decade_x":        "Jahrzehnt",
        "films_y":         "Anzahl veröffentlichter Filme",
        "country_hover":   "Filme",
        "avg_rev_hover":   "Ø Einnahmen (Mio. $)",
        "no_data":         "Keine Daten für die gewählten Filter. Jahrzehnt-Bereich oder Genres anpassen.",
        "error_msg":       "Diagramm-Fehler:\n",
        "fill_fields":     "Bitte alle Felder ausfüllen.",
        "note_2020s":      "⚠️ Hinweis: 2020er-Daten sind lückenhaft — COVID störte Veröffentlichungen und die TMDb API lieferte weniger Einträge für aktuelle Jahre.",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. APP INIT
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, GOOGLE_FONTS],
    suppress_callback_exceptions=True,
    title="🎬 Cinema BI",
)
server = app.server

# ─────────────────────────────────────────────────────────────────────────────
# 5. HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_th(theme):
    return DARK if theme == "dark" else LIGHT

def plot_layout(th, title_text="", height=None, **kwargs):
    """Minimal base layout — no xaxis/yaxis/legend/margin to avoid duplicate key errors."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=th["text"], size=12),
        hoverlabel=dict(
            bgcolor=th["card3"],
            font_color=th["text"],
            font_size=12,
            bordercolor=th["accent"],
        ),
    )
    if title_text:
        base["title"] = dict(
            text=title_text,
            font=dict(size=16, color=th["text"],
                      family="'Bebas Neue', sans-serif"),
            x=0, xanchor="left",
            pad=dict(l=10),
        )
    if height:
        base["height"] = height
    base.update(kwargs)
    return base

def error_card(msg, th):
    return html.Div([
        html.Div("⚠️", style={"fontSize":"2.5rem","marginBottom":"10px"}),
        html.Pre(msg, style={"color":th["subtext"],"fontSize":"0.75rem",
                             "whiteSpace":"pre-wrap","textAlign":"left",
                             "maxHeight":"200px","overflowY":"auto",
                             "background":th["card2"],"padding":"12px",
                             "borderRadius":"8px"}),
    ], style={"padding":"40px","textAlign":"center"})

def no_data_card(msg, th):
    return html.Div([
        html.Div("🔍", style={"fontSize":"2rem","marginBottom":"8px"}),
        html.Div(msg, style={"color":th["subtext"],"fontSize":"0.9rem"}),
    ], style={"padding":"60px","textAlign":"center"})

def insight_box(text, th):
    """Italic subtitle under chart title."""
    return html.P(text, style={
        "color": th["subtext"],
        "fontSize": "0.83rem",
        "marginBottom": "12px",
        "fontStyle": "italic",
        "lineHeight": "1.5",
        "borderLeft": f"3px solid {th['accent']}",
        "paddingLeft": "10px",
    })

def warning_box(text, th):
    return html.Div(text, style={
        "color": th["accent2"],
        "fontSize": "0.78rem",
        "background": f"rgba(224,123,57,0.10)",
        "border": f"1px solid {th['accent2']}",
        "borderRadius": "8px",
        "padding": "8px 14px",
        "marginTop": "10px",
        "marginBottom": "4px",
    })

def cs(th):
    return {
        "background":   th["card"],
        "borderRadius": "0 12px 12px 12px",
        "padding":      "26px 28px",
        "border":       f"1px solid {th['border']}",
        "borderTop":    f"2px solid {th['accent']}",
        "minHeight":    "520px",
    }

def kpi_card(icon, label, value, sub, th):
    return dbc.Col(html.Div([
        html.Div(icon, style={"fontSize":"1.4rem","marginBottom":"6px","lineHeight":"1"}),
        html.Div(value, style={
            "color":       th["accent"],
            "fontWeight":  "700",
            "fontSize":    "1.55rem",
            "fontFamily":  "'Bebas Neue', sans-serif",
            "letterSpacing":"1px",
            "lineHeight":  "1.1",
        }),
        html.Div(label, style={
            "color":         th["text"],
            "fontSize":      "0.68rem",
            "textTransform": "uppercase",
            "letterSpacing": "1.5px",
            "fontWeight":    "600",
            "marginTop":     "3px",
        }),
        html.Div(sub, style={"color":th["subtext"],"fontSize":"0.66rem","marginTop":"1px"}),
    ], style={
        "background":   th["card"],
        "border":       f"1px solid {th['border']}",
        "borderTop":    f"3px solid {th['accent']}",
        "borderRadius": "12px",
        "padding":      "16px 18px",
        "transition":   "transform 0.15s",
    }),
    xs=6, sm=4, lg=2)

# ─────────────────────────────────────────────────────────────────────────────
# 6. LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
app.layout = html.Div(id="root-div", style={"minHeight":"100vh"}, children=[
    dcc.Store(id="theme-store", data="dark"),
    dcc.Store(id="lang-store",  data="en"),
    html.Div(id="header-div"),
    dbc.Container(fluid=True, style={"padding":"22px 36px 14px"},
                  children=[dbc.Row(id="kpi-row", className="g-3")]),
    dbc.Container(fluid=True, style={"padding":"0 36px 16px"},
                  children=[html.Div(id="filter-div")]),
    dbc.Container(fluid=True, style={"padding":"0 36px 52px"}, children=[
        html.Div(id="tabs-shell"),
        html.Div(id="tab-content"),
    ]),
])

# ─────────────────────────────────────────────────────────────────────────────
# 7. SHELL CALLBACK
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("root-div",   "style"),
    Output("header-div", "children"),
    Output("kpi-row",    "children"),
    Output("filter-div", "children"),
    Output("tabs-shell", "children"),
    Input("theme-store", "data"),
    Input("lang-store",  "data"),
)
def render_shell(theme, lang):
    th = get_th(theme)
    t  = T[lang]

    root_style = {
        "minHeight":    "100vh",
        "backgroundColor": th["bg"],
        "color":        th["text"],
        "fontFamily":   "DM Sans, sans-serif",
    }

    btn_theme = dbc.Button(
        "☀️ Light" if theme=="dark" else "🌙 Dark",
        id="theme-btn", size="sm",
        style={"backgroundColor":th["card2"],"color":th["text"],
               "border":f"1px solid {th['border']}","borderRadius":"20px",
               "marginRight":"8px","fontSize":"0.77rem","cursor":"pointer",
               "padding":"5px 14px"}
    )
    btn_lang = dbc.Button(
        "🇩🇪 DE" if lang=="en" else "🇬🇧 EN",
        id="lang-btn", size="sm",
        style={"backgroundColor":th["card2"],"color":th["text"],
               "border":f"1px solid {th['border']}","borderRadius":"20px",
               "fontSize":"0.77rem","cursor":"pointer","padding":"5px 14px"}
    )

    header = html.Div([
        html.Div([
            html.Span("🎬", style={"fontSize":"2.2rem","marginRight":"14px",
                                   "verticalAlign":"middle"}),
            html.Div([
                html.H1(t["title"], style={
                    "color":       th["accent"],
                    "fontFamily":  "'Bebas Neue', sans-serif",
                    "fontSize":    "1.9rem",
                    "letterSpacing":"4px",
                    "marginBottom":"1px",
                    "lineHeight":  "1",
                }),
                html.P(t["subtitle"], style={
                    "color":         th["subtext"],
                    "marginBottom":  "0",
                    "fontSize":      "0.79rem",
                    "letterSpacing": "0.6px",
                }),
            ], style={"display":"inline-block","verticalAlign":"middle"}),
        ]),
        html.Div([btn_theme, btn_lang],
                 style={"display":"flex","alignItems":"center"}),
    ], style={
        "background":    f"linear-gradient(135deg, {th['card']} 0%, {th['bg']} 100%)",
        "padding":       "18px 36px",
        "borderBottom":  f"2px solid {th['accent']}",
        "display":       "flex",
        "justifyContent":"space-between",
        "alignItems":    "center",
    })

    # KPIs
    kpis = [
        kpi_card("🎞️", t["total_movies"], f"{len(full_df):,}",        t["since"],     th),
        kpi_card("💵", t["box_office"],   f"${fin_df['revenue_adj'].sum()/1e9:.1f}B", t["infl_adj"], th),
        kpi_card("🎬", t["med_budget"],   f"${fin_df['budget_adj'].median()/1e6:.1f}M", t["infl_adj"], th),
        kpi_card("🎭", t["top_genre"],    full_df["primary_genre"].value_counts().index[0], t["by_volume"], th),
        kpi_card("🌍", t["countries"],    str(full_df["primary_country"].nunique()),  t["producing"], th),
    ]

    # Filter bar
    filter_bar = html.Div([
        dbc.Row([
            dbc.Col([
                html.Label(f"📅  {t['decade_range'].upper()}",
                           style={"color":th["subtext"],"fontSize":"0.68rem",
                                  "letterSpacing":"2px","fontWeight":"700",
                                  "marginBottom":"12px","display":"block"}),
                dcc.RangeSlider(
                    id="decade-slider", min=1920, max=2025, step=5,
                    marks={y:{"label":str(y),
                              "style":{"color":th["subtext"],"fontSize":"10px"}}
                           for y in range(1920, 2030, 10)},
                    value=[1960, 2025],
                    tooltip={"placement":"bottom","always_visible":False},
                ),
            ], md=7),
            dbc.Col([
                html.Label(f"🎭  {t['genres'].upper()}",
                           style={"color":th["subtext"],"fontSize":"0.68rem",
                                  "letterSpacing":"2px","fontWeight":"700",
                                  "marginBottom":"8px","display":"block"}),
                dcc.Dropdown(
                    id="genre-filter",
                    options=[{"label":g,"value":g} for g in sorted(TOP_GENRES)],
                    value=TOP_GENRES[:6], multi=True,
                ),
            ], md=5),
        ], align="center"),
    ], style={
        "background":   th["card"],
        "border":       f"1px solid {th['border']}",
        "borderRadius": "12px",
        "padding":      "18px 24px",
    })

    # Tab bar
    ts = {
        "backgroundColor": "transparent",
        "color":           th["subtext"],
        "border":          "none",
        "borderBottom":    "2px solid transparent",
        "padding":         "11px 22px",
        "fontFamily":      "DM Sans, sans-serif",
        "fontWeight":      "600",
        "fontSize":        "0.82rem",
        "letterSpacing":   "0.3px",
    }
    ts_sel = {**ts,
              "color":       th["accent"],
              "borderBottom":f"2px solid {th['accent']}",
              "backgroundColor": th["card"]}

    tabs = dcc.Tabs(id="main-tabs", value="tab-genre", children=[
        dcc.Tab(label=t["tab_genre"],   value="tab-genre",   style=ts, selected_style=ts_sel),
        dcc.Tab(label=t["tab_map"],     value="tab-map",     style=ts, selected_style=ts_sel),
        dcc.Tab(label=t["tab_scatter"], value="tab-scatter", style=ts, selected_style=ts_sel),

        dcc.Tab(label=t["tab_revenue"], value="tab-revenue", style=ts, selected_style=ts_sel),
        dcc.Tab(label=t["tab_predict"], value="tab-predict", style=ts, selected_style=ts_sel),
    ], colors={"border":"transparent","primary":th["accent"],"background":"transparent"})

    return root_style, header, kpis, filter_bar, tabs


@app.callback(
    Output("theme-store","data"),
    Input("theme-btn","n_clicks"),
    State("theme-store","data"),
    prevent_initial_call=True,
)
def toggle_theme(n, cur):
    return "light" if cur=="dark" else "dark"


@app.callback(
    Output("lang-store","data"),
    Input("lang-btn","n_clicks"),
    State("lang-store","data"),
    prevent_initial_call=True,
)
def toggle_lang(n, cur):
    return "de" if cur=="en" else "en"


# ─────────────────────────────────────────────────────────────────────────────
# 8. TAB CONTENT
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("tab-content","children"),
    Input("main-tabs",    "value"),
    Input("decade-slider","value"),
    Input("genre-filter", "value"),
    Input("theme-store",  "data"),
    Input("lang-store",   "data"),
)
def render_tab(tab, decade_range, genres, theme, lang):
    th     = get_th(theme)
    t      = T[lang]
    d_min  = int(decade_range[0])
    d_max  = int(decade_range[1])
    genres = genres or TOP_GENRES[:6]
    card   = cs(th)

    df_fin = fin_df[
        (fin_df["release_year"] >= d_min) &
        (fin_df["release_year"] <= d_max) &
        (fin_df["primary_genre"].isin(genres))
    ].copy().reset_index(drop=True)

    # ── TAB 1 : Genre Trends — two separate charts (Kaggle vs TMDb) ─────────
    if tab == "tab-genre":
        try:
            KAGGLE_END = 2017
            TMDB_START = 2018

            def make_genre_fig(df_src, title_suffix, year_min, year_max, source_label):
                """Build a % share line chart for one data source."""
                gt_src = df_src[
                    df_src["primary_genre"].isin(genres) &
                    (df_src["decade_year"] >= year_min) &
                    (df_src["decade_year"] <= year_max)
                ].copy().reset_index(drop=True)

                if gt_src.empty:
                    return None, True  # empty flag

                total = (
                    df_src[
                        (df_src["decade_year"] >= year_min) &
                        (df_src["decade_year"] <= year_max)
                    ]
                    .groupby("release_year")["count"]
                    .sum()
                    .reset_index()
                    .rename(columns={"count":"total"})
                )
                gt_src = gt_src.merge(total, on="release_year", how="left")
                gt_src["pct"] = (gt_src["count"] / gt_src["total"] * 100).round(2)

                pivot = (
                    gt_src.pivot_table(index="release_year", columns="primary_genre",
                                       values="pct", fill_value=0)
                          .reset_index()
                )
                genre_cols = [c for c in pivot.columns if c != "release_year"]
                fig = go.Figure()
                for i, g in enumerate(genre_cols):
                    col = CHART_COLORS[i % len(CHART_COLORS)]
                    fig.add_trace(go.Scatter(
                        x=pivot["release_year"],
                        y=pivot[g],
                        name=g, mode="lines+markers",
                        line=dict(color=col, width=2.2),
                        marker=dict(color=col, size=5,
                                    line=dict(color=th["card"], width=1)),
                        hovertemplate=f"<b>{g}</b><br>%{{x}}: %{{y:.1f}}%<extra>{source_label}</extra>",
                    ))
                y_title = "Anteil am Jahrgang (%)" if lang=="de" else "Share of yearly releases (%)"
                fig.update_layout(
                    **plot_layout(th, title_text=title_suffix, height=380),
                    xaxis=dict(
                        title=t["decade_x"],
                        gridcolor=th["grid"],
                        tickfont=dict(color=th["subtext"], size=10),
                        showline=True, linecolor=th["border"],
                    ),
                    yaxis=dict(
                        title=y_title,
                        gridcolor=th["grid"],
                        tickfont=dict(color=th["subtext"], size=10),
                        ticksuffix="%",
                    ),
                    hovermode="x unified",
                    legend=dict(
                        bgcolor=th["card2"], bordercolor=th["border"],
                        borderwidth=1, font_color=th["text"], font_size=10,
                        orientation="v", x=1.01, xanchor="left",
                    ),
                    margin=dict(l=50, r=120, t=55, b=45),
                )
                return fig, False

            # Kaggle chart (complete data, 1960–2017)
            title_k = "📦 Kaggle — Vollständige Daten (1960–2017)" if lang=="de" else "📦 Kaggle — Complete Data (1960–2017)"
            fig_k, empty_k = make_genre_fig(genre_trends, title_k,
                                             max(d_min, 1960), min(d_max, KAGGLE_END),
                                             "Kaggle")

            # TMDb chart (top-200 per year, 2018–2025)
            title_t = "🌐 TMDb — Top-200 Filme/Jahr (2018–2025)" if lang=="de" else "🌐 TMDb — Top-200 Films/Year (2018–2025)"
            fig_t, empty_t = make_genre_fig(genre_trends, title_t,
                                             max(d_min, TMDB_START), min(d_max, 2025),
                                             "TMDb Top-200")

            note_de = ("📌 Zwei Datenquellen — deshalb zwei Grafiken. Kaggle (links) enthält alle Filme bis 2017 und ist vollständig repräsentativ. "
                       "TMDb (rechts) enthält nur die 200 populärsten Filme pro Jahr ab 2018 — Trends sind erkennbar aber nicht vollständig.")
            note_en = ("📌 Two data sources — two charts. Kaggle (left) contains all films up to 2017 and is fully representative. "
                       "TMDb (right) contains only the 200 most popular films per year from 2018 — trends are visible but not complete.")

            kaggle_block = dcc.Graph(figure=fig_k, style={"height":"380px"},
                                     config={"displayModeBar":True,"displaylogo":False}) if not empty_k                            else no_data_card("Keine Kaggle-Daten im gewählten Bereich.", th)

            tmdb_block = dcc.Graph(figure=fig_t, style={"height":"380px"},
                                   config={"displayModeBar":True,"displaylogo":False}) if not empty_t                          else no_data_card("Keine TMDb-Daten im gewählten Bereich." if lang=="de" else "No TMDb data in selected range.", th)

            return html.Div([
                insight_box(note_de if lang=="de" else note_en, th),
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Div("✅ Vollständig" if lang=="de" else "✅ Complete",
                                 style={"background":"rgba(46,204,113,0.15)",
                                        "color":th["green"],"fontSize":"0.72rem",
                                        "fontWeight":"700","letterSpacing":"1px",
                                        "borderRadius":"6px","padding":"3px 10px",
                                        "display":"inline-block","marginBottom":"8px"}),
                        kaggle_block,
                    ]), md=6),
                    dbc.Col(html.Div([
                        html.Div("⚠️ Top-200 Sample" if lang=="de" else "⚠️ Top-200 Sample",
                                 style={"background":"rgba(224,123,57,0.15)",
                                        "color":th["accent2"],"fontSize":"0.72rem",
                                        "fontWeight":"700","letterSpacing":"1px",
                                        "borderRadius":"6px","padding":"3px 10px",
                                        "display":"inline-block","marginBottom":"8px"}),
                        tmdb_block,
                    ]), md=6),
                ], className="g-3"),
            ], style=card)

        except Exception as e:
            return html.Div([error_card(t["error_msg"] + traceback.format_exc(), th)], style=card)

    # ── TAB 2 : World Map ─────────────────────────────────────────────────────
    elif tab == "tab-map":
        try:
            map_df = (
                full_df[
                    (full_df["release_year"] >= d_min) &
                    (full_df["release_year"] <= d_max)
                ]
                .groupby("country_clean", as_index=False)
                .agg(movies=("title","count"), avg_revenue=("revenue_adj","mean"))
            )
            map_df["avg_rev_M"] = (map_df["avg_revenue"] / 1e6).round(1)

            fig = px.choropleth(
                map_df,
                locations="country_clean", locationmode="country names",
                color="movies", hover_name="country_clean",
                hover_data={"movies":True,"avg_rev_M":True,"country_clean":False},
                color_continuous_scale=[
                    [0.00, th["card2"]],
                    [0.10, "#1e3a5f"],
                    [0.35, "#1a6b8a"],
                    [0.65, th["accent2"]],
                    [1.00, th["accent"]],
                ],
                labels={"movies":t["country_hover"],"avg_rev_M":t["avg_rev_hover"]},
            )
            fig.update_traces(
                hovertemplate=(
                    "<b>%{hovertext}</b><br>"
                    f"{t['country_hover']}: %{{z:,}}<br>"
                    "<extra></extra>"
                )
            )
            fig.update_layout(
                **plot_layout(th, title_text=t["chart_map"], height=480),
                geo=dict(
                    bgcolor="rgba(0,0,0,0)",
                    lakecolor=th["lake"],
                    landcolor=th["land"],
                    showframe=False,
                    showcoastlines=True,
                    coastlinecolor=th["coast"],
                    showocean=True,
                    oceancolor=th["bg"],
                    showcountries=True,
                    countrycolor=th["border"],
                ),
                coloraxis_colorbar=dict(
                    title=t["country_hover"],
                    tickfont_color=th["text"],
                    title_font_color=th["text"],
                    bgcolor=th["card2"],
                    bordercolor=th["border"],
                    thickness=14,
                    len=0.6,
                ),
                margin=dict(l=0, r=0, t=60, b=0),
            )
            return html.Div([
                insight_box(t["sub_map"], th),
                dcc.Graph(figure=fig, style={"height":"480px"},
                          config={"displayModeBar":True,"displaylogo":False}),
            ], style=card)

        except Exception as e:
            return html.Div([error_card(t["error_msg"] + traceback.format_exc(), th)], style=card)

    # ── TAB 3 : Budget vs Revenue ─────────────────────────────────────────────
    elif tab == "tab-scatter":
        try:
            if df_fin.empty:
                return html.Div([no_data_card(t["no_data"], th)], style=card)

            sample = (df_fin.sample(min(2500, len(df_fin)), random_state=42)
                            .reset_index(drop=True))

            # Separate oscar winners for special treatment
            winners = sample[sample["is_oscar_winner"]].reset_index(drop=True)
            others  = sample[~sample["is_oscar_winner"]].reset_index(drop=True)

            fig = go.Figure()

            # Regular films — colored by genre
            for i, g in enumerate(genres):
                gdf = others[others["primary_genre"] == g].reset_index(drop=True)
                if gdf.empty:
                    continue
                col = CHART_COLORS[i % len(CHART_COLORS)]
                fig.add_trace(go.Scatter(
                    x=gdf["budget_adj"], y=gdf["revenue_adj"],
                    mode="markers", name=g,
                    marker=dict(color=col, size=6, opacity=0.55,
                                line=dict(color="rgba(0,0,0,0)", width=0)),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        f"{t['decade_x']}: %{{customdata[1]}}<br>"
                        f"Budget: $%{{x:,.0f}}<br>"
                        f"Revenue: $%{{y:,.0f}}<extra>{g}</extra>"
                    ),
                    customdata=gdf[["title","release_year"]].values,
                ))

            # Oscar winners — gold stars on top
            if not winners.empty:
                fig.add_trace(go.Scatter(
                    x=winners["budget_adj"], y=winners["revenue_adj"],
                    mode="markers", name="⭐ Oscar Winner",
                    marker=dict(
                        symbol="star", color=th["accent"],
                        size=14, opacity=0.95,
                        line=dict(color=th["bg"], width=1),
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> ⭐<br>"
                        f"Budget: $%{{x:,.0f}}<br>"
                        f"Revenue: $%{{y:,.0f}}<extra>Oscar Winner</extra>"
                    ),
                    customdata=winners[["title","release_year"]].values,
                ))

            # Break-even line
            bmin = float(sample["budget_adj"].min())
            bmax = float(sample["budget_adj"].max())
            fig.add_trace(go.Scatter(
                x=[bmin, bmax], y=[bmin, bmax],
                mode="lines", name=t["break_even"],
                line=dict(color=th["subtext"], dash="dash", width=1.5),
                hoverinfo="skip",
            ))

            fig.update_layout(
                **plot_layout(th, title_text=t["chart_scatter"], height=500),
                xaxis=dict(
                    type="log", title=t["budget_x"],
                    gridcolor=th["grid"],
                    tickfont=dict(color=th["subtext"], size=11),
                    tickprefix="$", showline=True, linecolor=th["border"],
                ),
                yaxis=dict(
                    type="log", title=t["revenue_y"],
                    gridcolor=th["grid"],
                    tickfont=dict(color=th["subtext"], size=11),
                    tickprefix="$", showline=False,
                ),
                hovermode="closest",
                legend=dict(
                    bgcolor=th["card2"], bordercolor=th["border"],
                    borderwidth=1, font_color=th["text"], font_size=11,
                    x=1.02, xanchor="left",
                ),
            )
            search_label = "🔍 Film suchen & hervorheben:" if lang=="de" else "🔍 Search & highlight a film:"
            return html.Div([
                insight_box(t["sub_scatter"], th),
                html.Div([
                    html.Label(search_label,
                               style={"color":th["subtext"],"fontSize":"0.75rem",
                                      "fontWeight":"700","letterSpacing":"1px",
                                      "textTransform":"uppercase","marginBottom":"6px",
                                      "display":"block"}),
                    dcc.Input(
                        id="scatter-search",
                        type="text",
                        placeholder="z.B. Titanic, Avatar, Inception..." if lang=="de" else "e.g. Titanic, Avatar, Inception...",
                        debounce=True,
                        style={"backgroundColor":th["card2"],"color":th["text"],
                               "border":f"1px solid {th['border']}","borderRadius":"8px",
                               "padding":"8px 14px","width":"320px","fontSize":"0.88rem"},
                    ),
                ], style={"marginBottom":"14px"}),
                dcc.Graph(id="scatter-graph", figure=fig, style={"height":"500px"},
                          config={"displayModeBar":True,"displaylogo":False}),
            ], style=card)

        except Exception as e:
            return html.Div([error_card(t["error_msg"] + traceback.format_exc(), th)], style=card)

    # ── TAB 4 : Oscar Winners ─────────────────────────────────────────────────
    elif tab == "tab-oscar":
        try:
            osc = (
                full_df[
                    full_df["is_oscar_winner"] &
                    (full_df["release_year"] >= d_min) &
                    (full_df["release_year"] <= d_max)
                ]
                .copy().reset_index(drop=True)
                .sort_values("release_year")
            )
            if osc.empty:
                return html.Div([no_data_card(t["no_data"], th)], style=card)

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Revenue bars
            rev_vals = osc["revenue_adj"].fillna(0) / 1e6
            fig.add_trace(go.Bar(
                x=osc["release_year"], y=rev_vals,
                name=t["revenue_label"],
                marker=dict(
                    color=rev_vals,
                    colorscale=[
                        [0.0, th["card3"]],
                        [0.4, th["accent2"]],
                        [1.0, th["accent"]],
                    ],
                    showscale=False,
                    line=dict(color=th["bg"], width=0.5),
                ),
                hovertemplate=(
                    "<b>%{customdata}</b><br>"
                    "Year: %{x}<br>"
                    "Revenue: $%{y:.0f}M<extra></extra>"
                ),
                customdata=osc["title"],
                opacity=0.92,
            ), secondary_y=False)

            # Rating line
            fig.add_trace(go.Scatter(
                x=osc["release_year"], y=osc["vote_average"],
                name=t["rating_label"],
                mode="lines+markers",
                marker=dict(color=th["red"], size=8,
                            symbol="circle",
                            line=dict(color=th["card"], width=1.5)),
                line=dict(color=th["red"], width=2),
                hovertemplate="Rating: %{y:.1f}/10<extra></extra>",
            ), secondary_y=True)

            fig.update_layout(
                **plot_layout(th, title_text=t["chart_oscar"], height=500),
                hovermode="x unified",
                bargap=0.25,
                legend=dict(
                    bgcolor=th["card2"], bordercolor=th["border"],
                    borderwidth=1, font_color=th["text"], font_size=11,
                    orientation="h", y=1.08, x=0,
                ),
            )
            fig.update_yaxes(
                title_text=t["revenue_label"], secondary_y=False,
                gridcolor=th["grid"],
                tickfont=dict(color=th["subtext"], size=11),
                tickprefix="$",
            )
            fig.update_yaxes(
                title_text=t["rating_label"], secondary_y=True,
                range=[4, 11], showgrid=False,
                tickfont=dict(color=th["red"], size=11),
                title_font=dict(color=th["red"]),
            )
            return html.Div([
                insight_box(t["sub_oscar"], th),
                dcc.Graph(figure=fig, style={"height":"500px"},
                          config={"displayModeBar":True,"displaylogo":False}),
            ], style=card)

        except Exception as e:
            return html.Div([error_card(t["error_msg"] + traceback.format_exc(), th)], style=card)

    # ── TAB 5 : Revenue by Decade ─────────────────────────────────────────────
    elif tab == "tab-revenue":
        try:
            if df_fin.empty:
                return html.Div([no_data_card(t["no_data"], th)], style=card)

            rev = (
                df_fin.groupby("decade", as_index=False)
                      .agg(median_revenue=("revenue_adj","median"),
                           median_budget=("budget_adj","median"),
                           movie_count=("title","count"),
                           avg_roi=("roi","median"))
                      .sort_values("decade")
            )

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    t["panel_rev_bud"], t["panel_count"],
                    t["panel_roi"],     t["panel_index"],
                ],
                vertical_spacing=0.24,
                horizontal_spacing=0.14,
            )

            # Panel 1 — grouped bars
            fig.add_trace(go.Bar(
                x=rev["decade"], y=rev["median_revenue"]/1e6,
                name=t["revenue_label"],
                marker=dict(color=th["accent"], opacity=0.9,
                            line=dict(color=th["bg"], width=0.5)),
                hovertemplate="$%{y:.0f}M<extra>Revenue</extra>"),
                row=1, col=1)
            fig.add_trace(go.Bar(
                x=rev["decade"], y=rev["median_budget"]/1e6,
                name=t["budget_label"],
                marker=dict(color=th["blue"], opacity=0.9,
                            line=dict(color=th["bg"], width=0.5)),
                hovertemplate="$%{y:.0f}M<extra>Budget</extra>"),
                row=1, col=1)

            # Panel 2 — smooth line
            fig.add_trace(go.Scatter(
                x=rev["decade"], y=rev["movie_count"],
                mode="lines+markers", name=t["films_y"],
                line=dict(color=th["green"], width=2.5),
                marker=dict(size=8, color=th["green"],
                            line=dict(color=th["card"], width=1.5)),
                fill="tozeroy",
                fillcolor="rgba(46,204,113,0.08)",
                hovertemplate="%{y:,} films<extra></extra>"),
                row=1, col=2)

            # Panel 3 — ROI bars with conditional color
            roi_vals  = (rev["avg_roi"].fillna(0) * 100).round(1)
            roi_colors = [th["green"] if v >= 0 else th["red"] for v in roi_vals]
            fig.add_trace(go.Bar(
                x=rev["decade"], y=roi_vals,
                name="ROI %",
                marker=dict(color=roi_colors, opacity=0.9,
                            line=dict(color=th["bg"], width=0.5)),
                hovertemplate="%{y:.1f}%<extra>ROI</extra>"),
                row=2, col=1)

            # Panel 4 — growth index with annotation labels
            base = rev["median_revenue"].iloc[0]
            base = base if base > 0 else 1
            idx  = (rev["median_revenue"] / base * 100).round(0)
            fig.add_trace(go.Scatter(
                x=rev["decade"], y=idx,
                mode="lines+markers+text",
                text=[f"{v:.0f}" for v in idx],
                textposition="top center",
                textfont=dict(size=9, color=th["accent"]),
                line=dict(color=th["accent2"], width=2.5),
                marker=dict(size=8, color=th["accent2"],
                            line=dict(color=th["card"], width=1.5)),
                fill="tozeroy",
                fillcolor="rgba(224,123,57,0.07)",
                hovertemplate="Index: %{y:.0f}<extra></extra>"),
                row=2, col=2)

            fig.update_layout(
                **plot_layout(th, title_text=t["chart_rev"], height=560),
                showlegend=False,
                barmode="group",
            )
            fig.update_xaxes(tickangle=45, tickfont_size=10,
                             tickfont_color=th["subtext"],
                             gridcolor=th["grid"], linecolor=th["border"])
            fig.update_yaxes(gridcolor=th["grid"],
                             tickfont_color=th["subtext"], zeroline=False)
            for ann in fig.layout.annotations:
                ann.font.color  = th["text"]
                ann.font.size   = 12
                ann.font.family = "DM Sans, sans-serif"

            # ── Dynamic insight texts ─────────────────────────────────────
            best_roi_decade = rev.loc[rev["avg_roi"].idxmax(), "decade"] if not rev.empty else "—"
            best_roi_val    = rev["avg_roi"].max() * 100 if not rev.empty else 0
            best_rev_decade = rev.loc[rev["median_revenue"].idxmax(), "decade"] if not rev.empty else "—"
            best_rev_val    = rev["median_revenue"].max() / 1e6 if not rev.empty else 0
            total_films     = int(rev["movie_count"].sum()) if not rev.empty else 0
            base_val        = rev["median_revenue"].iloc[0] if not rev.empty and rev["median_revenue"].iloc[0] > 0 else 1
            growth_factor   = round(rev["median_revenue"].iloc[-1] / base_val, 1) if not rev.empty else 0

            if lang == "de":
                ins1 = f"💰 Höchste Median-Einnahmen in den {best_rev_decade} mit ${best_rev_val:.0f}M pro Film (inflationsbereinigt)."
                ins2 = f"📈 Bester ROI in den {best_roi_decade} — das typische Studio verdiente {best_roi_val:.0f}% über dem Produktionsbudget."
                ins3 = f"🎬 Insgesamt {total_films:,} Filme mit vollständigen Finanzdaten im gewählten Zeitraum."
                ins4 = f"📊 Einnahmen-Wachstum: Das letzte Jahrzehnt ist {growth_factor}× höher als das erste — trotz Inflationsbereinigung."
            else:
                ins1 = f"💰 Highest median revenue in the {best_rev_decade} at ${best_rev_val:.0f}M per film (inflation-adjusted)."
                ins2 = f"📈 Best ROI in the {best_roi_decade} — the typical studio earned {best_roi_val:.0f}% above its production budget."
                ins3 = f"🎬 {total_films:,} films with complete financial data in the selected period."
                ins4 = f"📊 Revenue growth: the last decade is {growth_factor}× higher than the first — even after inflation adjustment."

            def stat_insight(text, color):
                return html.Div(text, style={
                    "color": th["text"], "fontSize": "0.82rem",
                    "background": th["card2"],
                    "border": f"1px solid {th['border']}",
                    "borderLeft": f"3px solid {color}",
                    "borderRadius": "8px", "padding": "10px 14px", "lineHeight": "1.5",
                })

            return html.Div([
                insight_box(t["sub_revenue"], th),
                dcc.Graph(figure=fig, style={"height":"560px"},
                          config={"displayModeBar":True,"displaylogo":False}),
                dbc.Row([
                    dbc.Col(stat_insight(ins1, th["accent"]),  md=6, style={"marginTop":"12px"}),
                    dbc.Col(stat_insight(ins2, th["green"]),   md=6, style={"marginTop":"12px"}),
                    dbc.Col(stat_insight(ins3, th["blue"]),    md=6, style={"marginTop":"8px"}),
                    dbc.Col(stat_insight(ins4, th["accent2"]), md=6, style={"marginTop":"8px"}),
                ], className="g-2"),
            ], style=card)

        except Exception as e:
            return html.Div([error_card(t["error_msg"] + traceback.format_exc(), th)], style=card)

    # ── TAB 6 : 2026 Predictions ──────────────────────────────────────────────
    elif tab == "tab-predict":
        try:
            pred_s = pred_df.sort_values("Predicted Rev ($M)").reset_index(drop=True)
            n = len(pred_s)

            # Horizontal bar chart with gradient color + CI
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=pred_s["Movie"],
                x=pred_s["Predicted Rev ($M)"],
                orientation="h",
                marker=dict(
                    color=list(range(n)),
                    colorscale=[
                        [0.0, th["blue"]],
                        [0.5, th["accent2"]],
                        [1.0, th["accent"]],
                    ],
                    showscale=False,
                    line=dict(color=th["bg"], width=0.5),
                    opacity=0.9,
                ),
                error_x=dict(
                    type="data", symmetric=False,
                    array=(pred_s["High Estimate ($M)"] - pred_s["Predicted Rev ($M)"]).values,
                    arrayminus=(pred_s["Predicted Rev ($M)"] - pred_s["Low Estimate ($M)"]).values,
                    color=th["subtext"], thickness=2, width=8,
                ),
                text=[f"  ${v:.0f}M" for v in pred_s["Predicted Rev ($M)"]],
                textposition="outside",
                textfont=dict(color=th["text"], size=12,
                              family="'Bebas Neue', sans-serif"),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Predicted: $%{x:.0f}M<br>"
                    "<extra></extra>"
                ),
            ))
            fig.update_layout(
                **plot_layout(th, title_text=t["chart_pred"], height=360),
                xaxis=dict(
                    title="Worldwide Box Office Revenue ($M)",
                    gridcolor=th["grid"],
                    tickfont=dict(color=th["subtext"], size=11),
                    tickprefix="$", showline=True, linecolor=th["border"],
                    range=[0, pred_s["High Estimate ($M)"].max() * 1.35],
                ),
                yaxis=dict(
                    tickfont=dict(color=th["text"], size=12),
                    showline=False,
                ),
                margin=dict(l=160, r=40, t=70, b=50),
            )

            # ── Live predictor ────────────────────────────────────────────────
            predictor = html.Div([
                html.Div([
                    html.H5(t["predictor_title"],
                            style={"color":th["accent"],
                                   "fontFamily":"'Bebas Neue', sans-serif",
                                   "letterSpacing":"2px","fontSize":"1.4rem",
                                   "marginBottom":"4px"}),
                    html.P(t["predictor_sub"],
                           style={"color":th["subtext"],"fontSize":"0.82rem",
                                  "marginBottom":"18px","lineHeight":"1.5"}),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label(t["budget_label"],
                                   style={"color":th["subtext"],"fontSize":"0.7rem",
                                          "letterSpacing":"1.5px","fontWeight":"700",
                                          "textTransform":"uppercase","display":"block",
                                          "marginBottom":"5px"}),
                        dbc.Input(id="pred-budget", type="number", value=150,
                                  min=1, max=2000, step=10,
                                  style={"backgroundColor":th["card3"],
                                         "color":th["text"],
                                         "border":f"1px solid {th['border']}",
                                         "borderRadius":"8px","padding":"8px 12px"}),
                    ], md=3),
                    dbc.Col([
                        html.Label(t["genre_label"],
                                   style={"color":th["subtext"],"fontSize":"0.7rem",
                                          "letterSpacing":"1.5px","fontWeight":"700",
                                          "textTransform":"uppercase","display":"block",
                                          "marginBottom":"5px"}),
                        dcc.Dropdown(id="pred-genre",
                                     options=[{"label":g,"value":g}
                                              for g in sorted(KNOWN_GENRES)],
                                     value="Action", clearable=False),
                    ], md=3),
                    dbc.Col([
                        html.Label(t["month_label"],
                                   style={"color":th["subtext"],"fontSize":"0.7rem",
                                          "letterSpacing":"1.5px","fontWeight":"700",
                                          "textTransform":"uppercase","display":"block",
                                          "marginBottom":"5px"}),
                        dcc.Dropdown(id="pred-month",
                                     options=[{"label":m,"value":i}
                                              for i,m in enumerate(t["months"],1)],
                                     value=7, clearable=False),
                    ], md=3),
                    dbc.Col([
                        html.Label(t["runtime_label"],
                                   style={"color":th["subtext"],"fontSize":"0.7rem",
                                          "letterSpacing":"1.5px","fontWeight":"700",
                                          "textTransform":"uppercase","display":"block",
                                          "marginBottom":"5px"}),
                        dbc.Input(id="pred-runtime", type="number", value=130,
                                  min=60, max=240, step=5,
                                  style={"backgroundColor":th["card3"],
                                         "color":th["text"],
                                         "border":f"1px solid {th['border']}",
                                         "borderRadius":"8px","padding":"8px 12px"}),
                    ], md=3),
                ], className="g-3 mb-3"),
                dbc.Button(t["predict_btn"], id="predict-btn",
                           style={
                               "backgroundColor": th["accent"],
                               "border":          "none",
                               "color":           "#000",
                               "fontWeight":      "700",
                               "borderRadius":    "8px",
                               "padding":         "11px 28px",
                               "fontFamily":      "'Bebas Neue', sans-serif",
                               "letterSpacing":   "1.5px",
                               "fontSize":        "1.05rem",
                               "cursor":          "pointer",
                           }),
                html.Div(id="prediction-output", style={"marginTop":"20px"}),
            ], style={
                "background":   th["card2"],
                "border":       f"1px solid {th['border']}",
                "borderTop":    f"3px solid {th['accent']}",
                "borderRadius": "12px",
                "padding":      "24px 28px",
                "marginTop":    "22px",
            })

            return html.Div([
                insight_box(t["sub_predict"].format(model=MODEL_NAME), th),
                dcc.Graph(figure=fig,
                          config={"displayModeBar":True,"displaylogo":False}),
                predictor,
            ], style=card)

        except Exception as e:
            return html.Div([error_card(t["error_msg"] + traceback.format_exc(), th)], style=card)

    return html.Div("Select a tab.", style={"color":th["subtext"],"padding":"40px"})



# ─────────────────────────────────────────────────────────────────────────────
# SCATTER SEARCH CALLBACK — highlight a searched film
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("scatter-graph", "figure"),
    Input("scatter-search", "value"),
    State("decade-slider", "value"),
    State("genre-filter",  "value"),
    State("theme-store",   "data"),
    State("lang-store",    "data"),
    prevent_initial_call=True,
)
def highlight_film(search, decade_range, genres, theme, lang):
    th     = get_th(theme)
    t      = T[lang]
    d_min  = int(decade_range[0])
    d_max  = int(decade_range[1])
    genres = genres or TOP_GENRES[:6]

    df_fin = fin_df[
        (fin_df["release_year"] >= d_min) &
        (fin_df["release_year"] <= d_max) &
        (fin_df["primary_genre"].isin(genres))
    ].copy().reset_index(drop=True)

    sample = df_fin.sample(min(2500, len(df_fin)), random_state=42).reset_index(drop=True)

    winners = sample[sample["is_oscar_winner"]].reset_index(drop=True)
    others  = sample[~sample["is_oscar_winner"]].reset_index(drop=True)

    fig = go.Figure()

    for i, g in enumerate(genres):
        gdf = others[others["primary_genre"] == g].reset_index(drop=True)
        if gdf.empty:
            continue
        col = CHART_COLORS[i % len(CHART_COLORS)]
        fig.add_trace(go.Scatter(
            x=gdf["budget_adj"], y=gdf["revenue_adj"],
            mode="markers", name=g,
            marker=dict(color=col, size=6, opacity=0.45,
                        line=dict(color="rgba(0,0,0,0)", width=0)),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"Budget: $%{{x:,.0f}}<br>"
                f"Revenue: $%{{y:,.0f}}<extra>{g}</extra>"
            ),
            customdata=gdf[["title","release_year"]].values,
        ))

    if not winners.empty:
        fig.add_trace(go.Scatter(
            x=winners["budget_adj"], y=winners["revenue_adj"],
            mode="markers", name="⭐ Oscar Winner",
            marker=dict(symbol="star", color=th["accent"], size=14, opacity=0.95,
                        line=dict(color=th["bg"], width=1)),
            hovertemplate="<b>%{customdata[0]}</b> ⭐<br>$%{x:,.0f} → $%{y:,.0f}<extra></extra>",
            customdata=winners[["title","release_year"]].values,
        ))

    # Break-even line
    if len(sample) > 0:
        bmin = float(sample["budget_adj"].min())
        bmax = float(sample["budget_adj"].max())
        fig.add_trace(go.Scatter(
            x=[bmin, bmax], y=[bmin, bmax],
            mode="lines", name=t["break_even"],
            line=dict(color=th["subtext"], dash="dash", width=1.5),
            hoverinfo="skip",
        ))

    # Highlight searched film
    if search and len(search) >= 2:
        mask = fin_df["title"].str.contains(search, case=False, na=False)
        found = fin_df[mask].reset_index(drop=True)
        if not found.empty:
            fig.add_trace(go.Scatter(
                x=found["budget_adj"], y=found["revenue_adj"],
                mode="markers+text",
                name=f"🔍 {search}",
                text=found["title"],
                textposition="top center",
                textfont=dict(color=th["accent"], size=11, family="DM Sans"),
                marker=dict(
                    color=th["accent"], size=16, symbol="circle",
                    line=dict(color=th["bg"], width=2),
                    opacity=1.0,
                ),
                hovertemplate="<b>%{text}</b><br>Budget: $%{x:,.0f}<br>Revenue: $%{y:,.0f}<extra>🔍 Gefunden</extra>",
            ))

    fig.update_layout(
        **plot_layout(th, title_text=t["chart_scatter"], height=500),
        xaxis=dict(
            type="log", title=t["budget_x"],
            gridcolor=th["grid"],
            tickfont=dict(color=th["subtext"], size=11),
            tickprefix="$", showline=True, linecolor=th["border"],
        ),
        yaxis=dict(
            type="log", title=t["revenue_y"],
            gridcolor=th["grid"],
            tickfont=dict(color=th["subtext"], size=11),
            tickprefix="$", showline=False,
        ),
        hovermode="closest",
        legend=dict(
            bgcolor=th["card2"], bordercolor=th["border"],
            borderwidth=1, font_color=th["text"], font_size=11,
            x=1.02, xanchor="left",
        ),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 9. LIVE PREDICTOR CALLBACK
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("prediction-output","children"),
    Input("predict-btn",  "n_clicks"),
    State("pred-budget",  "value"),
    State("pred-genre",   "value"),
    State("pred-month",   "value"),
    State("pred-runtime", "value"),
    State("theme-store",  "data"),
    State("lang-store",   "data"),
    prevent_initial_call=True,
)
def live_predict(n, budget_m, genre, month, runtime, theme, lang):
    th = get_th(theme)
    t  = T[lang]

    if not all([budget_m, genre, month, runtime]):
        return dbc.Alert(t["fill_fields"], color="warning")

    try:
        budget_adj = float(budget_m) * 1e6 * 1.05
        log_budget = float(np.log1p(budget_adj))
        month      = int(month)
        is_summer  = int(month in [5,6,7,8])
        is_holiday = int(month in [11,12])
        genre_safe = genre if genre in GENRE_CODE else "Drama"
        genre_enc  = GENRE_CODE[genre_safe]

        X_live = pd.DataFrame([{
            "log_budget":      log_budget,
            "runtime":         float(runtime) if runtime else MEDIAN_RT,
            "release_month":   month,
            "is_summer":       is_summer,
            "is_holiday":      is_holiday,
            "genre_encoded":   genre_enc,
            "is_oscar_winner": 0,
            "release_year":    2026,
        }])[FEATURE_COLS]

        log_pred = float(model.predict(X_live)[0])
        rev_pred = float(np.expm1(log_pred)) / 1e6
        roi      = (rev_pred - float(budget_m)) / float(budget_m) * 100

        # 80% prediction interval, derived from the model's errors on the test set
        low  = float(np.expm1(log_pred + INTERVAL_LOG[0])) / 1e6
        high = float(np.expm1(log_pred + INTERVAL_LOG[1])) / 1e6

        roi_color = th["green"] if roi >= 0 else th["red"]
        season    = (t["summer"] if is_summer
                     else (t["holiday"] if is_holiday else t["standard"]))

        def stat_block(label, value, sub, color):
            return html.Div([
                html.Div(label, style={"color":th["subtext"],"fontSize":"0.66rem",
                                       "letterSpacing":"2px","fontWeight":"700",
                                       "textTransform":"uppercase","marginBottom":"3px"}),
                html.Div(value, style={"color":color,"fontWeight":"700",
                                       "fontSize":"2.1rem",
                                       "fontFamily":"'Bebas Neue', sans-serif",
                                       "letterSpacing":"2px","lineHeight":"1.1"}),
                html.Div(sub,   style={"color":th["subtext"],"fontSize":"0.75rem",
                                       "marginTop":"2px"}),
            ])

        return html.Div([
            dbc.Row([
                dbc.Col([
                    stat_block(t["pred_revenue"],
                               f"${rev_pred:,.0f}M",
                               f"{t['range_80']}: ${low:,.0f}M – ${high:,.0f}M",
                               th["accent"])
                ], md=4),
                dbc.Col([
                    stat_block(t["est_roi"],
                               f"{roi:+.1f}%",
                               f"${budget_m}M budget → ${rev_pred:,.0f}M revenue",
                               roi_color)
                ], md=4),
                dbc.Col([
                    html.Div(t["inputs"],
                             style={"color":th["subtext"],"fontSize":"0.66rem",
                                    "letterSpacing":"2px","fontWeight":"700",
                                    "textTransform":"uppercase","marginBottom":"8px"}),
                    html.Div([
                        html.Span(f"Genre: {genre}",
                                  style={"background":th["card3"],"borderRadius":"6px",
                                         "padding":"3px 8px","fontSize":"0.78rem",
                                         "marginRight":"6px","color":th["text"]}),
                        html.Span(t["months"][month-1],
                                  style={"background":th["card3"],"borderRadius":"6px",
                                         "padding":"3px 8px","fontSize":"0.78rem",
                                         "color":th["text"]}),
                    ], style={"marginBottom":"6px"}),
                    html.Div([
                        html.Span(f"{runtime} min",
                                  style={"background":th["card3"],"borderRadius":"6px",
                                         "padding":"3px 8px","fontSize":"0.78rem",
                                         "marginRight":"6px","color":th["text"]}),
                        html.Span(season,
                                  style={"background":th["card3"],"borderRadius":"6px",
                                         "padding":"3px 8px","fontSize":"0.78rem",
                                         "color":th["text"]}),
                    ]),
                ], md=4),
            ]),
        ], style={
            "background":   f"linear-gradient(135deg, {th['card']} 0%, {th['card2']} 100%)",
            "border":       f"2px solid {th['accent']}",
            "borderRadius": "12px",
            "padding":      "22px 28px",
        })

    except Exception as e:
        return dbc.Alert(f"Prediction error: {str(e)}", color="danger")


# ─────────────────────────────────────────────────────────────────────────────
# 10. RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*52)
    print("  🎬 Cinema BI Dashboard")
    print("  Open → http://127.0.0.1:8050")
    print("="*52 + "\n")
    app.run(debug=False, port=8050)
