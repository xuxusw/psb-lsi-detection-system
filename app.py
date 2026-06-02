import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from cache import load_data
from signal_engine import run_signal_engine, STRESS_EPISODES, CALM_EPISODES

# читаю API-ключ из .env файла, если он есть
def _load_env_key(name):
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.getenv(name, "")

GROQ_API_KEY = _load_env_key("GROQ_API_KEY")

# загружаю данные и запускаю расчёт LSI
print("Инициализация системы...")
RAW_DATA = load_data()
RESULTS = run_signal_engine(RAW_DATA, method="gbm")
DF_LSI = RESULTS["lsi"]
DF_LSI["date"] = pd.to_datetime(DF_LSI["date"])
BACKTEST = RESULTS["backtest"]
SENSITIVITY = RESULTS["sensitivity"]
SIGNALS = RESULTS["signals"]

# беру последнее значение LSI для отображения в шапке
LAST_LSI = round(float(DF_LSI["lsi"].iloc[-1]), 1)
LAST_STATUS = DF_LSI["status"].iloc[-1]
LAST_DATE = DF_LSI["date"].iloc[-1].strftime("%d.%m.%Y")

# собираю активные флаги на последнюю дату
last_row = DF_LSI.iloc[-1]
active_flags = []
if last_row.get("flag_demand", 0): active_flags.append("Переспрос репо ЦБ")
if last_row.get("flag_nedospros", 0): active_flags.append("Недоспрос ОФЗ")
if last_row.get("flag_budget_drain", 0): active_flags.append("Отток казначейства")
if last_row.get("tax_week_flag", 0): active_flags.append("Налоговая неделя")
if last_row.get("end_of_quarter_flag", 0): active_flags.append("Конец квартала")

# цвета и метаданные модулей
STATUS_COLOR = {"🟢 НОРМА": "#1D9E75", "🟡 ВНИМАНИЕ": "#BA7517", "🔴 СТРЕСС": "#E24B4A"}
STATUS_BG = {"🟢 НОРМА": "#E1F5EE", "🟡 ВНИМАНИЕ": "#FAEEDA", "🔴 СТРЕСС": "#FCEBEB"}
MODULE_META = [
    ("contrib_m1", "М1 Резервы", "#7F77DD"),
    ("contrib_m2", "М2 Репо ЦБ", "#1D9E75"),
    ("contrib_m3", "М3 ОФЗ", "#D4537E"),
    ("contrib_m4", "М4 Сезон", "#EF9F27"),
    ("contrib_m5", "М5 Казначейство", "#378ADD"),
]

print(f"Готово. LSI={LAST_LSI}, {LAST_STATUS}")

# стили приложения — шрифт, шапка, карточки, чат
GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
* { box-sizing: border-box; }
body { font-family: 'Inter', -apple-system, sans-serif; background: #F7F8FA; color: #1a1a2e; margin: 0; }
.hdr { display:flex; justify-content:space-between; align-items:center;
       padding:12px 24px; background:#fff; border-bottom:1px solid #EAECF0;
       position:sticky; top:0; z-index:100; }
.hdr-brand { display:flex; align-items:center; gap:10px; }
.hdr-dot { width:8px; height:8px; border-radius:50%; background:#378ADD; }
.hdr-title { font-size:14px; font-weight:600; color:#1a1a2e; }
.hdr-sub { font-size:11px; color:#6B7280; margin-top:1px; }
.status-badge { font-size:12px; font-weight:500; padding:4px 12px;
                border-radius:20px; border:1px solid transparent; }
.tabs-bar { display:flex; background:#fff; border-bottom:1px solid #EAECF0;
            padding:0 24px; gap:4px; }
.tab-btn { font-size:13px; padding:10px 16px; color:#6B7280; cursor:pointer;
           border:none; border-bottom:2px solid transparent; background:none;
           font-family:inherit; transition:color .15s; }
.tab-btn:hover { color:#374151; }
.tab-btn.active { color:#185FA5; border-bottom-color:#185FA5; font-weight:500; }
.page-body { padding:20px 24px; min-height:calc(100vh - 110px); }
.metrics-row { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-bottom:14px; }
.mc { background:#fff; border:1px solid #EAECF0; border-radius:10px;
      padding:12px 14px; border-top:3px solid #ccc; }
.mc-label { font-size:11px; color:#6B7280; margin-bottom:4px; }
.mc-value { font-size:20px; font-weight:600; color:#1a1a2e; line-height:1.2; }
.mc-sub { font-size:10px; color:#9CA3AF; margin-top:2px; }
.alert-bar { display:flex; align-items:center; gap:8px; padding:10px 14px;
             border-radius:8px; margin-bottom:14px; font-size:12px;
             border:1px solid transparent; }
.charts-row { display:grid; grid-template-columns:1fr 200px; gap:12px; margin-bottom:12px; }
.card { background:#fff; border:1px solid #EAECF0; border-radius:12px; padding:14px 16px; }
.card-title { font-size:10px; font-weight:600; color:#9CA3AF; text-transform:uppercase;
              letter-spacing:.06em; margin-bottom:10px; }
.modules-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:12px; }
.sec-title { font-size:11px; font-weight:600; color:#6B7280; text-transform:uppercase;
             letter-spacing:.05em; margin:18px 0 10px;
             border-bottom:1px solid #EAECF0; padding-bottom:7px; }
.bt-table { width:100%; border-collapse:collapse; font-size:12px; }
.bt-table th { font-size:10px; font-weight:600; color:#9CA3AF; padding:6px 10px;
               text-align:left; border-bottom:1px solid #EAECF0; }
.bt-table td { padding:8px 10px; border-bottom:1px solid #F3F4F6; color:#374151; }
.bt-stress { background:#FFF5F5; }
.bt-norm { background:#F0FDF4; }
.formula-box { background:#F9FAFB; border:1px solid #EAECF0; border-radius:8px;
               padding:12px 16px; font-family:monospace; font-size:12px;
               color:#374151; margin-top:10px; }
.qbtn { font-size:11px; padding:5px 11px; border-radius:6px; background:#F3F4F6;
        border:1px solid #E5E7EB; color:#374151; cursor:pointer; margin:0 4px 4px 0;
        font-family:inherit; transition:background .12s; }
.qbtn:hover { background:#E5E7EB; }
.chat-box { background:#F9FAFB; border:1px solid #EAECF0; border-radius:12px;
            padding:14px; min-height:220px; max-height:380px; overflow-y:auto;
            margin-bottom:10px; }
.msg-label { font-size:11px; font-weight:600; color:#9CA3AF; margin-bottom:4px; margin-top:16px; }
.msg-label:first-child { margin-top:0; }
.msg-user { background:linear-gradient(135deg,#185FA5,#378ADD); color:#fff; border-radius:16px 16px 4px 16px;
            padding:12px 18px; font-size:14px; line-height:1.7; margin-left:80px; box-shadow:0 2px 8px rgba(24,95,165,0.15); }
.msg-bot { background:#fff; border:1px solid #EAECF0; color:#374151; border-radius:16px 16px 16px 4px;
            padding:12px 18px; font-size:14px; line-height:1.7; margin-right:80px; box-shadow:0 2px 6px rgba(0,0,0,0.05); }
.inp-row { display:flex; gap:8px; }
.inp-row input { flex:1; padding:9px 13px; border:1px solid #D1D5DB; border-radius:8px;
                 font-size:12px; font-family:inherit; color:#374151;
                 background:#fff; outline:none; }
.inp-row input:focus { border-color:#378ADD; box-shadow:0 0 0 3px #EBF4FF; }
.send-btn { padding:9px 18px; background:#185FA5; color:#fff; border:none;
            border-radius:8px; font-size:12px; font-family:inherit;
            font-weight:500; cursor:pointer; }
.send-btn:hover { background:#0C447C; }
.clear-btn { padding:9px 14px; background:#F3F4F6; color:#6B7280; border:1px solid #E5E7EB;
             border-radius:8px; font-size:12px; font-family:inherit; cursor:pointer; }
"""

# базовые настройки всех графиков Plotly
PLOT_LAYOUT = dict(
    paper_bgcolor="#fff", plot_bgcolor="#fff",
    font=dict(family="Inter, sans-serif", color="#374151", size=11),
    margin=dict(l=44, r=16, t=36, b=36),
    xaxis=dict(gridcolor="#F3F4F6", showgrid=True, zeroline=False, linecolor="#EAECF0", linewidth=1),
    yaxis=dict(gridcolor="#F3F4F6", showgrid=True, zeroline=False),
    legend=dict(bgcolor="#fff", bordercolor="#EAECF0", borderwidth=1, font=dict(size=11)),
    hovermode="x unified",
)

# подсвечиваю стресс-эпизоды на графиках красными прямоугольниками
def stress_shapes():
    shapes = []
    for _, (s, e) in STRESS_EPISODES.items():
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=s, x1=e, y0=0, y1=1,
            fillcolor="rgba(226,75,74,0.06)",
            line=dict(color="rgba(226,75,74,0.25)", width=1)
        ))
    return shapes


# график LSI с зонами и маркерами стресса
def fig_lsi(df, dr=None):
    if dr and dr[0] and dr[1]:
        df = df[(df["date"] >= dr[0]) & (df["date"] <= dr[1])]
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=40, fillcolor="rgba(29,158,117,0.06)", line_width=0)
    fig.add_hrect(y0=40, y1=70, fillcolor="rgba(186,117,23,0.06)", line_width=0)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(226,75,74,0.06)", line_width=0)
    fig.add_hline(y=40, line=dict(color="#EF9F27", width=1, dash="dot"))
    fig.add_hline(y=70, line=dict(color="#E24B4A", width=1, dash="dot"))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["lsi"], mode="lines", name="LSI",
        line=dict(color="#185FA5", width=2),
        fill="tozeroy", fillcolor="rgba(55,138,221,0.08)"
    ))
    stress_mask = df["lsi"] >= 70
    if stress_mask.any():
        fig.add_trace(go.Scatter(
            x=df.loc[stress_mask, "date"], y=df.loc[stress_mask, "lsi"],
            mode="markers", name="Стресс",
            marker=dict(color="#E24B4A", size=4, opacity=0.7)
        ))
    layout = {**PLOT_LAYOUT,
        "height": 360,
        "title": dict(text="Liquidity Stress Index", font=dict(size=13)),
        "yaxis": dict(**PLOT_LAYOUT["yaxis"], range=[0, 105], title="LSI"),
        "shapes": stress_shapes()}
    fig.update_layout(**layout)
    return fig


# столбчатый график вклада каждого модуля в LSI
def fig_contributions(df, dr=None):
    if dr and dr[0] and dr[1]:
        df = df[(df["date"] >= dr[0]) & (df["date"] <= dr[1])]
    fig = go.Figure()
    for col, name, color in MODULE_META:
        if col in df.columns:
            fig.add_trace(go.Bar(
                x=df["date"], y=df[col].clip(0),
                name=name, marker_color=color, opacity=0.85
            ))
    layout = {**PLOT_LAYOUT,
        "barmode": "stack", "height": 260,
        "title": dict(text="Вклад модулей в LSI", font=dict(size=13)),
        "shapes": stress_shapes()}
    fig.update_layout(**layout)
    return fig


# график сезонного мультипликатора М4
def fig_m4_seasonal():
    df = SIGNALS["m4"].copy()
    df["date"] = pd.to_datetime(df["date"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["seasonal_factor"],
        mode="lines", line=dict(color="#E09B30", width=1.5),
        name="Seasonal Factor", fill="tozeroy",
        fillcolor="rgba(224,155,48,0.10)"
    ))
    fig.add_hline(y=1.0, line=dict(color="#94A3B8", width=1, dash="dash"))
    fig.update_layout(**{**PLOT_LAYOUT,
        "height": 220,
        "title": dict(text="М4 · Seasonal Factor (мультипликатор LSI)", font=dict(size=12)),
        "yaxis": dict(range=[0.9, 1.5]),
    })
    return fig


# график флагов налоговых периодов и концов кварталов
def fig_m4_flags():
    df = SIGNALS["m4"].copy()
    df["date"] = pd.to_datetime(df["date"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["tax_week_flag"],
        mode="lines", line=dict(color="#E09B30", width=1.5, shape="hv"),
        name="Налоговая неделя", fill="tozeroy",
        fillcolor="rgba(224,155,48,0.35)"
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["end_of_quarter_flag"],
        mode="lines", line=dict(color="#D4537E", width=1.5, shape="hv"),
        name="Конец квартала", fill="tozeroy",
        fillcolor="rgba(212,83,126,0.45)"
    ))
    fig.update_layout(**{**PLOT_LAYOUT,
        "height": 220,
        "title": dict(text="М4 · Флаги налоговых периодов", font=dict(size=12)),
        "yaxis": dict(tickvals=[0, 1], ticktext=["нет", "да"]),
    })
    return fig


# универсальный график для одного сигнала модуля
def fig_module(sig_key, y_col, title, color, height=220):
    df = SIGNALS[sig_key].copy()
    df["date"] = pd.to_datetime(df["date"])
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#EAECF0", width=1))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df[y_col], mode="lines",
        line=dict(color=color, width=1.5), name=y_col
    ))
    layout = {**PLOT_LAYOUT,
        "height": height,
        "title": dict(text=title, font=dict(size=12)),
        "shapes": stress_shapes()}
    fig.update_layout(**layout)
    return fig


# gauge с текущим значением LSI и подписью состояния
def fig_gauge(value, status):
    color = STATUS_COLOR.get(status, "#1D9E75")
    if value < 40:
        status_label = "Норма"
    elif value < 70:
        status_label = "Внимание"
    else:
        status_label = "Стресс"

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number=dict(font=dict(size=40, color=color, family="Inter")),
        domain=dict(x=[0, 1], y=[0.28, 1.0]),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(size=9, color="#9CA3AF"), tickwidth=1, tickcolor="#EAECF0"),
            bar=dict(color=color, thickness=0.22),
            bgcolor="#F4F6F8",
            bordercolor="#EAECF0", borderwidth=1,
            steps=[
                dict(range=[0, 40], color="rgba(29,158,117,0.30)"),
                dict(range=[40, 70], color="rgba(186,117,23,0.28)"),
                dict(range=[70, 100], color="rgba(226,75,74,0.28)"),
            ],
        )
    ))
    fig.add_annotation(
        x=0.5, y=0.1, xref="paper", yref="paper",
        showarrow=False,
        text=f"<b>{status_label}</b>",
        font=dict(size=15, color=color, family="Inter"),
    )
    fig.update_layout(
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        margin=dict(l=16, r=16, t=8, b=16), height=210,
        font=dict(family="Inter, sans-serif"),
    )
    return fig


# подключаюсь к Groq API для генерации комментариев
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def _get_proxies():
    for var in ["HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"]:
        val = os.environ.get(var)
        if val:
            return {"http": val, "https": val}
    return {}

# отправляю запрос к LLM с историей диалога
def _groq(system_msg, user_msg, history=None):
    msgs = [{"role": "system", "content": system_msg}]
    if history:
        msgs += [{"role": h["role"], "content": h["content"]} for h in history[-6:]]
    msgs.append({"role": "user", "content": user_msg})
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "max_tokens": 800, "temperature": 0.7, "messages": msgs},
            timeout=30,
            proxies=_get_proxies()
        )
        data = r.json()
        if "choices" not in data:
            err = data.get("error", {}).get("message", str(data))
            return f"Ошибка Groq [{r.status_code}]: {err}"
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Ошибка соединения: {e}"


# формирую промпт для автоматического комментария по текущему LSI
def llm_comment(lsi, status, contribs, flags, tax, quarter):
    system = ("Ты аналитик казначейства ПСБ (Промсвязьбанк). "
              "Пишешь профессиональные аналитические комментарии о рынке ликвидности. "
              "ВАЖНО: отвечай ТОЛЬКО на русском языке. Никаких других языков.")
    prompt = f"""Напиши аналитический комментарий (4–5 предложений).

Текущий LSI: {lsi:.1f}/100 — {status}
Вклад модулей: М1={contribs.get('contrib_m1', 0):.1f}, М2={contribs.get('contrib_m2', 0):.1f}, М3={contribs.get('contrib_m3', 0):.1f}, М4={contribs.get('contrib_m4', 0):.1f}, М5={contribs.get('contrib_m5', 0):.1f}
Активные флаги: {', '.join(flags) if flags else 'нет'}
Налоговая неделя: {'да' if tax else 'нет'} | Конец квартала: {'да' if quarter else 'нет'}

Объясни: что означает текущий LSI, какие модули доминируют и почему, чего ожидать."""
    return _groq(system, prompt)


# обрабатываю вопрос пользователя в чате
def llm_chat(user_msg, history, ctx):
    system = ("Ты аналитик казначейства ПСБ. Знаешь систему мониторинга ликвидности: "
              "модули М1 (резервы), М2 (репо ЦБ), М3 (ОФЗ), М4 (налоги), М5 (казначейство). "
              "LSI 0–100: до 40 норма, 40–70 внимание, выше 70 стресс. "
              "Знаешь историю денежного рынка РФ: кризис дек.2014 (ставка 17%), "
              "фев.2022 (санкции, LSI=100), авг.2023 (ослабление рубля). "
              "ВАЖНО: отвечай ТОЛЬКО на русском языке. Никаких других языков.")
    return _groq(system, f"Контекст системы: {ctx}\n\nВопрос: {user_msg}", history)


# карточка с метрикой для верхней строки дашборда
def metric_card(label, value, sub, top_color):
    return html.Div([
        html.Div(label, className="mc-label"),
        html.Div(value, className="mc-value"),
        html.Div(sub, className="mc-sub"),
    ], className="mc", style={"borderTopColor": top_color})


def section(text):
    return html.Div(text, className="sec-title")


app = dash.Dash(__name__, title="Система мониторинга ликвидности | ПСБ",
                suppress_callback_exceptions=True)

app.index_string = f"""
<!DOCTYPE html><html><head>
<title>Система мониторинга ликвидности | ПСБ</title>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
{{%metas%}}{{%favicon%}}{{%css%}}
<style>{GLOBAL_CSS}</style>
</head><body>{{%app_entry%}}
<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body></html>"""

status_clr = STATUS_COLOR.get(LAST_STATUS, "#1D9E75")
status_bg = STATUS_BG.get(LAST_STATUS, "#E1F5EE")
Lsi_ctx = (f"LSI={LAST_LSI}, статус={LAST_STATUS}, "
           f"флаги: {', '.join(active_flags) if active_flags else 'нет'}. "
           f"Данные 2014–2024. Эпизоды: дек.2014 (17%), фев.2022 (санкции), авг.2023 (рубль).")

# описываю структуру страницы: шапка, вкладки, контент
app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div(className="hdr-dot"),
            html.Div([
                html.Div("Система мониторинга ликвидности", className="hdr-title"),
                html.Div(f"ПСБ Казначейство · данные по {LAST_DATE}", className="hdr-sub"),
            ])
        ], className="hdr-brand"),
        html.Span(LAST_STATUS, className="status-badge",
                  style={"background": status_bg, "color": status_clr, "borderColor": status_clr + "44"}),
    ], className="hdr"),

    html.Div([
        html.Button("Дашборд", id="tab-0", className="tab-btn active", n_clicks=0),
        html.Button("Модули", id="tab-1", className="tab-btn", n_clicks=0),
        html.Button("Бэктест", id="tab-2", className="tab-btn", n_clicks=0),
        html.Button("Цифровой помощник", id="tab-3", className="tab-btn", n_clicks=0),
    ], className="tabs-bar"),

    html.Div(id="page-content", className="page-body"),

    dcc.Store(id="active-tab", data=0),
    dcc.Store(id="chat-history", data=[]),
    dcc.Store(id="lsi-ctx", data=Lsi_ctx),
])


# переключаю активную вкладку по клику на кнопку
@app.callback(
    [Output("active-tab", "data"),
     Output("tab-0", "className"), Output("tab-1", "className"),
     Output("tab-2", "className"), Output("tab-3", "className")],
    [Input("tab-0", "n_clicks"), Input("tab-1", "n_clicks"),
     Input("tab-2", "n_clicks"), Input("tab-3", "n_clicks")],
    prevent_initial_call=True
)
def switch_tab(n0, n1, n2, n3):
    tid = callback_context.triggered[0]["prop_id"].split(".")[0]
    idx = int(tid.split("-")[1])
    cls = ["tab-btn active" if i == idx else "tab-btn" for i in range(4)]
    return idx, *cls


# рендерю содержимое нужной вкладки
@app.callback(Output("page-content", "children"), Input("active-tab", "data"))
def render(tab):

    if tab == 0:
        # дашборд: метрики, алерт, графики LSI и вклада модулей
        m1s = round(float(last_row.get("m1_signal", 0)), 2)
        m2s = round(float(last_row.get("m2_signal", 0)), 2)
        m3s = round(float(last_row.get("m3_signal", 0)), 2)
        m5s = round(float(last_row.get("m5_signal", 0)), 2)

        alert_color = "#FAEEDA" if active_flags else "#E1F5EE"
        alert_border = "#FAC775" if active_flags else "#9FE1CB"
        alert_text = "#633806" if active_flags else "#085041"
        alert_icon = "⚠️" if active_flags else "✅"
        alert_msg = ", ".join(active_flags) if active_flags else "Активных сигналов стресса нет"

        return html.Div([
            html.Div([
                metric_card("LSI сейчас", str(LAST_LSI), LAST_STATUS, status_clr),
                metric_card("М1 Резервы", str(m1s), "MAD-сигнал", "#7F77DD"),
                metric_card("М2 Репо ЦБ", str(m2s), "MAD-сигнал", "#1D9E75"),
                metric_card("М3 ОФЗ", str(m3s), "MAD-сигнал", "#D4537E"),
                metric_card("М5 Казначейство", str(m5s), "MAD-сигнал", "#378ADD"),
            ], className="metrics-row"),

            html.Div([
                html.Span(f"{alert_icon} ", style={"marginRight": "4px"}),
                html.Span(alert_msg, style={"color": alert_text}),
            ], className="alert-bar",
               style={"background": alert_color, "borderColor": alert_border, "color": alert_text}),

            html.Div([
                html.Span("Период: ", style={"fontSize": "12px", "color": "#6B7280", "marginRight": "10px"}),
                dcc.DatePickerRange(
                    id="date-range", display_format="DD.MM.YYYY",
                    start_date="2014-01-01", end_date="2024-12-31",
                    style={"fontSize": "12px"}
                )
            ], style={"marginBottom": "12px", "display": "flex", "alignItems": "center"}),

            html.Div([
                html.Div([
                    dcc.Graph(id="lsi-chart", figure=fig_lsi(DF_LSI), config={"displayModeBar": False})
                ], className="card", style={"flex": "3", "padding": "12px 14px"}),

                html.Div([
                    html.Div("Текущий уровень", className="card-title",
                             style={"textAlign": "center", "paddingTop": "8px"}),
                    dcc.Graph(figure=fig_gauge(LAST_LSI, LAST_STATUS), config={"displayModeBar": False}),
                    html.Div([
                        html.P(f"● {l}", style={"color": c, "fontSize": "16px", "margin": "0", "whiteSpace": "nowrap"})
                        for c, l in [
                            ("#1D9E75", "0–40 Норма"),
                            ("#BA7517", "40–70 Внимание"),
                            ("#E24B4A", "70–100 Стресс")
                        ]
                    ], style={"padding": "16px", "display": "flex", "flexDirection": "column",
                              "gap": "8px", "borderTop": "1px solid #EAECF0", "marginTop": "8px"})
                ], className="card", style={"flex": "1"}),
            ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),

            html.Div([
                dcc.Graph(id="contrib-chart", figure=fig_contributions(DF_LSI), config={"displayModeBar": False})
            ], className="card"),

            section("Автоматический комментарий (LLM)"),
            html.Div([
                html.Div(id="auto-comment", style={"flex": 1},
                         children=html.P("Нажмите кнопку для получения аналитического комментария.",
                                         style={"color": "#9CA3AF", "fontStyle": "italic", "fontSize": "13px", "margin": 0})),
                html.Button("Обновить", id="btn-comment", n_clicks=0, className="send-btn",
                            style={"alignSelf": "flex-start", "whiteSpace": "nowrap"}),
            ], className="card", style={"display": "flex", "gap": "20px", "alignItems": "flex-start", "marginTop": "12px"}),
        ])

    elif tab == 1:
        # вкладка модулей: графики сигналов по каждому модулю
        return html.Div([
            section("М1 — Усреднение обязательных резервов"),
            html.Div([
                html.Div(dcc.Graph(figure=fig_module("m1", "mad_score_spread", "MAD-score спреда резервов", "#7F77DD"), config={"displayModeBar": False}), className="card"),
                html.Div(dcc.Graph(figure=fig_module("m1", "mad_score_ruonia", "MAD-score RUONIA", "#7F77DD"), config={"displayModeBar": False}), className="card"),
            ], className="modules-grid"),

            section("М2 — Аукционы репо ЦБ"),
            html.Div([
                html.Div(dcc.Graph(figure=fig_module("m2", "mad_score_cover", "MAD-score cover ratio репо", "#1D9E75"), config={"displayModeBar": False}), className="card"),
                html.Div(dcc.Graph(figure=fig_module("m2", "mad_score_rate_spread", "MAD-score спреда ставки", "#1D9E75"), config={"displayModeBar": False}), className="card"),
            ], className="modules-grid"),

            section("М3 — Размещение ОФЗ"),
            html.Div([
                html.Div(dcc.Graph(figure=fig_module("m3", "mad_score_cover", "MAD-score cover ratio ОФЗ (инверт.)", "#D4537E"), config={"displayModeBar": False}), className="card"),
                html.Div(dcc.Graph(figure=fig_module("m3", "mad_score_yield_spread", "MAD-score спреда доходности", "#D4537E"), config={"displayModeBar": False}), className="card"),
            ], className="modules-grid"),

            section("М4 — Налоговый период и сезонность"),
            html.Div([
                html.Div(dcc.Graph(figure=fig_m4_seasonal(), config={"displayModeBar": False}), className="card"),
                html.Div(dcc.Graph(figure=fig_m4_flags(), config={"displayModeBar": False}), className="card"),
            ], className="modules-grid"),

            section("М5 — Федеральное казначейство"),
            html.Div([
                html.Div(dcc.Graph(figure=fig_module("m5", "mad_score_structural", "MAD-score структурного баланса", "#378ADD"), config={"displayModeBar": False}), className="card"),
                html.Div(dcc.Graph(figure=fig_module("m5", "mad_score_delta", "MAD-score оттока казначейства", "#378ADD"), config={"displayModeBar": False}), className="card"),
            ], className="modules-grid"),
        ])

    elif tab == 2:
        # бэктест: проверяю, правильно ли модель реагировала на исторические кризисы
        bt_rows = []
        for _, row in BACKTEST.iterrows():
            is_stress = row["тип"] == "СТРЕСС"
            cls = "bt-stress" if is_stress else "bt-norm"
            det = (f"{row.get('детекция_%', '—')}% дней >60"
                   if is_stress else f"{row.get('специфичность_%', '—')}% дней <45")
            bt_rows.append(html.Tr([
                html.Td(row["период"]),
                html.Td(row["avg_LSI"]),
                html.Td(row["max_LSI"]),
                html.Td(det),
            ], className=cls))

        sens_rows = []
        for _, row in SENSITIVITY.iterrows():
            clr = "#E24B4A" if row["delta_lsi"] > 0.5 else "#1D9E75" if row["delta_lsi"] < -0.5 else "#374151"
            sens_rows.append(html.Tr([
                html.Td(row["module"]),
                html.Td(row["weight_change"]),
                html.Td(f"{row['avg_lsi_base']:.1f}"),
                html.Td(f"{row['avg_lsi_new']:.1f}"),
                html.Td(f"{row['delta_lsi']:+.2f}", style={"color": clr, "fontWeight": "500"}),
            ]))

        return html.Div([
            section("Бэктест на исторических стресс-эпизодах"),
            html.Div([
                html.Div(s, style={"background": "#FFF5F5", "border": "1px solid #FCA5A5",
                    "borderRadius": "8px", "padding": "9px 12px", "fontSize": "12px", "color": "#7F1D1D"})
                for s in [
                    "Декабрь 2014: ЦБ поднял ключевую ставку с 10.5% до 17% за одну ночь",
                    "Февраль 2022: Введение беспрецедентных санкций, временный запрет торгов",
                    "Август 2023: Резкое ослабление рубля, ЦБ экстренно поднял ставку до 12%",
                ]
            ], style={"display": "flex", "flexDirection": "column", "gap": "6px", "marginBottom": "14px"}),

            html.Div(html.Table([
                html.Thead(html.Tr([html.Th("Период"), html.Th("Avg LSI"), html.Th("Max LSI"), html.Th("Результат")])),
                html.Tbody(bt_rows),
            ], className="bt-table"), className="card", style={"marginBottom": "16px"}),

            section("Анализ чувствительности весов (±20%)"),
            html.Div(html.Table([
                html.Thead(html.Tr([html.Th("Модуль"), html.Th("Изменение"), html.Th("LSI базовый"), html.Th("LSI новый"), html.Th("Δ LSI")])),
                html.Tbody(sens_rows),
            ], className="bt-table"), className="card", style={"marginBottom": "16px"}),

            section("Обоснование метода агрегации"),
            html.Div([
                html.P([html.Strong("Метод: GradientBoostingRegressor с псевдо-разметкой.")],
                       style={"marginBottom": "10px", "fontSize": "13px"}),
                html.Ul([
                    html.Li("Нелинейные взаимодействия: модель улавливает совместный эффект модулей"),
                    html.Li("Интерпретируемость: вклад модулей через feature_importances_"),
                    html.Li("М4 входит как мультипликатор (1.0–1.4) — устраняет двойной счёт"),
                ], style={"fontSize": "12px", "color": "#6B7280", "lineHeight": "2", "paddingLeft": "18px"}),
                html.Div("LSI = GBM(M1, M2, M3, M5, флаги) × Seasonal_Factor(M4)", className="formula-box"),
            ], className="card"),
        ])

    elif tab == 3:
        # цифровой помощник: чат с LLM про рынок ликвидности
        return html.Div([
            html.Div([
                html.Div([
                    html.Div("ЦП", style={"width": "42px", "height": "42px", "borderRadius": "12px",
                                          "background": "linear-gradient(135deg,#185FA5,#378ADD)",
                                          "display": "flex", "alignItems": "center", "justifyContent": "center",
                                          "color": "#fff", "fontWeight": "700", "fontSize": "13px", "flexShrink": "0"}),
                    html.Div([
                        html.Div("Цифровой помощник", style={"fontSize": "18px", "fontWeight": "700", "color": "#1a1a2e"}),
                        html.Div("Отвечает на вопросы о рынке ликвидности", style={"fontSize": "13px", "color": "#6B7280", "marginTop": "2px"}),
                    ]),
                ], style={"display": "flex", "alignItems": "center", "gap": "14px"}),
            ], style={"marginBottom": "16px", "paddingBottom": "16px", "borderBottom": "1px solid #EAECF0"}),

            html.Div([
                html.Div("Быстрые вопросы:", style={"fontSize": "12px", "color": "#9CA3AF", "marginBottom": "10px", "fontWeight": "600", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
                html.Div([
                    html.Button(q, id={"type": "qbtn", "index": i}, className="qbtn", n_clicks=0,
                                style={"fontSize": "13px", "padding": "8px 16px", "borderRadius": "20px"})
                    for i, q in enumerate([
                        "Что происходило в феврале 2022?",
                        "Почему в декабре 2014 был стресс?",
                        "Что означает текущий LSI?",
                        "Как налоговая неделя влияет на репо ЦБ?",
                        "Чего ожидать при текущих сигналах?",
                    ])
                ], style={"display": "flex", "flexWrap": "wrap", "gap": "8px"})
            ], className="card", style={"marginBottom": "16px", "background": "#F9FAFB"}),

            dcc.Loading(type="circle", color="#185FA5", children=[
                html.Div(id="chat-messages", className="chat-box",
                         style={"minHeight": "400px", "maxHeight": "500px", "padding": "24px",
                                "background": "#F9FAFB", "border": "1px solid #EAECF0", "borderRadius": "16px"},
                         children=[
                             html.Div("Цифровой помощник", className="msg-label"),
                             html.Div("Добро пожаловать в систему мониторинга ликвидности. "
                                      "Задайте вопрос о рынке ликвидности, выберите готовый "
                                      "вопрос выше или спросите о конкретном периоде.",
                                      className="msg-bot"),
                         ]),
            ]),

            html.Div([
                dcc.Input(id="chat-input", type="text", debounce=False,
                          placeholder="Напишите вопрос...",
                          style={"flex": "1", "padding": "8px 12px", "border": "1px solid #D1D5DB",
                                 "borderRadius": "8px", "fontSize": "13px", "fontFamily": "Inter,sans-serif",
                                 "color": "#374151", "background": "#fff", "outline": "none",
                                 "lineHeight": "normal", "boxSizing": "border-box"}),
                html.Button("Отправить", id="chat-send", n_clicks=0, className="send-btn"),
                html.Button("Очистить", id="chat-clear", n_clicks=0, className="clear-btn"),
            ], style={"display": "flex", "gap": "10px", "marginTop": "14px"}),
        ])

    return html.Div("404")


# обновляю графики при изменении диапазона дат
@app.callback(
    [Output("lsi-chart", "figure"), Output("contrib-chart", "figure")],
    [Input("date-range", "start_date"), Input("date-range", "end_date")],
    prevent_initial_call=True
)
def update_charts(s, e):
    return fig_lsi(DF_LSI, (s, e)), fig_contributions(DF_LSI, (s, e))


# генерирую автоматический LLM-комментарий по кнопке
@app.callback(Output("auto-comment", "children"), Input("btn-comment", "n_clicks"), prevent_initial_call=True)
def update_comment(_):
    contribs = {c: float(last_row.get(c, 0)) for c, _, _ in MODULE_META}
    text = llm_comment(LAST_LSI, LAST_STATUS, contribs, active_flags,
                       bool(last_row.get("tax_week_flag", 0)),
                       bool(last_row.get("end_of_quarter_flag", 0)))
    return html.P(text, style={"fontSize": "13px", "lineHeight": "1.7", "color": "#374151", "margin": 0})


# обрабатываю сообщения в чате: отправку, очистку и быстрые вопросы
@app.callback(
    [Output("chat-messages", "children"),
     Output("chat-history", "data"),
     Output("chat-input", "value")],
    [Input("chat-send", "n_clicks"),
     Input("chat-clear", "n_clicks"),
     Input({"type": "qbtn", "index": dash.ALL}, "n_clicks")],
    [State("chat-input", "value"),
     State("chat-history", "data"),
     State("lsi-ctx", "data")],
    prevent_initial_call=True
)
def handle_chat(send, clear, qbtns, user_input, history, ctx_data):
    trig = callback_context.triggered[0]["prop_id"]
    history = history or []

    if "chat-clear" in trig:
        return _render_chat([]), [], ""

    quick_qs = [
        "Что происходило в феврале 2022?",
        "Почему в декабре 2014 был стресс?",
        "Что означает текущий LSI?",
        "Как налоговая неделя влияет на репо ЦБ?",
        "Чего ожидать при текущих сигналах?",
    ]
    if "qbtn" in trig:
        import json
        idx = json.loads(trig.split(".")[0])["index"]
        user_input = quick_qs[idx]

    if not user_input or not user_input.strip():
        return _render_chat(history), history, ""

    msg = user_input.strip()
    history.append({"role": "user", "content": msg})
    answer = llm_chat(msg, history, ctx_data or "")
    history.append({"role": "assistant", "content": answer})
    return _render_chat(history), history, ""


# рендерю историю чата в виде пузырей
def _render_chat(history):
    if not history:
        return [
            html.Div("Аналитик", className="msg-label"),
            html.Div("Чат очищен. Задайте новый вопрос.", className="msg-bot"),
        ]
    els = []
    for msg in history:
        is_u = msg["role"] == "user"
        els.append(html.Div("Вы" if is_u else "Цифровой помощник", className="msg-label"))
        els.append(html.Div(msg["content"], className="msg-user" if is_u else "msg-bot"))
    return els


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
