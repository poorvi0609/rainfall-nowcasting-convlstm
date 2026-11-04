"""Rain Watch: short-range rain warnings for the Maharashtra coast.

Shows the saved predictions of the heavy-rain-weighted ConvLSTM on the test period.
Run from the project folder with:  streamlit run src/dashboard.py
"""
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------- settings you can change ----------
RESULTS_DIR = "results"
RUN = "convlstm_adam_lr0.001_do0.0_l20.0_hw5.0"

# ---------- facts about the dataset (from build_dataset.py / make_windows.py) ----------
START_TIME = datetime(2024, 7, 1, 0, 0)    # UTC time of the very first frame
TOTAL_FRAMES = 31 * 48                      # 1 to 31 July, one frame every 30 minutes
INPUT_STEPS, OUTPUT_STEPS, GAP = 6, 2, 8
N_WINDOWS = TOTAL_FRAMES - INPUT_STEPS - OUTPUT_STEPS + 1
FIRST_TEST = int(N_WINDOWS * 0.85) + GAP    # index of the first test window
LAT0, LON0, PIXEL = 16.05, 70.05, 0.1       # centre of pixel (0, 0) and pixel size
LATS = LAT0 + PIXEL * np.arange(60)
LONS = LON0 + PIXEL * np.arange(60)

# ---------- warning levels ----------
LEVELS = [
    dict(name="No rain", min=0.0, colour="#607d8b", ink="#ffffff", emoji="✅",
         advice="Dry, or barely any rain", range="under 1 mm/hr"),
    dict(name="Light", min=1.0, colour="#43a047", ink="#ffffff", emoji="🌦️",
         advice="A light jacket is enough", range="1 to 5 mm/hr"),
    dict(name="Moderate", min=5.0, colour="#fbc02d", ink="#1a1a1a", emoji="🌧️",
         advice="Carry an umbrella", range="5 to 10 mm/hr"),
    dict(name="Heavy", min=10.0, colour="#ef6c00", ink="#ffffff", emoji="⛈️",
         advice="Travel with care, watch low-lying roads", range="10 to 20 mm/hr"),
    dict(name="Severe", min=20.0, colour="#d32f2f", ink="#ffffff", emoji="🚨",
         advice="Avoid travel if you can", range="20+ mm/hr"),
]

CITY_COORDS = {   # name: (latitude, longitude)
    "Mumbai": (19.08, 72.88),
    "Pune": (18.52, 73.86),
    "Nashik": (20.00, 73.79),
    "Surat": (21.17, 72.83),
    "Ratnagiri": (16.99, 73.30),
    "Kolhapur": (16.70, 74.24),
}

# Map colours line up with the alert levels (green = light, yellow = moderate, ...)
RAIN_SCALE = [
    [0.0, "#0e1f35"], [0.033, "#1f5f8b"], [0.10, "#43a047"], [0.167, "#fbc02d"],
    [0.333, "#ef6c00"], [0.667, "#d32f2f"], [1.0, "#7b1fa2"],
]
COLORAXIS = dict(colorscale=RAIN_SCALE, cmin=0, cmax=30,
                 colorbar=dict(title="mm/hr", thickness=12, len=0.85))
HOVER = "Lat %{y:.2f}, Lon %{x:.2f}<br>Rain: %{z:.1f} mm/hr<extra></extra>"
PLOT_BG = "#0b1220"


def to_pixel(lat, lon):
    row = int(round((lat - LAT0) / PIXEL))
    col = int(round((lon - LON0) / PIXEL))
    return min(max(row, 0), 59), min(max(col, 0), 59)


CITIES = {name: (lat, lon) + to_pixel(lat, lon) for name, (lat, lon) in CITY_COORDS.items()}


def city_rain(frame, row, col):
    """Highest rain rate in the 3x3 pixels around a city (about 30 km square)."""
    return float(frame[max(row - 1, 0):row + 2, max(col - 1, 0):col + 2].max())


def level_of(rain):
    chosen = LEVELS[0]
    for level in LEVELS:
        if rain >= level["min"]:
            chosen = level
    return chosen


def issue_time(k):
    """UTC time of the last input frame of test window k (when the forecast is made)."""
    frame = FIRST_TEST + k + INPUT_STEPS - 1
    return START_TIME + timedelta(minutes=30 * frame)


def to_ist(t):
    return t + timedelta(hours=5, minutes=30)


def ist_label(t):
    return f"{to_ist(t):%a %d %b, %I:%M %p}"


@st.cache_data
def load(run):
    pred = np.load(f"{RESULTS_DIR}/{run}_pred.npy")[..., 0]    # (windows, 2, 60, 60) mm/hr
    truth = np.load(f"{RESULTS_DIR}/{run}_truth.npy")[..., 0]
    return pred, truth


@st.cache_data
def city_series(run, step):
    """For every city: predicted and observed rain (3x3 max) for all test windows."""
    pred, truth = load(run)
    series = {}
    for name, (_, _, row, col) in CITIES.items():
        area = (slice(None), step, slice(max(row - 1, 0), row + 2), slice(max(col - 1, 0), col + 2))
        series[name] = (pred[area].max(axis=(1, 2)), truth[area].max(axis=(1, 2)))
    return series


def nearest_city(lat, lon):
    def km(item):
        c_lat, c_lon = item[1]
        return 111 * np.hypot(c_lat - lat, (c_lon - lon) * 0.945)
    name, coords = min(CITY_COORDS.items(), key=km)
    return name, km((name, coords))


# ---------- plotting helpers ----------
def rain_heatmap(z):
    return go.Heatmap(z=z, x=LONS, y=LATS, coloraxis="coloraxis", hovertemplate=HOVER)


def city_dots(frame=None):
    lats, lons, names, colours, rains = [], [], [], [], []
    for name, (lat, lon, row, col) in CITIES.items():
        rain = city_rain(frame, row, col) if frame is not None else 0.0
        lats.append(lat)
        lons.append(lon)
        names.append(name)
        colours.append(level_of(rain)["colour"] if frame is not None else "#ffffff")
        rains.append(rain)
    return go.Scatter(
        x=lons, y=lats, mode="markers+text", text=names, textposition="top right",
        textfont=dict(color="white", size=12), customdata=rains, showlegend=False,
        marker=dict(size=15, color=colours, line=dict(color="white", width=2)),
        hovertemplate="%{text}: %{customdata:.1f} mm/hr<extra></extra>")


def style(fig, height):
    fig.update_layout(template="plotly_dark", paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                      height=height, margin=dict(l=10, r=10, t=50, b=10),
                      coloraxis=COLORAXIS, font=dict(size=13))


def map_figure(frame):
    fig = go.Figure([rain_heatmap(frame), city_dots(frame)])
    style(fig, 580)
    fig.update_xaxes(range=[70, 76], title="Longitude", showgrid=False)
    fig.update_yaxes(range=[16, 22], title="Latitude", showgrid=False, scaleanchor="x")
    fig.add_annotation(x=70.6, y=18.0, text="Arabian Sea", showarrow=False,
                       font=dict(color="rgba(255,255,255,.55)", size=14))
    return fig


def replay_figure(pred, truth, start, n_frames=12):
    ks = list(range(start, min(start + n_frames, len(pred))))
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.06,
                        subplot_titles=("What actually happened", "Model forecast"))
    fig.add_trace(rain_heatmap(truth[ks[0], 0]), row=1, col=1)
    fig.add_trace(rain_heatmap(pred[ks[0], 0]), row=1, col=2)
    fig.add_trace(city_dots(), row=1, col=1)
    fig.add_trace(city_dots(), row=1, col=2)
    labels = [ist_label(issue_time(kk) + timedelta(minutes=30)) for kk in ks]
    fig.frames = [go.Frame(name=lab, traces=[0, 1],
                           data=[rain_heatmap(truth[kk, 0]), rain_heatmap(pred[kk, 0])])
                  for kk, lab in zip(ks, labels)]
    style(fig, 520)
    fig.update_xaxes(range=[70, 76], showgrid=False)
    fig.update_yaxes(range=[16, 22], showgrid=False)
    fig.update_yaxes(scaleanchor="x", row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", row=1, col=2)
    play = dict(frame=dict(duration=700, redraw=True), fromcurrent=True, transition=dict(duration=0))
    pause = dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=70),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.02, xanchor="left",
                          yanchor="top", pad=dict(t=10),
                          buttons=[dict(label="▶ Play", method="animate", args=[None, play]),
                                   dict(label="⏸ Pause", method="animate", args=[[None], pause])])],
        sliders=[dict(active=0, x=0.18, len=0.8, y=-0.02, yanchor="top", pad=dict(t=10),
                      currentvalue=dict(prefix="Time (IST): "),
                      steps=[dict(method="animate", label=lab[-8:],
                                  args=[[lab], dict(mode="immediate", transition=dict(duration=0),
                                                    frame=dict(duration=0, redraw=True))])
                             for lab in labels])])
    return fig


def timeline_figure(series_city, times, k, lead):
    pred_s, true_s = series_city
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=true_s, name="Actually recorded", fill="tozeroy",
                             line=dict(color="#90caf9", width=1.5),
                             fillcolor="rgba(144,202,249,0.25)"))
    fig.add_trace(go.Scatter(x=times, y=pred_s, name=f"Model forecast ({lead} min ahead)",
                             line=dict(color="#ffb74d", width=3)))
    fig.add_hline(y=5, line_dash="dash", line_color="rgba(255,255,255,.5)",
                  annotation_text="Moderate rain (5 mm/hr)", annotation_position="top left")
    now = times[k]
    fig.add_shape(type="line", x0=now, x1=now, y0=0, y1=1, yref="paper",
                  line=dict(color="#ef5350", width=2, dash="dot"))
    style(fig, 420)
    fig.update_layout(coloraxis=None, margin=dict(l=10, r=10, t=20, b=10),
                      legend=dict(orientation="h", y=1.12), hovermode="x unified")
    fig.update_yaxes(title="Rain rate (mm/hr)", gridcolor="rgba(255,255,255,.1)")
    fig.update_xaxes(title="Time (IST)", showgrid=False)
    return fig


# ---------- scoring ----------
def score_table(series, threshold):
    rows, totals = [], np.zeros(3, dtype=int)
    for name in CITIES:
        p, o = series[name][0] >= threshold, series[name][1] >= threshold
        hits, misses, alarms = int((p & o).sum()), int((~p & o).sum()), int((p & ~o).sum())
        totals += [hits, misses, alarms]
        rows.append(score_row(name, hits, misses, alarms))
    total_row = score_row("All six cities", *totals)
    return pd.DataFrame(rows + [total_row]), total_row


def score_row(name, hits, misses, alarms):
    def ratio(a, b):
        return round(a / b, 2) if b > 0 else float("nan")
    return {
        "City": name,
        "Warnings that came true": int(hits),
        "Missed storms": int(misses),
        "False alarms": int(alarms),
        "Storms caught (POD)": ratio(hits, hits + misses),
        "Wrong warnings (FAR)": ratio(alarms, hits + alarms),
        "Overall score (CSI)": ratio(hits, hits + misses + alarms),
    }


def pct(x):
    return "n/a" if x != x else f"{x:.0%}"


def join_names(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1250px;}
footer {visibility: hidden;}
.hero {position: relative; overflow: hidden; border-radius: 18px; padding: 26px 32px;
       margin-bottom: 18px; color: #fff;
       background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);}
.hero::before {content: ""; position: absolute; inset: 0; opacity: .30;
       background-image: linear-gradient(100deg, transparent 46%, rgba(255,255,255,.55) 50%, transparent 54%);
       background-size: 26px 64px; animation: fall .8s linear infinite;}
.hero > * {position: relative; z-index: 1;}
.hero-title {font-size: 2.3rem; font-weight: 800; letter-spacing: .5px;}
.hero-sub {font-size: 1.05rem; opacity: .92; margin-top: 4px;}
.hero-tag {display: inline-block; margin-top: 12px; padding: 4px 12px; border-radius: 20px;
       background: rgba(255,255,255,.16); font-size: .85rem;}
@keyframes fall {from {background-position: 0 0;} to {background-position: -26px 64px;}}
.status {border-radius: 14px; padding: 14px 20px; margin: 6px 0 16px; font-size: 1.15rem;
       font-weight: 600; border-left: 8px solid var(--c);}
.grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
       gap: 14px; margin-bottom: 18px;}
.card {background: linear-gradient(160deg, #101a2e 0%, #0b1220 100%); border-radius: 16px;
       padding: 14px 16px; color: #fff; border-top: 5px solid var(--c);
       box-shadow: 0 4px 14px rgba(0,0,0,.25);}
.card.pulse {animation: pulse 1.6s infinite;}
.card-top {display: flex; justify-content: space-between; align-items: center;
       font-size: 1.1rem; font-weight: 700;}
.chip {display: inline-block; margin: 8px 0 2px; padding: 2px 10px; border-radius: 12px;
       font-size: .8rem; font-weight: 700; background: var(--c);}
.rain {font-size: 1.9rem; font-weight: 800; line-height: 1.1;}
.rain span {font-size: .85rem; font-weight: 500; opacity: .75;}
.advice {font-size: .88rem; opacity: .9; margin-top: 4px;}
.actual {font-size: .75rem; opacity: .6; margin-top: 8px;}
.dot {display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px;}
.legend-row {font-size: .88rem; margin: 5px 0;}
@keyframes pulse {0% {box-shadow: 0 0 0 0 var(--c);} 70% {box-shadow: 0 0 0 12px transparent;}
       100% {box-shadow: 0 0 0 0 transparent;}}
</style>
"""

# ======================= page =======================
st.set_page_config(page_title="Rain Watch", page_icon="🌧️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

if not os.path.exists(f"{RESULTS_DIR}/{RUN}_pred.npy"):
    st.error(f"Could not find {RESULTS_DIR}/{RUN}_pred.npy. Run this from the project folder.")
    st.stop()

pred, truth = load(RUN)
n = pred.shape[0]
if n != N_WINDOWS - FIRST_TEST:
    st.warning(f"Expected {N_WINDOWS - FIRST_TEST} test windows but found {n}. Times may be off.")

first_day, last_day = to_ist(issue_time(0)), to_ist(issue_time(n - 1))
st.markdown(
    f"""<div class="hero"><div class="hero-title">🌧️ Rain Watch</div>
    <div class="hero-sub">Rain warnings for the next hour along the Maharashtra coast, made by an AI model</div>
    <div class="hero-tag">Demo replay of {first_day:%d %b} to {last_day:%d %b %Y} · all times in IST</div></div>""",
    unsafe_allow_html=True)

# ---------- controls ----------
col_time, col_lead, col_jump = st.columns([3, 2, 2])
with col_lead:
    lead = st.radio("Looking ahead", [30, 60], format_func=lambda m: f"Next {m} min", horizontal=True)
step = 0 if lead == 30 else 1
series = city_series(RUN, step)


def stormiest():
    return int(np.argmax(sum(series[name][1] for name in CITIES)))


def jump():
    st.session_state.k = stormiest()


if "k" not in st.session_state:
    st.session_state.k = stormiest()
with col_jump:
    st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    st.button("⚡ Jump to the stormiest moment", on_click=jump)
with col_time:
    k = st.select_slider("Pick a moment", options=list(range(n)), key="k",
                         format_func=lambda i: ist_label(issue_time(i)))

target_time = issue_time(k) + timedelta(minutes=lead)
now_rain = {name: (series[name][0][k], series[name][1][k]) for name in CITIES}

# ---------- headline ----------
ranks = {name: LEVELS.index(level_of(now_rain[name][0])) for name in CITIES}
top = max(ranks.values())
top_level = LEVELS[top]
top_cities = [name for name in CITIES if ranks[name] == top]
when = f"{to_ist(target_time):%I:%M %p}"
if top >= 2:
    headline = f"{top_level['emoji']} {top_level['name']} rain expected near {join_names(top_cities)} by {when}"
elif top == 1:
    headline = f"🌦️ Light rain around {join_names(top_cities)} by {when}, nothing serious"
else:
    headline = f"✅ No significant rain forecast for the six cities by {when}"
st.markdown(f'<div class="status" style="--c:{top_level["colour"]}; background:{top_level["colour"]}26">'
            f'{headline}</div>', unsafe_allow_html=True)

# ---------- city cards ----------
cards = ""
for name in CITIES:
    p, o = now_rain[name]
    level = level_of(p)
    pulse = " pulse" if level["min"] >= 10 else ""
    cards += (f'<div class="card{pulse}" style="--c:{level["colour"]}">'
              f'<div class="card-top"><span>{name}</span><span>{level["emoji"]}</span></div>'
              f'<div class="chip" style="color:{level["ink"]}">{level["name"]}</div>'
              f'<div class="rain">{p:.1f}<span> mm/hr</span></div>'
              f'<div class="advice">{level["advice"]}</div>'
              f'<div class="actual">Actually recorded: {o:.1f} mm/hr</div></div>')
st.markdown(f'<div class="grid">{cards}</div>', unsafe_allow_html=True)

# ---------- tabs ----------
tab_map, tab_replay, tab_time, tab_rel, tab_about = st.tabs(
    ["🗺️ Rain map", "🎞️ Storm replay", "📈 City timeline", "✅ How reliable?", "ℹ️ About"])

with tab_map:
    left, right = st.columns([3, 1.3])
    with left:
        view = st.radio("Show", ["Model forecast", "What actually happened"], horizontal=True)
        frame = pred[k, step] if view == "Model forecast" else truth[k, step]
        st.plotly_chart(map_figure(frame))
    with right:
        peak = float(frame.max())
        row, col = np.unravel_index(int(np.argmax(frame)), frame.shape)
        near, dist = nearest_city(LATS[row], LONS[col])
        alerts = sum(level_of(city_rain(frame, r, c))["min"] >= 5 for _, _, r, c in CITIES.values())
        st.metric("Heaviest rain on the map", f"{peak:.1f} mm/hr")
        st.caption(f"About {dist:.0f} km from {near}")
        st.metric("Cities with moderate rain or worse", f"{alerts} of {len(CITIES)}")
        st.metric("Area with rain", f"{(frame >= 1).mean():.0%} of the map")
        st.markdown("**What the colours mean**")
        for level in LEVELS[1:]:
            st.markdown(f'<div class="legend-row"><span class="dot" style="background:{level["colour"]}">'
                        f'</span>{level["name"]} · {level["range"]}</div>', unsafe_allow_html=True)

with tab_replay:
    st.write("Watch six hours of rain unfold, observed on the left and forecast on the right. "
             "Press play, or drag the slider.")
    start = min(k, max(n - 12, 0))
    st.plotly_chart(replay_figure(pred, truth, start))
    st.caption("The replay starts at the moment picked above. The forecast on the right is each "
               "frame's 30-minute-ahead prediction.")

with tab_time:
    city = st.selectbox("City", list(CITIES), index=1)
    times = [to_ist(issue_time(i) + timedelta(minutes=lead)) for i in range(n)]
    st.plotly_chart(timeline_figure(series[city], times, k, lead))
    st.caption("The red dotted line is the moment you picked. The closer the orange line follows the "
               "blue area, the better the forecast.")

with tab_rel:
    options = {"Moderate rain or worse (5+ mm/hr)": 5, "Light rain or worse (1+ mm/hr)": 1,
               "Heavy rain or worse (10+ mm/hr)": 10}
    choice = st.selectbox("Which warnings should we check?", list(options))
    table, total = score_table(series, options[choice])
    a, b, c = st.columns(3)
    a.metric("Storms caught", pct(total["Storms caught (POD)"]),
             help="Of all the times that much rain really fell, how often the model had warned.")
    b.metric("Wrong warnings", pct(total["Wrong warnings (FAR)"]),
             help="Of all the warnings the model gave, how many were not followed by that much rain.")
    c.metric("Overall score (CSI)", f"{total['Overall score (CSI)']:.2f}"
             if total["Overall score (CSI)"] == total["Overall score (CSI)"] else "n/a",
             help="1.0 would be perfect. It counts hits, misses and false alarms together.")
    st.caption(f"Checked on {n} forecasts, {lead} minutes ahead, across the six cities. The test period "
               "is only about 4.5 days, so treat these as indicative, not final.")
    with st.expander("See the numbers for each city"):
        st.dataframe(table, hide_index=True)
        st.markdown(
            "- **Warnings that came true**: the model warned and that much rain fell.\n"
            "- **Missed storms**: that much rain fell but there was no warning.\n"
            "- **False alarms**: the model warned but the rain did not come.\n"
            "- Counts are 30-minute forecasts, not separate storms.")

with tab_about:
    st.markdown(
        """
**How it works**
1. NASA's IMERG satellite data gives a rain map of the region every 30 minutes.
2. The model looks at the last 3 hours of maps and predicts the next 30 and 60 minutes.
3. It is a ConvLSTM, a neural network that learns how rain patterns move and change.
4. It was trained to care extra about heavy rain (5 mm/hr and above), because plain training
   made it miss storms.
5. The six cities get an alert level from the highest rain in the 3x3 pixels (about 30 km) around them.

**Alert levels used in this prototype**
Light from 1 mm/hr, Moderate from 5, Heavy from 10, Severe from 20. These are our own demo
thresholds, not the official weather office categories.

**Please remember**
- This is a student prototype, not an official forecast. For real decisions follow the India
  Meteorological Department.
- It was trained on one month (July 2024) of monsoon rain and tested on about 4.5 days, so we
  cannot claim it works in other seasons yet.
- This page replays saved predictions for the test period. It is not connected to live data.
""")
