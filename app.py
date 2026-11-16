"""
Will It Rain Tomorrow? — Rainfall Prediction Dashboard
5th Sem BDA Mini Project

Run with:  streamlit run app.py
"""

from datetime import date
import io
from pathlib import Path
import random

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Will It Rain Tomorrow?", page_icon="🌧️", layout="wide")

DATA_PATH = Path(__file__).with_name("weatherAUS.csv")
RAIN_COLOR = "#2563eb"   # blue  = rain
DRY_COLOR = "#f59e0b"    # amber = no rain
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# The measurements a user can adjust on the prediction tab.
# column: (friendly label, unit, min, max, step)
KEY_FEATURES = {
    "Humidity3pm":   ("Humidity at 3 pm",          "%",            0.0,   100.0, 1.0),
    "Humidity9am":   ("Humidity at 9 am",          "%",            0.0,   100.0, 1.0),
    "Pressure3pm":   ("Air pressure at 3 pm",      "hPa",          975.0, 1040.0, 0.5),
    "Sunshine":      ("Hours of sunshine today",   "hours",        0.0,   14.5,  0.5),
    "Cloud3pm":      ("Cloud cover at 3 pm",       "oktas, 0–8",   0.0,   8.0,   1.0),
    "WindGustSpeed": ("Strongest wind gust",       "km/h",         0.0,   135.0, 1.0),
    "Rainfall":      ("Rain that fell today",      "mm",           0.0,   100.0, 0.5),
    "MinTemp":       ("Lowest temperature today",  "°C",           -9.0,  34.0,  0.5),
    "MaxTemp":       ("Highest temperature today", "°C",           -5.0,  48.0,  0.5),
    "Temp3pm":       ("Temperature at 3 pm",       "°C",           -6.0,  47.0,  0.5),
}

OTHER_NAMES = {
    "Location": "Location", "Month": "Month", "RainToday": "Rained today (Yes/No)",
    "WindGustDir": "Wind gust direction", "WindDir9am": "Wind direction at 9 am", "WindDir3pm": "Wind direction at 3 pm",
    "WindSpeed9am": "Wind speed at 9 am", "WindSpeed3pm": "Wind speed at 3 pm", "Pressure9am": "Air pressure at 9 am",
    "Cloud9am": "Cloud cover at 9 am", "Temp9am": "Temperature at 9 am", "Evaporation": "Evaporation",
}

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; }
      .hero { padding: 1.4rem 1.6rem; border-radius: 14px; margin-bottom: 1.2rem;
              background: linear-gradient(120deg, #1e3a8a, #2563eb); color: #fff; }
      .hero h1 { margin: 0; font-size: 2.1rem; color: #fff; }
      .hero p  { margin: .4rem 0 0; font-size: 1.05rem; opacity: .9; }
      div[data-testid="stMetric"] { border: 1px solid rgba(128,128,128,.25); border-radius: 12px;
                                    padding: .9rem 1rem; }
      .result { border-radius: 16px; padding: 1.5rem 1.6rem; color: #fff; margin-bottom: 1rem; }
      .result.rain { background: linear-gradient(135deg, #1d4ed8, #3b82f6); }
      .result.dry  { background: linear-gradient(135deg, #d97706, #fbbf24); }
      .result .big { font-size: 2.2rem; font-weight: 800; line-height: 1.15; }
      .result .sub { font-size: 1.05rem; opacity: .95; margin-top: .35rem; }
      .bar { background: rgba(255,255,255,.35); border-radius: 999px; height: 14px; margin-top: .9rem; }
      .bar > div { background: #fff; border-radius: 999px; height: 14px; }
      .scale { display: flex; justify-content: space-between; font-size: .8rem; opacity: .9; margin-top: .25rem; }
      .note { border-left: 4px solid #2563eb; background: rgba(37,99,235,.07); padding: .7rem 1rem;
              border-radius: 6px; margin: .4rem 0 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Data + model (same cleaning and training steps as RainPredictor.ipynb)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data = data.drop_duplicates().dropna(subset=["RainTomorrow"]).copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Month"] = data["Date"].dt.month
    data = data.drop(columns=["RISK_MM"], errors="ignore")
    data = data[data["RainTomorrow"].isin(["Yes", "No"])].reset_index(drop=True)
    return data


@st.cache_data(show_spinner=False)
def load_model_data() -> pd.DataFrame:
    data = load_raw().drop(columns=["Date"])
    data["RainTomorrow"] = data["RainTomorrow"].map({"No": 0, "Yes": 1}).astype(int)
    return data


@st.cache_resource(show_spinner=False)
def train_models():
    data = load_model_data()
    features = data.drop(columns=["RainTomorrow"])
    target = data["RainTomorrow"]

    split_index = int(len(data) * 0.8)          # 80% train / 20% test
    x_train, x_test = features.iloc[:split_index], features.iloc[split_index:]
    y_train, y_test = target.iloc[:split_index], target.iloc[split_index:]

    numeric_cols = x_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = x_train.select_dtypes(exclude="number").columns.tolist()

    def make_preprocessor():
        return ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols),
        ])

    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=6000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1
        ),
    }

    models, rows, matrices = {}, [], {}
    for name, estimator in estimators.items():
        pipe = Pipeline([("preprocessing", make_preprocessor()), ("model", estimator)])
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        models[name] = pipe
        matrices[name] = confusion_matrix(y_test, pred, labels=[0, 1])
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-Score": f1_score(y_test, pred, zero_division=0),
        })
    metrics = pd.DataFrame(rows).set_index("Model")

    # Which inputs the Random Forest relies on most (one-hot columns summed back
    # into their original feature, e.g. all "Location_*" columns -> "Location").
    rf = models["Random Forest"]
    names = rf.named_steps["preprocessing"].get_feature_names_out()
    importances = rf.named_steps["model"].feature_importances_
    original = []
    for n in names:
        kind, col = n.split("__", 1)
        if kind == "cat":
            col = next(c for c in categorical_cols if col.startswith(c + "_"))
        original.append(col)
    importance = (
        pd.DataFrame({"Feature": original, "Importance": importances})
        .groupby("Feature", as_index=False)["Importance"].sum()
        .sort_values("Importance", ascending=False)
    )
    return models, metrics, matrices, importance, len(x_train), len(x_test)


def pct(x: float) -> str:
    return f"{x:.0%}" if x >= 0.995 or x < 0.1 else f"{x:.1%}"


def chance_words(p: float) -> str:
    if p >= 0.80:
        return "Very likely to rain"
    if p >= 0.60:
        return "Likely to rain"
    if p >= 0.50:
        return "Slightly more likely to rain than not"
    if p >= 0.35:
        return "Possible, but more likely dry"
    if p >= 0.15:
        return "Unlikely to rain"
    return "Very unlikely to rain"


# ----------------------------------------------------------------------------
# Custom inputs <-> weatherAUS.csv
# A custom entry is one row in the CSV with RainTomorrow = NA (tomorrow isn't known yet).
# Same location + same day  -> that same row is reused (moved to the end, never duplicated).
# New day                   -> a new row is appended at the end.
# The notebook's Step 10 always predicts the LAST row of the CSV.
# ----------------------------------------------------------------------------
def read_csv_lines() -> list[str]:
    with open(DATA_PATH, encoding="utf-8", newline="") as f:
        return f.read().splitlines()


def find_saved_row(day_str: str, location: str):
    """Return (line_number, row_dict) of the custom row for this day + location, or (None, None)."""
    lines = read_csv_lines()
    header = lines[0].split(",")
    i_date, i_loc, i_rt = header.index("Date"), header.index("Location"), header.index("RainTomorrow")
    for n in range(len(lines) - 1, 0, -1):          # newest rows are at the end
        parts = lines[n].split(",")
        if len(parts) == len(header) and parts[i_date] == day_str and parts[i_loc] == location \
                and parts[i_rt] in ("NA", ""):
            return n, dict(zip(header, parts))
    return None, None


def save_custom_row(day_str: str, location: str, values: dict) -> int:
    """Write the custom inputs as a row at the END of weatherAUS.csv. Returns its line number."""
    lines = read_csv_lines()
    header = lines[0].split(",")
    old_line, _ = find_saved_row(day_str, location)
    if old_line is not None:                         # same day again -> replace, don't duplicate
        del lines[old_line]
    row = {col: "NA" for col in header}
    row.update({"Date": day_str, "Location": location, "RainTomorrow": "NA",
                "RainToday": "Yes" if values["Rainfall"] >= 1 else "No"})
    for feat, v in values.items():
        row[feat] = f"{v:g}"
    lines.append(",".join(row[c] for c in header))
    with open(DATA_PATH, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines) - 1


def row_to_model_input(row: dict) -> pd.DataFrame:
    """Turn one CSV row into the exact table the model expects (same steps as the notebook)."""
    header = list(row.keys())
    frame = pd.read_csv(io.StringIO(",".join(header) + "\n" + ",".join(row.values())))
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Month"] = frame["Date"].dt.month
    frame = frame.drop(columns=["Date", "RISK_MM", "RainTomorrow"], errors="ignore")
    feature_cols = [c for c in model_df.columns if c != "RainTomorrow"]
    frame = frame[feature_cols]
    for col in feature_cols:
        if model_df[col].dtype.kind not in "biuf":   # text columns stay text even when empty
            frame[col] = frame[col].astype(object)
    return frame


def show_forecast(prob: float, caption: str):
    will_rain = prob >= 0.5
    css, icon, headline = ("rain", "🌧️", "Rain expected tomorrow") if will_rain else ("dry", "☀️", "No rain expected tomorrow")
    st.markdown(
        f"""
        <div class="result {css}">
          <div class="big">{icon} {headline}</div>
          <div class="sub">Chance of rain: <b>{prob:.0%}</b> — {chance_words(prob)}</div>
          <div class="bar"><div style="width:{prob*100:.1f}%"></div></div>
          <div class="scale"><span>0% · dry</span><span>50%</span><span>100% · rain</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(caption + " The model says “rain” when the chance is 50% or more.")
    return will_rain


def show_reasons(values: dict):
    st.subheader("Why this forecast?")
    st.caption("Each of today's values compared with what's typical before a rainy day vs. before a dry day.")
    reasons = []
    for feat in ["Humidity3pm", "Sunshine", "Cloud3pm", "Pressure3pm", "Rainfall", "WindGustSpeed"]:
        label, unit = KEY_FEATURES[feat][0], KEY_FEATURES[feat][1].split(",")[0]
        v, r_med, d_med = values[feat], rainy[feat].median(), dry[feat].median()
        if pd.isna(v):
            signal, shown = "➖ Not given", "—"
        else:
            signal = "🌧️ Points to rain" if abs(v - r_med) < abs(v - d_med) else "☀️ Points to dry"
            shown = f"{v:g} {unit}"
        reasons.append({"Today's weather": label, "Your value": shown,
                        "Usual before rain": f"{r_med:g} {unit}", "Usual before dry": f"{d_med:g} {unit}",
                        "Signal": signal})
    st.dataframe(pd.DataFrame(reasons), hide_index=True, width="stretch")
    n_rain = sum("rain" in r["Signal"] for r in reasons)
    st.caption(f"{n_rain} of {len(reasons)} signals point towards rain. "
               "The model also uses wind direction, location and month.")


def weather_sliders(start: dict, key_prefix: str) -> dict:
    """Two columns of sliders, pre-filled from `start`. Returns (chosen values, starting values)."""
    values, starts = {}, {}
    cols = st.columns(2)
    for i, (feat, (label, unit, lo, hi, step)) in enumerate(KEY_FEATURES.items()):
        val = float(min(max(start[feat], lo), hi))
        val = round(round(val / step) * step, 2)
        starts[feat] = val
        values[feat] = cols[i % 2].slider(f"{label} ({unit})", lo, hi, val, step, key=f"{feat}|{key_prefix}",
                                          format="%.0f" if step >= 1 else "%.1f")
    return values, starts


def typical_values(location: str, month: int | None = None, row: dict | None = None) -> dict:
    """Start values for the sliders: the given row where it has a value, else the place's usual value."""
    pool = model_df[model_df["Location"] == location]
    if month is not None and (pool["Month"] == month).any():
        pool = pool[pool["Month"] == month]
    med = pool.median(numeric_only=True)
    out = {}
    for feat in KEY_FEATURES:
        v = np.nan
        if row is not None:
            v = pd.to_numeric(row.get(feat), errors="coerce")
        if pd.isna(v):
            v = med.get(feat, np.nan)
        if pd.isna(v):
            v = model_df[feat].median()
        out[feat] = float(v)
    return out


# ----------------------------------------------------------------------------
# Load everything
# ----------------------------------------------------------------------------
if not DATA_PATH.exists():
    st.error("weatherAUS.csv was not found. Put it in the same folder as app.py and refresh.")
    st.stop()

with st.spinner("Loading weather data and training the models (first run takes about a minute)…"):
    raw = load_raw()
    model_df = load_model_data()
    models, metrics, matrices, importance, n_train, n_test = train_models()

best_model_name = metrics["F1-Score"].idxmax()
rain_rate = (raw["RainTomorrow"] == "Yes").mean()
rainy = model_df[model_df["RainTomorrow"] == 1]
dry = model_df[model_df["RainTomorrow"] == 0]

# ----------------------------------------------------------------------------
# Header + headline numbers
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
      <h1>🌧️ Will It Rain Tomorrow?</h1>
      <p>Enter today's weather and a machine-learning model predicts whether it will rain tomorrow.
      Built on {len(raw):,} days of real weather records from {raw['Location'].nunique()} places in Australia
      ({raw['Date'].dt.year.min()}–{raw['Date'].dt.year.max()}).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Days of weather data", f"{len(raw):,}")
k2.metric("Weather stations", raw["Location"].nunique())
k3.metric("Days followed by rain", pct(rain_rate), help="Share of days where it rained the next day.")
k4.metric(f"Accuracy ({best_model_name})", pct(metrics.loc[best_model_name, 'Accuracy']),
          help="How often the chosen model was right on days it had never seen.")

tab_predict, tab_explore, tab_model, tab_about = st.tabs(
    ["🔮 Make a prediction", "📊 Explore the data", "🤖 How good is the model?", "📘 About the project"]
)

# ============================================================================
# TAB 1 — PREDICT
# ============================================================================
with tab_predict:
    locations = sorted(raw["Location"].unique())
    mode = st.radio(
        "What do you want to do?",
        ["✏️ Enter my own weather (saved to the CSV)", "📅 Look up a past day from the dataset"],
        horizontal=True, key="mode",
    )
    custom_mode = mode.startswith("✏️")

    if custom_mode:
        st.markdown(
            '<div class="note"><b>How it works:</b> ① pick the place and the day, ② set today\'s weather, '
            "③ click <b>Save to CSV &amp; predict</b>. Your inputs are added as a new row at the end of "
            "<code>weatherAUS.csv</code> and the model predicts from that row. "
            "Checking the <b>same day again</b> reuses that same row; a <b>new day</b> gets a new row.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="note"><b>How it works:</b> pick a place and any day from 2007–2017. '
            "The real weather from that day is loaded, the model predicts the next day, and you can see if it was right. "
            "Moving a slider makes it a “what-if” (nothing is saved).</div>",
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.subheader("① Place and day")
        if "loc" not in st.session_state:
            st.session_state["loc"] = "Sydney" if "Sydney" in locations else locations[0]
        c1, c2 = st.columns(2)
        location = c1.selectbox("Location", locations, key="loc")

        if custom_mode:
            day = c2.date_input("Day (type it or pick from the calendar)", value=date.today(),
                                min_value=date(2000, 1, 1), max_value=date(2100, 12, 31),
                                format="DD/MM/YYYY", key="custom_day")
            day_str = day.strftime("%Y-%m-%d")
            saved_line, saved_row = find_saved_row(day_str, location)
            if saved_row:
                st.success(f"📂 You already saved weather for **{location}** on **{day:%d %b %Y}** — "
                           f"loaded that same row (row index {saved_line - 1} in the CSV).")
            else:
                st.info(f"🆕 Nothing saved yet for **{location}** on **{day:%d %b %Y}**. "
                        "Sliders start at this place's usual values for this month.")

            st.subheader("② Today's weather")
            start = typical_values(location, day.month, saved_row)
            user_values, _ = weather_sliders(start, f"custom|{location}|{day_str}")

            if st.button("💾 Save to CSV & predict", type="primary", width="stretch"):
                saved_line = save_custom_row(day_str, location, user_values)
                saved_line, saved_row = find_saved_row(day_str, location)
                st.toast(f"Saved to weatherAUS.csv as row index {saved_line - 1}", icon="💾")
        else:
            loc_rows = raw[raw["Location"] == location].sort_values("Date")
            first, last = loc_rows["Date"].min().date(), loc_rows["Date"].max().date()
            if "past_day" not in st.session_state or not (first <= st.session_state["past_day"] <= last):
                st.session_state["past_day"] = last
            if c2.button("🎲 Random day", width="stretch"):
                st.session_state["past_day"] = random.choice(loc_rows["Date"].tolist()).date()
            day = c2.date_input("Day (type it or pick from the calendar)", min_value=first, max_value=last,
                                format="DD/MM/YYYY", key="past_day")
            nearest = (loc_rows["Date"] - pd.Timestamp(day)).abs().idxmin()
            if loc_rows.loc[nearest, "Date"].date() != day:
                st.warning(f"No record for {location} on {day:%d %b %Y} — using the closest day, "
                           f"{loc_rows.loc[nearest, 'Date']:%d %b %Y}.")
            base_row = model_df.loc[[nearest]].drop(columns=["RainTomorrow"])
            actual_tomorrow = raw.loc[nearest, "RainTomorrow"]
            shown_day = loc_rows.loc[nearest, "Date"]

            st.subheader("② Today's weather")
            st.caption("Filled in from that day. Drag any slider to try a what-if.")
            start = typical_values(location, int(base_row["Month"].iloc[0]), base_row.iloc[0].to_dict())
            user_values, start_vals = weather_sliders(start, f"past|{location}|{shown_day:%Y-%m-%d}")

        model_name = st.radio(
            "Model to use", list(models.keys()), index=list(models.keys()).index(best_model_name),
            horizontal=True,
            help="Random Forest is the default because it catches more rainy days (see the 'How good is the model?' tab).",
        )

    with right:
        st.subheader("③ Tomorrow's forecast")
        if custom_mode:
            if not saved_row:
                st.markdown("#### 👈 Click **Save to CSV & predict** to get the forecast.")
            else:
                x_input = row_to_model_input(saved_row)
                prob = float(models[model_name].predict_proba(x_input)[0][1])
                show_forecast(prob, f"{location}, day after {day:%d %b %Y} · model: {model_name} · "
                                    f"predicted from row index {saved_line - 1} of weatherAUS.csv.")
                saved_vals = {f: float(saved_row[f]) if saved_row[f] not in ("NA", "") else np.nan for f in KEY_FEATURES}
                if any(not np.isclose(user_values[f], saved_vals[f], equal_nan=False) for f in KEY_FEATURES
                       if not pd.isna(saved_vals[f])):
                    st.warning("✏️ You've changed values since saving. Click **Save to CSV & predict** again to "
                               "update this day's row.")
                with st.expander("See this row exactly as it is in weatherAUS.csv"):
                    st.dataframe(pd.DataFrame([saved_row]), hide_index=True, width="stretch")
                show_reasons(saved_vals)
        else:
            x_input = base_row.copy()
            for feat, v in user_values.items():
                x_input[feat] = v
            x_input["RainToday"] = "Yes" if user_values["Rainfall"] >= 1 else "No"
            prob = float(models[model_name].predict_proba(x_input)[0][1])
            will_rain = show_forecast(prob, f"{location}, day after {shown_day:%d %b %Y} · model: {model_name}.")
            edited = any(abs(user_values[f] - start_vals[f]) > 1e-9 for f in KEY_FEATURES)
            if edited:
                st.info("✏️ You changed some values, so this is a **what-if** forecast — there's no real answer to compare with.")
            else:
                actual_text = "🌧️ It **did rain**" if actual_tomorrow == "Yes" else "☀️ It **did not rain**"
                verdict = "✅ The model got it right." if (actual_tomorrow == "Yes") == will_rain else "❌ The model got this one wrong."
                st.markdown(f"**What really happened the next day:** {actual_text}. {verdict}")
            show_reasons(user_values)

    with st.expander("📄 Last 5 rows of weatherAUS.csv (your saved entries appear here)"):
        tail = read_csv_lines()
        st.dataframe(pd.read_csv(io.StringIO("\n".join([tail[0]] + tail[-5:]))), hide_index=True, width="stretch")
        st.caption(f"The file has {len(tail) - 1:,} rows. The notebook's Step 10 predicts the very last one.")

# ============================================================================
# TAB 2 — EXPLORE
# ============================================================================
with tab_explore:
    st.markdown(
        '<div class="note">These charts show the patterns in the historical data that the model learns from. '
        "<b>Blue = rain the next day</b>, <b>amber = no rain</b>.</div>",
        unsafe_allow_html=True,
    )
    place = st.selectbox("Show data for", ["All locations"] + sorted(raw["Location"].unique()), key="explore_loc")
    view = raw if place == "All locations" else raw[raw["Location"] == place]
    view_rain = (view["RainTomorrow"] == "Yes")

    a, b = st.columns(2, gap="large")

    with a:
        st.subheader("Which months are the wettest?")
        by_month = view.assign(rain=view_rain).groupby("Month")["rain"].mean().reindex(range(1, 13)).reset_index()
        by_month["Month name"] = [MONTHS[m - 1] for m in by_month["Month"]]
        chart = alt.Chart(by_month).mark_bar(color=RAIN_COLOR, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Month name:N", sort=MONTHS, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("rain:Q", title="Chance of rain next day", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("Month name:N", title="Month"), alt.Tooltip("rain:Q", title="Rain chance", format=".1%")],
        ).properties(height=300)
        st.altair_chart(chart, width="stretch")
        wet, dry_m = by_month.loc[by_month["rain"].idxmax()], by_month.loc[by_month["rain"].idxmin()]
        st.caption(f"Wettest month: **{wet['Month name']}** ({wet['rain']:.0%}). "
                   f"Driest month: **{dry_m['Month name']}** ({dry_m['rain']:.0%}).")

    with b:
        st.subheader("Does rain today mean rain tomorrow?")
        rt = view.dropna(subset=["RainToday"]).assign(rain=view_rain)
        rt = rt.groupby("RainToday")["rain"].mean().reindex(["No", "Yes"]).reset_index()
        rt["Today"] = rt["RainToday"].map({"No": "Dry today", "Yes": "Rained today"})
        chart = alt.Chart(rt).mark_bar(size=70, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Today:N", title=None, sort=["Dry today", "Rained today"], axis=alt.Axis(labelAngle=0, labelFontSize=13)),
            y=alt.Y("rain:Q", title="Chance of rain next day", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Today:N", scale=alt.Scale(domain=["Dry today", "Rained today"], range=[DRY_COLOR, RAIN_COLOR]), legend=None),
            tooltip=[alt.Tooltip("Today:N"), alt.Tooltip("rain:Q", title="Rain next day", format=".1%")],
        ).properties(height=300)
        st.altair_chart(chart, width="stretch")
        if rt["rain"].notna().all():
            st.caption(f"After a rainy day, rain follows **{rt['rain'].iloc[1]:.0%}** of the time — "
                       f"after a dry day only **{rt['rain'].iloc[0]:.0%}**.")

    c, d = st.columns(2, gap="large")

    with c:
        st.subheader("Afternoon humidity is the strongest clue")
        h = view.dropna(subset=["Humidity3pm"]).copy()
        h["bin"] = (h["Humidity3pm"] // 5 * 5).clip(upper=95)
        h["Next day"] = np.where(h["RainTomorrow"] == "Yes", "Rain", "No rain")
        hist = h.groupby(["bin", "Next day"]).size().reset_index(name="Days")
        hist["Share"] = hist["Days"] / hist.groupby("Next day")["Days"].transform("sum")
        hist["Humidity"] = hist["bin"].astype(int).astype(str) + "%"
        chart = alt.Chart(hist).mark_bar().encode(
            x=alt.X("Humidity:N", sort=[f"{b}%" for b in range(0, 100, 5)], title="Humidity at 3 pm",
                    axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Next day:N", sort=["No rain", "Rain"]),
            y=alt.Y("Share:Q", title="Share of days", axis=alt.Axis(format="%")),
            color=alt.Color("Next day:N", scale=alt.Scale(domain=["No rain", "Rain"], range=[DRY_COLOR, RAIN_COLOR]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["Next day:N", alt.Tooltip("Humidity:N", title="Humidity from"), alt.Tooltip("Share:Q", format=".1%")],
        ).properties(height=300)
        st.altair_chart(chart, width="stretch")
        hr = view.loc[view_rain, "Humidity3pm"].median()
        hd = view.loc[~view_rain, "Humidity3pm"].median()
        st.caption(f"Before rainy days the typical 3 pm humidity is **{hr:.0f}%**, before dry days **{hd:.0f}%**.")

    with d:
        if place == "All locations":
            st.subheader("Wettest and driest places")
            by_loc = raw.assign(rain=raw["RainTomorrow"] == "Yes").groupby("Location")["rain"].mean()
            top = pd.concat([by_loc.nlargest(8), by_loc.nsmallest(8)]).reset_index()
            top["Group"] = np.where(top["rain"] >= by_loc.median(), "Wettest", "Driest")
            chart = alt.Chart(top).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                y=alt.Y("Location:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=200)),
                x=alt.X("rain:Q", title="Chance of rain next day", axis=alt.Axis(format="%")),
                color=alt.Color("Group:N", scale=alt.Scale(domain=["Driest", "Wettest"], range=[DRY_COLOR, RAIN_COLOR]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["Location:N", alt.Tooltip("rain:Q", title="Rain chance", format=".1%")],
            ).properties(height=380)
            st.altair_chart(chart, width="stretch")
            st.caption(f"8 wettest and 8 driest of {raw['Location'].nunique()} locations.")
        else:
            st.subheader(f"Rain over the years in {place}")
            yearly = view.assign(rain=view_rain, Year=view["Date"].dt.year).groupby("Year")["rain"].mean().reset_index()
            chart = alt.Chart(yearly).mark_line(point=True, color=RAIN_COLOR).encode(
                x=alt.X("Year:O", title=None),
                y=alt.Y("rain:Q", title="Chance of rain next day", axis=alt.Axis(format="%")),
                tooltip=["Year:O", alt.Tooltip("rain:Q", title="Rain chance", format=".1%")],
            ).properties(height=300)
            st.altair_chart(chart, width="stretch")
            st.caption(f"Overall in {place}: rain follows **{view_rain.mean():.0%}** of days "
                       f"(all of Australia: {rain_rate:.0%}).")

    with st.expander("See the raw data"):
        st.dataframe(view.head(500), width="stretch", hide_index=True)
        st.caption(f"Showing the first 500 of {len(view):,} rows.")

# ============================================================================
# TAB 3 — MODEL PERFORMANCE
# ============================================================================
with tab_model:
    st.markdown(
        f'<div class="note">Both models were trained on <b>{n_train:,}</b> days and then tested on '
        f"<b>{n_test:,}</b> days they had never seen. These scores come from that test.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Scores at a glance")
    m_cols = st.columns(len(metrics))
    for col, (name, row) in zip(m_cols, metrics.iterrows()):
        with col:
            st.markdown(f"#### {name}{' ⭐ (chosen)' if name == best_model_name else ''}")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Accuracy", pct(row["Accuracy"]))
            s2.metric("Precision", pct(row["Precision"]))
            s3.metric("Recall", pct(row["Recall"]))
            s4.metric("F1-Score", pct(row["F1-Score"]))

    with st.expander("What do these scores mean? (plain English)", expanded=True):
        st.markdown(
            "- **Accuracy** — out of all days, how many did the model get right?\n"
            "- **Precision** — when the model said *“rain”*, how often did it actually rain?\n"
            "- **Recall** — out of all the days it really rained, how many did the model catch?\n"
            "- **F1-Score** — one number that balances precision and recall.\n\n"
            f"Only about **{rain_rate:.0%}** of days are followed by rain, so a model that always says "
            f"“no rain” would already be **{1 - rain_rate:.0%}** accurate while being useless. "
            "That's why we pick the model using **F1-Score** for rainy days, not accuracy alone."
        )

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Side-by-side comparison")
        long = metrics.reset_index().melt(id_vars="Model", var_name="Score", value_name="Value")
        chart = alt.Chart(long).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Score:N", sort=["Accuracy", "Precision", "Recall", "F1-Score"], title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13)),
            xOffset="Model:N",
            y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None),
            color=alt.Color("Model:N", scale=alt.Scale(range=["#64748b", RAIN_COLOR]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["Model:N", "Score:N", alt.Tooltip("Value:Q", format=".1%")],
        ).properties(height=320)
        st.altair_chart(chart, width="stretch")

    with right:
        st.subheader("What the model pays attention to")
        top_imp = importance.head(10).copy()
        top_imp["Name"] = top_imp["Feature"].map(lambda f: KEY_FEATURES.get(f, (OTHER_NAMES.get(f, f),))[0])
        chart = alt.Chart(top_imp).mark_bar(color=RAIN_COLOR, cornerRadiusTopRight=3, cornerRadiusBottomRight=3).encode(
            y=alt.Y("Name:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False)),
            x=alt.X("Importance:Q", title="Importance (Random Forest)", axis=alt.Axis(format="%")),
            tooltip=["Name:N", alt.Tooltip("Importance:Q", format=".1%")],
        ).properties(height=320)
        st.altair_chart(chart, width="stretch")
        st.caption(f"Top clue: **{top_imp['Name'].iloc[0]}** — the model relies on it more than anything else.")

    st.subheader("Right vs. wrong on the test days")
    st.caption("Each box counts test days. Green boxes are correct predictions, red boxes are mistakes.")
    cm_cols = st.columns(len(matrices), gap="large")
    for col, (name, cm) in zip(cm_cols, matrices.items()):
        tn, fp, fn, tp = cm.ravel()
        cells = pd.DataFrame([
            {"Actual": "Really dry", "Predicted": "Said dry", "Days": tn, "Result": "Correct", "Label": f"{tn:,}\ncorrect"},
            {"Actual": "Really dry", "Predicted": "Said rain", "Days": fp, "Result": "Wrong", "Label": f"{fp:,}\nfalse alarms"},
            {"Actual": "Really rained", "Predicted": "Said dry", "Days": fn, "Result": "Wrong", "Label": f"{fn:,}\nmissed rain"},
            {"Actual": "Really rained", "Predicted": "Said rain", "Days": tp, "Result": "Correct", "Label": f"{tp:,}\ncorrect"},
        ])
        base = alt.Chart(cells).encode(
            x=alt.X("Predicted:N", sort=["Said dry", "Said rain"], title="Model's prediction", axis=alt.Axis(labelAngle=0, orient="top")),
            y=alt.Y("Actual:N", sort=["Really dry", "Really rained"], title="What really happened"),
        )
        rect = base.mark_rect(cornerRadius=6, stroke="white", strokeWidth=3).encode(
            color=alt.Color("Result:N", scale=alt.Scale(domain=["Correct", "Wrong"], range=["#16a34a", "#dc2626"]), legend=None),
            opacity=alt.value(0.85),
            tooltip=["Actual:N", "Predicted:N", alt.Tooltip("Days:Q", format=",")],
        )
        text = base.mark_text(color="white", fontSize=15, fontWeight="bold", lineBreak="\n").encode(text="Label:N")
        with col:
            st.markdown(f"**{name}**")
            st.altair_chart((rect + text).properties(height=240), width="stretch")
            st.caption(f"Caught {tp:,} of {tp + fn:,} rainy days ({tp / (tp + fn):.0%}); "
                       f"{fp:,} false alarms.")

    lr, rf = metrics.loc["Logistic Regression"], metrics.loc["Random Forest"]
    st.success(
        f"**Conclusion:** Logistic Regression is slightly more accurate overall ({pct(lr['Accuracy'])} vs {pct(rf['Accuracy'])}), "
        f"but Random Forest catches far more of the rainy days (recall {pct(rf['Recall'])} vs {pct(lr['Recall'])}). "
        f"For a rain forecast, missing rain is the worse mistake, so **{best_model_name}** is the model used for predictions."
    )

# ============================================================================
# TAB 4 — ABOUT
# ============================================================================
with tab_about:
    st.subheader("Project summary")
    st.markdown(
        "This project uses machine learning to predict whether it will rain the next day from today's weather "
        "(temperature, humidity, wind, pressure, cloud and sunshine). Two models — **Logistic Regression** and "
        "**Random Forest** — are trained on historical Australian weather data and compared."
    )
    st.subheader("How it works")
    steps = [
        ("1. Load data", f"`weatherAUS.csv` — daily weather records from {raw['Location'].nunique()} Australian locations."),
        ("2. Clean", "Remove duplicates and rows with no RainTomorrow label; turn the date into a Month column; Yes/No → 1/0."),
        ("3. Prepare", "Missing numbers are filled with the median, missing categories with the most common value; "
                       "text columns (location, wind direction) are one-hot encoded."),
        ("4. Split", f"First 80% of rows for training ({n_train:,}), last 20% for testing ({n_test:,})."),
        ("5. Train", "Logistic Regression (max_iter = 6000) and Random Forest (100 trees, balanced class weights)."),
        ("6. Evaluate", "Accuracy, Precision, Recall, F1-Score and confusion matrix on the test set."),
        ("7. Predict", f"The best model by F1-Score (**{best_model_name}**) powers the prediction tab."),
    ]
    for title, text in steps:
        st.markdown(f"**{title}** — {text}")

    st.subheader("Dataset")
    st.markdown(
        "Rain in Australia (`weatherAUS.csv`), daily observations from Australian weather stations, "
        "originally from the Australian Bureau of Meteorology and published on Kaggle."
    )
    st.subheader("Tech used")
    st.markdown("Python · pandas · scikit-learn · Streamlit · Altair")
    st.caption("Run locally with:  `streamlit run app.py`")
