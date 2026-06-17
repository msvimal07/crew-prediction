# =============================================================================
# GROUND CREW REQUIREMENT PREDICTION SYSTEM
# Airport Operations Intelligence Platform
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from autogluon.tabular import TabularPredictor
except Exception:
    TabularPredictor = None


st.set_page_config(
    page_title="Ground Crew Requirement Prediction",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TARGET = "required_ground_crew_count"
DATA_PATH = Path("airport_ground_handling_synthetic.csv")
MODEL_PATH = Path("models/ground_crew_autogluon")
ARTIFACT_PATH = Path("artifacts")
FEATURE_METADATA_PATH = ARTIFACT_PATH / "feature_metadata.json"
EVALUATION_REPORT_PATH = ARTIFACT_PATH / "evaluation_report.json"
PREDICTION_ARTIFACT_PATH = ARTIFACT_PATH / "validation_predictions.csv"

RAW_FEATURES = [
    "date",
    "shift",
    "scheduled_arrivals",
    "scheduled_departures",
    "international_flights",
    "domestic_flights",
    "wide_body_count",
    "narrow_body_count",
    "expected_passengers",
    "cargo_tonnage",
    "mail_tonnage",
    "dangerous_goods_tonnage",
    "weather_severity_index",
    "wind_speed",
    "visibility_km",
    "rainfall_mm",
    "equipment_utilization_rate",
    "equipment_breakdown_count",
    "active_gates",
    "occupied_gates",
    "staff_absenteeism_rate",
    "delay_minutes",
    "workload_index",
]

SHIFT_OPTIONS = ["Morning", "Afternoon", "Night"]
SHIFT_MULTIPLIERS = {"Morning": 1.08, "Afternoon": 1.00, "Night": 0.92}
SEASON_MAP = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
}


st.markdown(
    """
<style>
.block-container { padding-top: 1rem; padding-bottom: 0rem; }
.main { background-color: #f5f7fb; }
h1 { color: #0b1b5e; font-weight: 800; }
h2, h3 { color: #0b1b5e; }
.stButton > button {
    background-color: #1565ff; color: white; border-radius: 10px; height: 52px;
    width: 100%; font-size: 17px; font-weight: bold; border: none; margin-top: 4px;
    transition: background 0.2s;
}
.stButton > button:hover { background-color: #0d4ed8; }
[data-testid="stMetric"] {
    background: white; padding: 20px 22px; border-radius: 18px;
    border-top: 5px solid #1565ff; box-shadow: 0px 2px 10px rgba(0,0,0,0.07);
    text-align: left;
}
[data-testid="stMetricValue"] { font-size: 34px; font-weight: 700; color: #111827; }
[data-testid="stMetricLabel"] { font-size: 14px; color: #6B7280; }
.pred-card-green, .pred-card-yellow, .pred-card-red { padding: 28px 32px; border-radius: 14px; margin-bottom: 16px; }
.pred-card-green { background: linear-gradient(135deg, #dcfce7, #f0fdf4); border-left: 6px solid #16a34a; }
.pred-card-yellow { background: linear-gradient(135deg, #fef9c3, #fefce8); border-left: 6px solid #ca8a04; }
.pred-card-red { background: linear-gradient(135deg, #fee2e2, #fff1f2); border-left: 6px solid #dc2626; }
.pred-title { font-size: 13px; font-weight: 600; color: #6B7280; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 6px; }
.pred-value { font-size: 46px; font-weight: 800; color: #111827; line-height: 1.1; }
.pred-sub { font-size: 18px; color: #374151; margin-top: 6px; }
.pred-risk { font-size: 20px; font-weight: 700; margin-top: 10px; }
.section-header {
    font-size: 18px; font-weight: 700; color: #0b1b5e; border-bottom: 2px solid #1565ff;
    padding-bottom: 6px; margin-bottom: 14px;
}
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { height: 46px; padding: 0 24px; border-radius: 8px 8px 0 0; font-size: 15px; font-weight: 600; }
.stTabs [aria-selected="true"] { background-color: #1565ff; color: white; }
.stDataFrame { border-radius: 10px; overflow: hidden; }
.stSelectbox div[data-baseweb="select"] > div { min-height: 42px; }
.stTextInput input { height: 42px; }
</style>
""",
    unsafe_allow_html=True,
)


def safe_divide(numerator, denominator):
    return np.where(np.asarray(denominator) == 0, 0, np.asarray(numerator) / np.asarray(denominator))


def add_airport_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["year"] = data["date"].dt.year.fillna(0).astype(int)
    data["month"] = data["date"].dt.month.fillna(0).astype(int)
    data["quarter"] = data["date"].dt.quarter.fillna(0).astype(int)
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype("Int64").fillna(0).astype(int)
    data["day_of_week"] = data["date"].dt.dayofweek.fillna(0).astype(int)
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
    data["season"] = data["month"].map(SEASON_MAP).fillna("Unknown")
    data["shift"] = data["shift"].fillna("Morning")
    data["shift_workload_multiplier"] = data["shift"].map(SHIFT_MULTIPLIERS).fillna(1.0)

    data["total_flights"] = data["scheduled_arrivals"] + data["scheduled_departures"]
    data["international_ratio"] = safe_divide(data["international_flights"], data["total_flights"])
    data["domestic_ratio"] = safe_divide(data["domestic_flights"], data["total_flights"])
    aircraft_total = data["wide_body_count"] + data["narrow_body_count"]
    data["wide_body_ratio"] = safe_divide(data["wide_body_count"], aircraft_total)
    data["narrow_body_ratio"] = safe_divide(data["narrow_body_count"], aircraft_total)
    data["passengers_per_flight"] = safe_divide(data["expected_passengers"], data["total_flights"])
    data["passenger_density"] = safe_divide(data["expected_passengers"], data["active_gates"])
    data["cargo_per_flight"] = safe_divide(data["cargo_tonnage"], data["total_flights"])
    total_cargo = data["cargo_tonnage"] + data["mail_tonnage"] + data["dangerous_goods_tonnage"]
    data["mail_ratio"] = safe_divide(data["mail_tonnage"], total_cargo)
    data["dangerous_goods_ratio"] = safe_divide(data["dangerous_goods_tonnage"], total_cargo)

    data["low_visibility_flag"] = (data["visibility_km"] < 5).astype(int)
    data["heavy_rain_flag"] = (data["rainfall_mm"] >= 10).astype(int)
    data["high_wind_flag"] = (data["wind_speed"] >= 35).astype(int)
    data["weather_risk_score"] = (
        data["weather_severity_index"] * 12
        + data["low_visibility_flag"] * 15
        + data["heavy_rain_flag"] * 12
        + data["high_wind_flag"] * 10
    )
    data["gate_occupancy_ratio"] = safe_divide(data["occupied_gates"], data["active_gates"])
    data["equipment_stress_score"] = data["equipment_utilization_rate"] + data["equipment_breakdown_count"] * 8
    data["delay_severity"] = np.select(
        [data["delay_minutes"] < 30, data["delay_minutes"] < 90, data["delay_minutes"] < 180],
        ["Low", "Moderate", "High"],
        default="Critical",
    )
    data["staff_shortage_score"] = data["staff_absenteeism_rate"] * data["shift_workload_multiplier"]
    data["airport_workload_score"] = (
        data["workload_index"] * 0.35
        + data["total_flights"] * 0.20
        + data["passengers_per_flight"] * 0.12
        + data["cargo_per_flight"] * 3.0
        + data["wide_body_ratio"] * 20
    )
    data["resource_pressure_score"] = (
        data["gate_occupancy_ratio"] * 45
        + data["equipment_stress_score"] * 0.35
        + data["staff_shortage_score"] * 2.0
    )
    data["operational_complexity_score"] = (
        data["airport_workload_score"] * 0.45
        + data["weather_risk_score"] * 0.25
        + data["resource_pressure_score"] * 0.20
        + data["dangerous_goods_ratio"] * 25
        + data["international_ratio"] * 12
    )
    return data.replace([np.inf, -np.inf], 0)


def prediction_risk(crew_count: float) -> Tuple[str, str]:
    if crew_count >= 140:
        return "pred-card-red", "High staffing pressure"
    if crew_count >= 95:
        return "pred-card-yellow", "Moderate staffing pressure"
    return "pred-card-green", "Normal staffing pressure"


def recommendation_text(row: pd.Series, prediction: float) -> List[str]:
    notes = []
    if prediction >= 140:
        notes.append("Open overtime or reserve crew options before shift start.")
    elif prediction >= 95:
        notes.append("Keep standby crew visible to the duty manager.")
    else:
        notes.append("Planned staffing is likely sufficient for the forecast workload.")
    if row.get("weather_risk_score", 0) > 50:
        notes.append("Assign additional ramp supervision for weather-related turnaround risk.")
    if row.get("equipment_breakdown_count", 0) >= 3 or row.get("equipment_utilization_rate", 0) > 90:
        notes.append("Check GSE availability and pre-position replacement equipment.")
    if row.get("gate_occupancy_ratio", 0) > 0.85:
        notes.append("Coordinate gate sequencing to avoid crew idle time and towing conflicts.")
    if row.get("staff_absenteeism_rate", 0) > 6:
        notes.append("Escalate absenteeism risk to rostering and confirm backup coverage.")
    return notes


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    missing = sorted(set(RAW_FEATURES + [TARGET]) - set(df.columns))
    if missing:
        st.error(f"Dataset is missing required columns: {missing}")
        st.stop()
    return df


@st.cache_resource
def load_predictor(model_path: Path):
    if TabularPredictor is None or not model_path.exists():
        return None
    return TabularPredictor.load(str(model_path))


@st.cache_data
def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_model_path(metadata: Dict) -> Path:
    comparisons = metadata.get("split_subset_comparison", [])
    if isinstance(comparisons, list) and comparisons:
        best_entry = max(comparisons, key=lambda item: item.get("r2", float("-inf")))
        if best_entry.get("model_path"):
            return Path(best_entry["model_path"])
    return MODEL_PATH


def resolve_best_model_info(metadata: Dict) -> Tuple[str, str]:
    comparisons = metadata.get("split_subset_comparison", [])
    if isinstance(comparisons, list) and comparisons:
        best_entry = max(comparisons, key=lambda item: item.get("r2", float("-inf")))
        algorithm = best_entry.get("best_model") or "XGBooster"
        candidate = best_entry.get("candidate", "Unavailable")
        return algorithm, candidate
    return "XGBooster", "Unavailable"


def predict_with_model(predictor, raw_input: pd.DataFrame) -> np.ndarray:
    features = add_airport_features(raw_input)
    features = features.drop(columns=[TARGET, "date"], errors="ignore")
    model_features = feature_metadata.get("model_features", []) if isinstance(feature_metadata, dict) else []
    if model_features:
        missing = sorted(set(model_features) - set(features.columns))
        if missing:
            raise ValueError(f"Missing model features after preprocessing: {missing}")
        features = features[model_features]
    return predictor.predict(features).to_numpy()


st.markdown(
    """
<h1 style='font-size:38px; color:#1d2340; font-weight:700; margin-bottom:2px;'>
Ground Crew Requirement Prediction
</h1>
<p style='color:#6B7280; font-size:16px; margin-bottom:0;'>
Airport Operations Intelligence Platform
</p>
""",
    unsafe_allow_html=True,
)
st.divider()

raw_dataset = load_data()
dataset = add_airport_features(raw_dataset)
evaluation_report = load_json(EVALUATION_REPORT_PATH)
feature_metadata = load_json(FEATURE_METADATA_PATH)
resolved_model_path = resolve_model_path(feature_metadata)
predictor = load_predictor(resolved_model_path)
selected_algorithm, selected_candidate = resolve_best_model_info(feature_metadata)

if predictor is None:
    st.warning(
        f"Model artifacts are not available yet at {resolved_model_path}. Run the notebook through the model saving section to enable predictions."
    )

tab1, tab2, tab3, tab4 = st.tabs(
    ["Crew Requirement Prediction", "EDA", "Analytics Dashboard", "Bulk Crew Analysis"]
)

with tab1:
    st.markdown("<div class='section-header'>Shift Scenario Inputs</div>", unsafe_allow_html=True)
    st.subheader("Operations Planning Inputs")
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.markdown("**Shift & Schedule**")
        input_date = st.date_input("Date", value=pd.Timestamp.today().date())
        shift = st.selectbox("Shift", SHIFT_OPTIONS)
        scheduled_arrivals = st.number_input("Scheduled Arrivals", 0, 250, 70)
        scheduled_departures = st.number_input("Scheduled Departures", 0, 250, 72)

    with col_b:
        st.markdown("**Flight Mix & Passengers**")
        international_flights = st.number_input("International Flights", 0, 250, 45)
        domestic_flights = st.number_input("Domestic Flights", 0, 250, 95)
        wide_body_count = st.number_input("Wide Body Aircraft", 0, 150, 18)
        narrow_body_count = st.number_input("Narrow Body Aircraft", 0, 250, 122)
        expected_passengers = st.number_input("Expected Passengers", 0, 80000, 24000, step=100)

    with col_c:
        st.markdown("**Cargo & Weather**")
        cargo_tonnage = st.number_input("Cargo Tonnage", 0.0, 500.0, 150.0, step=1.0)
        mail_tonnage = st.number_input("Mail Tonnage", 0.0, 80.0, 12.0, step=0.5)
        dangerous_goods_tonnage = st.number_input("Dangerous Goods Tonnage", 0.0, 25.0, 1.5, step=0.1)
        weather_severity_index = st.slider("Weather Severity Index", 0.0, 5.0, 2.0, 0.1)
        wind_speed = st.slider("Wind Speed", 0.0, 80.0, 24.0, 0.5)
        visibility_km = st.slider("Visibility (km)", 0.1, 30.0, 12.0, 0.1)
        rainfall_mm = st.slider("Rainfall (mm)", 0.0, 80.0, 2.0, 0.5)

    with col_d:
        st.markdown("**Resources & Disruptions**")
        equipment_utilization_rate = st.slider("Equipment Utilization Rate (%)", 0.0, 100.0, 76.0, 0.5)
        equipment_breakdown_count = st.number_input("Equipment Breakdowns", 0, 20, 1)
        active_gates = st.number_input("Active Gates", 1, 200, 60)
        occupied_gates = st.number_input("Occupied Gates", 0, 200, 44)
        staff_absenteeism_rate = st.slider("Staff Absenteeism Rate (%)", 0.0, 25.0, 3.0, 0.1)
        delay_minutes = st.number_input("Delay Minutes", 0.0, 600.0, 35.0, step=1.0)
        workload_index = st.slider("Workload Index", 0.0, 100.0, 58.0, 0.1)

    input_row = pd.DataFrame(
        [
            {
                "date": str(input_date),
                "shift": shift,
                "scheduled_arrivals": scheduled_arrivals,
                "scheduled_departures": scheduled_departures,
                "international_flights": international_flights,
                "domestic_flights": domestic_flights,
                "wide_body_count": wide_body_count,
                "narrow_body_count": narrow_body_count,
                "expected_passengers": expected_passengers,
                "cargo_tonnage": cargo_tonnage,
                "mail_tonnage": mail_tonnage,
                "dangerous_goods_tonnage": dangerous_goods_tonnage,
                "weather_severity_index": weather_severity_index,
                "wind_speed": wind_speed,
                "visibility_km": visibility_km,
                "rainfall_mm": rainfall_mm,
                "equipment_utilization_rate": equipment_utilization_rate,
                "equipment_breakdown_count": equipment_breakdown_count,
                "active_gates": active_gates,
                "occupied_gates": occupied_gates,
                "staff_absenteeism_rate": staff_absenteeism_rate,
                "delay_minutes": delay_minutes,
                "workload_index": workload_index,
            }
        ]
    )
    engineered_input = add_airport_features(input_row)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Flights", int(engineered_input["total_flights"].iloc[0]))
    c2.metric("Gate Occupancy", f"{engineered_input['gate_occupancy_ratio'].iloc[0] * 100:.1f}%")
    c3.metric("Weather Risk", f"{engineered_input['weather_risk_score'].iloc[0]:.1f}")
    c4.metric("Complexity Score", f"{engineered_input['operational_complexity_score'].iloc[0]:.1f}")

    if st.button("Predict Ground Crew Requirement", disabled=predictor is None):
        pred = float(predict_with_model(predictor, input_row)[0])
        pred_rounded = int(round(max(pred, 0)))
        card_class, risk_label = prediction_risk(pred_rounded)
        st.markdown(
            f"""
<div class='{card_class}'>
  <div class='pred-title'>Predicted Required Ground Crew</div>
  <div class='pred-value'>{pred_rounded} crew</div>
  <div class='pred-sub'>Model estimate for the selected shift scenario</div>
  <div class='pred-risk'>{risk_label}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='section-header'>Crew Requirement Recommendations</div>", unsafe_allow_html=True)
        for note in recommendation_text(engineered_input.iloc[0], pred_rounded):
            st.info(note)
        explanation = pd.DataFrame(
            {
                "Driver": [
                    "Total Flights",
                    "Passengers per Flight",
                    "Cargo per Flight",
                    "Weather Risk",
                    "Equipment Stress",
                    "Gate Occupancy",
                    "Staff Shortage",
                ],
                "Value": [
                    engineered_input["total_flights"].iloc[0],
                    engineered_input["passengers_per_flight"].iloc[0],
                    engineered_input["cargo_per_flight"].iloc[0],
                    engineered_input["weather_risk_score"].iloc[0],
                    engineered_input["equipment_stress_score"].iloc[0],
                    engineered_input["gate_occupancy_ratio"].iloc[0] * 100,
                    engineered_input["staff_shortage_score"].iloc[0],
                ],
            }
        )
        fig = px.bar(explanation, x="Driver", y="Value", color="Value", color_continuous_scale="Blues")
        fig.update_layout(
            height=420,
            showlegend=False,
            title="Prediction Explanation",
            margin={"l": 20, "r": 20, "t": 60, "b": 100},
            xaxis={"tickangle": -20, "automargin": True},
            yaxis={"automargin": True},
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("<div class='section-header'>Aviation Operations EDA</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records", f"{len(dataset):,}")
    c2.metric("Avg Crew", f"{dataset[TARGET].mean():.1f}")
    c3.metric("Avg Flights", f"{dataset['total_flights'].mean():.1f}")
    c4.metric("Avg Passengers", f"{dataset['expected_passengers'].mean():,.0f}")
    c5.metric("Avg Delay", f"{dataset['delay_minutes'].mean():.1f} min")

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        daily = dataset.groupby("date", as_index=False)[["total_flights", TARGET]].mean().sort_values("date")
        st.plotly_chart(px.line(daily, x="date", y="total_flights", title="Flight Volume Trends"), use_container_width=True)
    with row1_col2:
        st.plotly_chart(px.histogram(dataset, x=TARGET, nbins=35, marginal="box", title="Crew Demand Distribution"), use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        monthly = dataset.groupby("month", as_index=False)[["expected_passengers", "cargo_tonnage"]].mean()
        st.plotly_chart(px.line(monthly, x="month", y="expected_passengers", markers=True, title="Passenger Trends by Month"), use_container_width=True)
    with row2_col2:
        st.plotly_chart(px.scatter(dataset, x="weather_risk_score", y=TARGET, color="shift", size="total_flights", title="Weather Impact on Crew Demand"), use_container_width=True)

    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        st.plotly_chart(px.box(dataset, x="shift", y=TARGET, color="shift", title="Shift-wise Crew Demand"), use_container_width=True)
    with row3_col2:
        st.plotly_chart(px.scatter(dataset, x="delay_minutes", y=TARGET, color="delay_severity", title="Delay Analysis"), use_container_width=True)

    corr_cols = [
        TARGET,
        "total_flights",
        "expected_passengers",
        "cargo_tonnage",
        "weather_risk_score",
        "equipment_stress_score",
        "gate_occupancy_ratio",
        "delay_minutes",
        "airport_workload_score",
        "resource_pressure_score",
        "operational_complexity_score",
    ]
    st.plotly_chart(
        px.imshow(dataset[corr_cols].corr(numeric_only=True), text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Correlation Heatmap"),
        use_container_width=True,
    )

with tab3:
    st.markdown("<div class='section-header'>Model Metrics Dashboard</div>", unsafe_allow_html=True)

    if evaluation_report:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{evaluation_report.get('mae', 0):.2f}")
        c2.metric("RMSE", f"{evaluation_report.get('rmse', 0):.2f}")
        c3.metric("R2", f"{evaluation_report.get('r2', 0):.3f}")
        c4.metric("MAPE", f"{evaluation_report.get('mape', 0):.2f}%")
    else:
        st.info("Evaluation report will reappear after the notebook saves model artifacts.")

    p1, p2 = st.columns(2)
    with p1:
        if PREDICTION_ARTIFACT_PATH.exists():
            preds = pd.read_csv(PREDICTION_ARTIFACT_PATH)
            fig = px.scatter(preds, x="actual", y="predicted", trendline="ols", title="Actual vs Predicted")
            fig.add_trace(
                go.Scatter(
                    x=[preds["actual"].min(), preds["actual"].max()],
                    y=[preds["actual"].min(), preds["actual"].max()],
                    mode="lines",
                    name="Ideal",
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(px.scatter(dataset.sample(min(500, len(dataset)), random_state=42), x="workload_index", y=TARGET, color="shift", title="Workload Analysis"), use_container_width=True)
    with p2:
        importance = feature_metadata.get("feature_importance", [])
        if importance:
            imp = pd.DataFrame(importance).head(20)
            fig = px.bar(imp, x="importance", y="feature", orientation="h", title="Feature Importance")
            fig.update_layout(
                yaxis={"categoryorder": "total ascending", "automargin": True},
                xaxis={"automargin": True},
                margin={"l": 20, "r": 20, "t": 60, "b": 60},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            seasonal = dataset.groupby("season", as_index=False)[TARGET].mean()
            st.plotly_chart(px.bar(seasonal, x="season", y=TARGET, color="season", title="Seasonality Analysis"), use_container_width=True)

with tab4:
    st.markdown("<div class='section-header'>Bulk Ground Crew Requirement Analysis</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV with airport operations features", type=["csv"])
    sample = raw_dataset[RAW_FEATURES].head(20).copy()
    st.download_button(
        "Download Input Template",
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name="ground_crew_prediction_template.csv",
        mime="text/csv",
    )

    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        missing = sorted(set(RAW_FEATURES) - set(batch.columns))
        if missing:
            st.error(f"Uploaded file is missing required columns: {missing}")
        elif predictor is None:
            st.warning("Run the notebook training workflow before batch prediction.")
        else:
            predictions = predict_with_model(predictor, batch[RAW_FEATURES])
            results = add_airport_features(batch[RAW_FEATURES])
            results["predicted_required_ground_crew_count"] = np.maximum(np.round(predictions), 0).astype(int)
            results["risk_indicator"] = pd.cut(
                results["predicted_required_ground_crew_count"],
                bins=[-1, 94, 139, np.inf],
                labels=["Normal", "Moderate", "High"],
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Shifts", f"{len(results):,}")
            c2.metric("Avg Predicted Crew", f"{results['predicted_required_ground_crew_count'].mean():.1f}")
            c3.metric("Max Predicted Crew", f"{results['predicted_required_ground_crew_count'].max():.0f}")
            c4.metric("High Risk Shifts", f"{(results['risk_indicator'] == 'High').sum():,}")
            st.plotly_chart(px.histogram(results, x="predicted_required_ground_crew_count", color="risk_indicator", title="Batch Prediction Distribution"), use_container_width=True)
            st.dataframe(results, use_container_width=True)
            st.download_button(
                "Download Prediction Results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="ground_crew_prediction_results.csv",
                mime="text/csv",
            )
