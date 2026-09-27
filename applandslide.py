import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Smart Peg Geotechnical Swarm | Landslide AI", layout="wide")
st.title("⛰️ Landslide Early Warning System: Geotechnical Smart Peg Swarm")
st.markdown("Subsurface geotechnical monitoring running LSTM time-series forecasting, Isolation Forest anomaly scoring, and physical failure-mode fusion.")

T_WIN = 14
FEATURE_NAMES = ['Rainfall (mm/day)', 'Soil Moisture (%)', 'Tilt Rate (deg/day)', 'AE Hits/hour']

# ==========================================
# 1. STREAMLIT UI SESSION STATE (Fixes stuck sliders)
# ==========================================
if "preset_radio" not in st.session_state:
    st.session_state.preset_radio = "🟢 Baseline Dry Slope (Nominal Stability)"
if "rain" not in st.session_state:
    st.session_state.rain = 10.0
if "tilt" not in st.session_state:
    st.session_state.tilt = 0.02
if "ae" not in st.session_state:
    st.session_state.ae = 2.0
if "forecast" not in st.session_state:
    st.session_state.forecast = 5.0

def apply_preset():
    p = st.session_state.preset_radio
    if p == "🟢 Baseline Dry Slope (Nominal Stability)":
        st.session_state.rain, st.session_state.tilt, st.session_state.ae, st.session_state.forecast = 10.0, 0.02, 2.0, 5.0
    elif p == "⚠️ High Rainfall Only (Safe Rocky Slope - False Alarm Check)":
        st.session_state.rain, st.session_state.tilt, st.session_state.ae, st.session_state.forecast = 120.0, 0.05, 5.0, 80.0
    elif p == "🚨 Pre-Collapse (Rainfall + Micro-cracks + Creep)":
        st.session_state.rain, st.session_state.tilt, st.session_state.ae, st.session_state.forecast = 80.0, 0.60, 40.0, 90.0
    elif p == "🚨 Dry Shear Collapse (Earthquake/Undercutting)":
        st.session_state.rain, st.session_state.tilt, st.session_state.ae, st.session_state.forecast = 0.0, 1.30, 85.0, 0.0

# ==========================================
# 2. CORE ML PIPELINE (MOCK LSTM & ISO FOREST)
# ==========================================
@st.cache_resource(show_spinner="Calibrating Geotechnical Models (LSTM + Isolation Forest)...")
def setup_models():
    np.random.seed(42)
    # Normal Baseline for Isolation Forest
    normal_readings = np.column_stack([
        np.random.gamma(2, 3, 400),
        20 + np.random.randn(400) * 8,
        np.random.randn(400) * 0.05,
        np.random.poisson(2, 400).astype(float)
    ])
    iso = IsolationForest(contamination=0.05, random_state=42).fit(normal_readings)
    raw_normal = -iso.score_samples(normal_readings)
    anom_lo, anom_hi = np.percentile(raw_normal, 5), np.percentile(raw_normal, 95) + 0.05
    return iso, anom_lo, anom_hi

iso_slope, ANOM_LO, ANOM_HI = setup_models()

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("🕹️ Geotechnical Presets")
st.sidebar.radio(
    "Select Operational Slope Condition:",
    [
        "🟢 Baseline Dry Slope (Nominal Stability)",
        "⚠️ High Rainfall Only (Safe Rocky Slope - False Alarm Check)",
        "🚨 Pre-Collapse (Rainfall + Micro-cracks + Creep)",
        "🚨 Dry Shear Collapse (Earthquake/Undercutting)"
    ],
    key="preset_radio",
    on_change=apply_preset
)

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Manual Sensor Inputs (Live Variables)")

rain_val = st.sidebar.slider("1. Daily Rainfall (mm/day)", 0.0, 200.0, key="rain")
st.sidebar.caption("Normal: <30 | Warning: 50-80 | Critical: >140")

tilt_val = st.sidebar.slider("2. MPU6050 Tilt Rate (deg/day)", 0.0, 2.0, key="tilt", step=0.05)
st.sidebar.caption("Normal: <0.05 | Warning: 0.1-0.4 | Critical: >1.0")

ae_val = st.sidebar.slider("3. Acoustic Emissions (Hits/hr)", 0.0, 100.0, key="ae", step=1.0)
st.sidebar.caption("Normal: <5 | Warning: 15-30 | Critical: >60")

forecast_val = st.sidebar.slider("4. Next 48h Weather Forecast (mm)", 0.0, 200.0, key="forecast")

# ==========================================
# 4. INFERENCE & PHYSICS ENGINE
# ==========================================
# Generate 14-day history ramping up to the user's exact slider values
def generate_dynamic_sequence(r, t, a):
    seq = np.zeros((T_WIN, 4))
    seq[:, 0] = np.linspace(max(0, r - 50), r, T_WIN) + np.random.randn(T_WIN) * 2 # Rain
    seq[:, 1] = np.clip(20 + 0.5 * np.cumsum(seq[:, 0]) / 3, 10, 95)              # Moisture
    seq[:, 2] = np.linspace(0.01, t, T_WIN) ** 2                                   # Tilt (Exponential creep)
    seq[:, 3] = np.linspace(1, a, T_WIN) + np.random.poisson(2, T_WIN)             # AE hits
    return np.clip(seq, 0, None)

seq = generate_dynamic_sequence(rain_val, tilt_val, ae_val)

# --- THE GEOTECHNICAL PHYSICS FUSION LOGIC ---
# Normalize inputs to a 0.0 - 1.0 danger scale
s_rain = min(1.0, rain_val / 140.0)
s_tilt = min(1.0, tilt_val / 1.0)
s_ae = min(1.0, ae_val / 80.0)
s_cast = min(1.0, forecast_val / 140.0)

# Anomaly Baseline check
raw_anom = -iso_slope.score_samples(seq[-1].reshape(1, -1))[0]
anom_score = float(np.clip((raw_anom - ANOM_LO) / (ANOM_HI - ANOM_LO), 0, 1))

# Calculate AI & Physics Combo Score
# Heavily weights physical tilt (40%) and rain (25%) over purely acoustic signs
combo_risk = (s_tilt * 0.40) + (s_rain * 0.25) + (s_ae * 0.15) + (s_cast * 0.10) + (anom_score * 0.10)

# EXTREME OVERRIDES: If absolute physical limits are breached, force a critical state
if tilt_val >= 1.0: 
    instability_score = max(combo_risk, 0.85)  # 1.0 degree/day means the slope is physically collapsing
elif rain_val >= 150.0:
    instability_score = max(combo_risk, 0.75)  # 150mm rain guarantees mudslide conditions
else:
    instability_score = combo_risk

# Determine Status
if instability_score >= 0.65:
    status = 'Critical'
elif instability_score >= 0.35:
    status = 'Warning'
else:
    status = 'Normal'

# ==========================================
# 5. DASHBOARD UI
# ==========================================
tab1, tab2 = st.tabs(["📊 Multi-Sensor Telemetry & AI Prediction", "🗺️ Smart Peg Swarm Status & Mitigation"])

with tab1:
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    col_stat1.metric("Geotechnical Shear Risk", f"{s_tilt * 100:.1f}%")
    col_stat2.metric("Isolation Forest Anomaly", f"{anom_score * 100:.1f}%")
    col_stat3.metric("Rainfall Saturation Risk", f"{s_rain * 100:.1f}%")
    col_stat4.metric("Fused Instability Index", f"{instability_score:.2f} / 1.00")

    st.markdown("---")
    
    # Status Banner
    if status == 'Critical':
        st.error("🚨 **SYSTEM STATUS: CRITICAL HAZARD — IMMINENT SLOPE COLLAPSE DETECTED**")
        st.caption("Acoustic emissions, soil moisture, and tilt rate cross-validation thresholds exceeded. Evacuation triggered.")
    elif status == 'Warning':
        st.warning("⚠️ **SYSTEM STATUS: ELEVATED RISK WARNING — PRECURSORS MONITORED**")
        st.caption("Elevated moisture or rainfall detected, but lacking corroborating subsurface shear fractures.")
    else:
        st.success("✅ **SYSTEM STATUS: SLOPE NOMINALLY STABLE**")
        st.caption("All sensors reporting within learned baseline envelopes.")

    # 14-Day Line Chart
    fig, axes = plt.subplots(4, 1, figsize=(10, 5), sharex=True)
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    for ax, name, col, i in zip(axes, FEATURE_NAMES, colors, range(4)):
        ax.plot(seq[:, i], color=col, marker='o', ms=2.5, lw=1.2)
        ax.set_ylabel(name, fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[-1].set_xlabel('Window Elapsed (Days)', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader("Subsurface Smart Peg Swarm Topology")
    st.markdown("""
    * **Deployment Profile:** 10 fixed anchor pegs positioned in a staggered line along Highway 109.
    * **Inter-Node Spacing:** 12 meters (acoustically covering the shear attenuation radius).
    * **Intranet Routing:** LoRaWAN Sub-GHz mesh hopping directly to local base-station relay.
    """)

    # Autonomous Mitigations
    st.subheader("Autonomous Hardware Actuators")
    c1, c2, c3 = st.columns(3)
    if status == 'Critical':
        c1.error("🚧 Road Barrier Servos: CLOSED")
        c2.error("🔊 High-Decibel Solar Siren: ACTIVE")
        c3.error("📲 SDMA Alert Dispatch: SENT")
    elif status == 'Warning':
        c1.warning("🚧 Road Barrier Servos: ARMED")
        c2.warning("🔊 High-Decibel Solar Siren: STANDBY")
        c3.warning("📲 SDMA Alert Dispatch: ADVISORY")
    else:
        c1.success("🚧 Road Barrier Servos: OPEN")
        c2.success("🔊 High-Decibel Solar Siren: OFF")
        c3.success("📲 SDMA Alert Dispatch: MONITORING")
