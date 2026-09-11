"""
CORE — Cognitive Operations and Risk Engine for Oil and Gas Industries
Anomaly Detection System  —  Tennessee Eastman Process
"""

import io, pickle, time, warnings, os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))

# ── PASSWORD GATE ─────────────────────────────────────────────────────────────
def _check_password():
    try:
        PASSWORD = st.secrets["app_password"]
    except (KeyError, FileNotFoundError):
        st.error("⚙️ App password not configured. Add `app_password` to Streamlit secrets.")
        st.stop()

    if st.session_state.get("authenticated"):
        return True

    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #0a1628 0%, #0D2137 100%); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { padding-top: 60px !important; }
    /* Input field — dark styling.
       BaseWeb nests the field as  stTextInputRootElement > base-input > input,
       and paints its default light fill on the middle wrapper — so the root is
       styled as the visible control and everything inside it is made
       transparent, otherwise that fill shows through. */
    [data-testid="stTextInputRootElement"] {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.22) !important;
        border-radius: 10px !important;
    }
    [data-testid="stTextInputRootElement"]:focus-within {
        border-color: #C9A84C !important;
    }
    [data-testid="stTextInputRootElement"] > div,
    [data-testid="stTextInputRootElement"] input {
        background: transparent !important;
    }
    .stTextInput input, .stTextInput input:focus {
        color: #ffffff !important;
        font-size: 15px !important;
        text-align: center !important;
        caret-color: #C9A84C !important;
        box-shadow: none !important;
    }
    .stTextInput input::placeholder { color: rgba(255,255,255,0.35) !important; }
    /* Show/hide-password button sits inside the field — keep it legible on dark */
    [data-testid="stTextInputRootElement"] button { color: rgba(255,255,255,0.55) !important; }
    [data-testid="stTextInputRootElement"] button:hover { color: #ffffff !important; }
    [data-testid="stTextInputRootElement"] button svg { fill: currentColor !important; }
    .stTextInput > label { display: none !important; }
    /* Primary button */
    .stButton > button[kind="primary"] {
        background: #C9A84C !important;
        border: none !important;
        color: #0a1628 !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        font-size: 15px !important;
        letter-spacing: 0.5px !important;
    }
    .stButton > button[kind="primary"]:hover { background: #e0bc62 !important; }
    </style>
    """, unsafe_allow_html=True)

    # Centre the card using columns
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        # Card container
        st.markdown("""
        <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.12);
                    border-radius:20px; padding:48px 40px 36px; text-align:center; margin-top:20px;">
            <div style="font-size:52px; margin-bottom:12px;">🛢️</div>
            <div style="color:#ffffff; font-size:28px; font-weight:700;
                        font-family:sans-serif; margin-bottom:6px;">CORE</div>
            <div style="color:rgba(255,255,255,0.5); font-size:13px;
                        font-family:sans-serif; margin-bottom:28px; line-height:1.6;">
                Anomaly Detection System<br>Enter access password to continue
            </div>
        </div>
        """, unsafe_allow_html=True)

        pwd = st.text_input("password", type="password", placeholder="Enter password…",
                            label_visibility="collapsed")
        if st.button("Access System", use_container_width=True, type="primary"):
            if pwd == PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")

        st.markdown("""
        <div style="color:rgba(255,255,255,0.2); font-size:11px; text-align:center; margin-top:20px;">
            Cognitive Operations and Risk Engine for Oil and Gas Industries
        </div>
        """, unsafe_allow_html=True)

    return False

if not _check_password():
    st.stop()
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CORE Anomaly Detection System",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
[data-testid="stSidebar"] { background:#0a3d62; }
[data-testid="stSidebar"] * { color:#fff !important; }
.alarm-red   { background:#fdecea; border-left:5px solid #e74c3c; padding:12px 18px;
               border-radius:6px; font-weight:700; color:#c0392b; font-size:1.1em; margin-bottom:8px;}
.alarm-green { background:#e8f8e8; border-left:5px solid #27ae60; padding:12px 18px;
               border-radius:6px; font-weight:700; color:#1e8449; font-size:1.1em; margin-bottom:8px;}
.alarm-warn  { background:#fff8e1; border-left:5px solid #f39c12; padding:12px 18px;
               border-radius:6px; font-weight:700; color:#d68910; font-size:1.1em; margin-bottom:8px;}
</style>
""", unsafe_allow_html=True)

VNAMES = {
    "xmeas_1":"A Feed Flow","xmeas_2":"D Feed Flow","xmeas_3":"E Feed Flow",
    "xmeas_4":"A+C Feed Flow","xmeas_5":"Recycle Flow","xmeas_6":"Reactor Feed Rate",
    "xmeas_7":"Reactor Pressure","xmeas_8":"Reactor Level",
    "xmeas_9":"Reactor Temperature","xmeas_10":"Purge Rate",
    "xmeas_11":"Prod Sep Temp","xmeas_12":"Prod Sep Level",
    "xmeas_13":"Prod Sep Pressure","xmeas_14":"Prod Sep Underflow",
    "xmeas_15":"Stripper Level","xmeas_16":"Stripper Pressure",
    "xmeas_17":"Stripper Underflow","xmeas_18":"Stripper Temp",
    "xmeas_19":"Stripper Steam Flow","xmeas_20":"Compressor Work",
    "xmeas_21":"Reactor CW Out Temp","xmeas_22":"Sep CW Out Temp",
    "xmeas_23":"Comp A (Str6)","xmeas_24":"Comp B (Str6)","xmeas_25":"Comp C (Str6)",
    "xmeas_26":"Comp D (Str6)","xmeas_27":"Comp E (Str6)","xmeas_28":"Comp F (Str6)",
    "xmeas_29":"Comp A (Str9)","xmeas_30":"Comp B (Str9)","xmeas_31":"Comp C (Str9)",
    "xmeas_32":"Comp D (Str9)","xmeas_33":"Comp E (Str9)","xmeas_34":"Comp F (Str9)",
    "xmeas_35":"Comp G (Str9)","xmeas_36":"Comp H (Str9)",
    "xmeas_37":"Comp D (Str11)","xmeas_38":"Comp E (Str11)","xmeas_39":"Comp F (Str11)",
    "xmeas_40":"Comp G (Str11)","xmeas_41":"Comp H (Str11)",
    "xmv_1":"D Feed Valve","xmv_2":"E Feed Valve","xmv_3":"A Feed Valve",
    "xmv_4":"A+C Feed Valve","xmv_5":"Compressor Recycle","xmv_6":"Purge Valve",
    "xmv_7":"Sep Liquid Valve","xmv_8":"Stripper Liquid Valve",
    "xmv_9":"Stripper Steam Valve","xmv_10":"Reactor CW Valve","xmv_11":"Condenser CW Valve",
}
FDESC = {
    0:"Normal Operation",1:"A/C Feed Ratio",2:"B Composition",
    3:"D Feed Temp (Hard)",4:"Reactor CW Temp",5:"Condenser CW Temp",
    6:"A Feed Loss",7:"C Header Pressure",8:"A,B,C Feed Comp Random",
    9:"D Feed Temp Random (Hard)",10:"C Feed Temp Random",11:"Reactor CW Random",
    12:"Condenser CW Random",13:"Reaction Kinetics Drift",14:"Reactor CW Valve Stick",
    15:"Condenser CW Valve Stick (Hard)",16:"Unknown 16",17:"Unknown 17",
    18:"Unknown 18",19:"Unknown 19",20:"Unknown 20",
}

FDESC_LAYMAN = {
    0:  "Everything is normal — no issues detected",
    1:  "Wrong gas mix arriving from upstream pipeline",
    2:  "Too much inert / waste gas in the feed",
    3:  "Feed stream suddenly too hot or too cold",
    4:  "Reactor cooling system failing — temperature rising",
    5:  "Condenser cooling system disrupted",
    6:  "Complete loss of primary gas supply",
    7:  "Main gas pipeline pressure has dropped dangerously",
    8:  "Inconsistent / unpredictable gas quality from source",
    9:  "Erratic temperature in secondary feed stream",
    10: "Unstable temperature in another feed stream",
    11: "Reactor cooling water behaving erratically",
    12: "Heat exchanger slowly losing effectiveness (fouling / scaling)",
    13: "Processing efficiency gradually declining (catalyst degrading)",
    14: "Reactor cooling valve is physically stuck — cannot move",
    15: "Condenser cooling valve is physically stuck — cannot move",
    16: "Novel / unclassified anomaly detected (Type 16)",
    17: "Novel / unclassified anomaly detected (Type 17)",
    18: "Novel / unclassified anomaly detected (Type 18)",
    19: "Novel / unclassified anomaly detected (Type 19)",
    20: "Novel / unclassified anomaly detected (Type 20)",
}

FDESC_CONSEQUENCE = {
    0:  "All sensors are within normal operating ranges. The plant is running as expected.",
    1:  ("The ratio of key gas components in the feed has suddenly shifted. "
         "This happens when the upstream pipeline switches gas sources, or when reservoir "
         "conditions change. Product quality may drift off specification if not corrected."),
    2:  ("The concentration of inert gases (e.g., nitrogen or CO₂) in the incoming feed has spiked. "
         "This dilutes the useful gas, reduces plant output, and can push the process outside its "
         "normal operating range."),
    3:  ("The temperature of a secondary feed stream has jumped suddenly. This likely means an upstream "
         "heat exchanger has failed or been bypassed. Sudden temperature changes stress equipment "
         "and affect reaction conditions downstream."),
    4:  ("The water supply used to cool the main reactor vessel has become warmer than normal. "
         "This could be a cooling tower failure, pump issue, or extreme weather overwhelming the system. "
         "If not addressed, reactor temperature will begin to rise — a direct safety risk."),
    5:  ("The condenser (which converts gas back to liquid for separation) is losing its cooling "
         "effectiveness. Separator pressure will rise and product flows will become unpredictable."),
    6:  ("The primary gas feed to the plant has been completely cut off. This is the most critical feed "
         "fault — it could be a pipeline rupture, emergency valve closure, or upstream supply failure. "
         "The plant cannot continue operating without primary feed."),
    7:  ("Pressure in a major gas supply header has dropped sharply. This indicates a possible pipeline "
         "breach, compressor station failure, or a large valve blowout. There is a risk of uncontrolled "
         "gas release to atmosphere — immediate emergency response is required."),
    8:  ("The composition of the incoming gas is fluctuating randomly. This suggests an unstable upstream "
         "source — a poorly controlled gas well, blending errors, or inconsistent supply. "
         "Makes the plant very difficult to control and maintain product quality."),
    9:  ("The temperature of a secondary feed stream is varying erratically. This points to an "
         "intermittently failing heat exchanger, a stuck bypass valve, or an unstable steam supply upstream."),
    10: ("Similar to Fault 9 but affecting a different feed stream. Likely an unstable upstream "
         "processing unit or a cooling issue on the feed preparation side."),
    11: ("The reactor cooling water temperature is swinging unpredictably. This could be a cooling pump "
         "cycling on/off, a partially blocked water line, or a faulty flow control valve. "
         "Makes reactor temperature control unstable."),
    12: ("The effectiveness of the condenser is slowly declining over time. This is the classic signature "
         "of heat exchanger fouling — mineral scale or deposits building up inside the tubes, "
         "gradually reducing heat transfer. Without AI monitoring, this goes completely unnoticed "
         "until a sudden major failure occurs."),
    13: ("The overall efficiency of the main processing reaction is slowly declining. In real life, this "
         "represents catalyst deactivation — the catalyst gradually losing its effectiveness. Product yield "
         "quietly drops over weeks. This is exactly the type of slow, hard-to-spot problem that AI "
         "monitoring is built to catch before it causes a shutdown."),
    14: ("The valve controlling cooling water flow to the reactor is physically stuck and cannot move. "
         "This is a serious mechanical failure — if stuck closed, the reactor overheats; if stuck open, "
         "the process is destabilised. Either scenario poses a safety risk requiring urgent maintenance."),
    15: ("The condenser cooling valve is physically stuck. Same mechanical failure type as Fault 14, "
         "but on the condenser. Loss of control over condenser cooling causes pressure to build in "
         "the separator system."),
    16: ("This is one of five undisclosed fault types in the benchmark dataset. The system has detected "
         "an anomaly pattern that deviates from normal operation. The contribution plot shows which "
         "sensors are behaving abnormally."),
    17: ("Undisclosed fault type. The system has detected unusual sensor behaviour. Review the "
         "contribution plot to identify which part of the plant is affected."),
    18: ("Undisclosed fault type. An anomaly pattern has been detected. The contribution plot below "
         "shows which sensors are most abnormal."),
    19: ("Undisclosed fault type. An anomaly has been detected in the process sensor readings. "
         "Physical inspection of the highlighted equipment is recommended."),
    20: ("Undisclosed fault type. The system has identified unusual behaviour — further investigation "
         "of the highlighted sensors is recommended."),
}

FAULT_ALARM_ACTION = {
    0:  "No action required. Continue monitoring.",
    1:  "Check upstream pipeline composition logs. Contact the gas supply operator. "
        "Adjust downstream process setpoints to compensate for the composition shift.",
    2:  "Identify and isolate the source of inert gas ingress. Check pipeline connections "
        "and upstream separation equipment. Consider reducing throughput until composition stabilises.",
    3:  "Inspect the upstream heat exchanger for the affected feed stream. Check utility supply "
        "(steam/cooling water). Verify bypass valves are in the correct position.",
    4:  "Inspect the cooling tower and cooling water pumps immediately. Check for blockages "
        "in the cooling water supply line. Reduce reactor load if temperature continues to rise.",
    5:  "Check condenser cooling water supply. Inspect the condenser for fouling. Monitor separator "
        "pressure closely — consider load reduction if pressure rises above limits.",
    6:  "🚨 EMERGENCY: Initiate controlled plant shutdown procedure. Locate and isolate the "
        "source of feed loss. Contact upstream pipeline operator. Do not restart until feed is confirmed safe.",
    7:  "🚨 EMERGENCY: Activate emergency isolation valves. Evacuate the area if safe to do so. "
        "Contact pipeline operator and safety team immediately.",
    8:  "Contact the upstream gas supplier about feed quality instability. Increase composition "
        "sampling frequency. Tighten control loops to compensate for variability.",
    9:  "Inspect the heat exchanger for the secondary feed stream. Check steam supply regulators "
        "and bypass valve positions. Schedule maintenance if the issue persists.",
    10: "Inspect upstream processing units feeding this stream. Check for ambient temperature effects "
        "and any cooling utility issues upstream.",
    11: "Inspect cooling water pumps and the flow control valve on the reactor cooling loop. "
        "Check for partial blockages or fouling in the cooling line.",
    12: "Schedule heat exchanger cleaning/inspection. The tubes need to be descaled or cleaned. "
        "Plan a shutdown window before the exchanger fails completely.",
    13: "Schedule catalyst inspection and replacement if confirmed. This is a planned maintenance "
        "action — product yield will continue to decline if left unchecked.",
    14: "⚠️ URGENT: Dispatch maintenance team to inspect and repair/replace the reactor cooling valve. "
        "Monitor reactor temperature manually. Prepare for controlled shutdown if temperature "
        "approaches safety limits.",
    15: "⚠️ URGENT: Dispatch maintenance team to inspect the condenser cooling valve. Monitor "
        "separator pressure manually. Prepare for load reduction if the valve cannot be freed.",
    16: "Investigate the contribution plot to identify which sensors are most abnormal. "
        "Conduct a physical inspection of the highlighted equipment. Log the event for analysis.",
    17: "Investigate the contribution plot. Conduct a physical inspection of the highlighted equipment.",
    18: "Investigate the contribution plot. Conduct a physical inspection of the highlighted equipment.",
    19: "Investigate the contribution plot. Conduct a physical inspection of the highlighted equipment.",
    20: "Investigate the contribution plot. Conduct a physical inspection of the highlighted equipment.",
}

FAULT_SEVERITY = {
    0:"Normal", 1:"Medium", 2:"Medium", 3:"High",
    4:"High", 5:"High", 6:"Critical", 7:"Critical",
    8:"Medium", 9:"Medium", 10:"Medium", 11:"Medium",
    12:"High", 13:"High", 14:"Critical", 15:"High",
    16:"Variable", 17:"Variable", 18:"Variable", 19:"Variable", 20:"Variable",
}

SEV_COLOR = {
    "Normal":"#1e8449", "Medium":"#2471a3",
    "High":"#d68910", "Critical":"#c0392b", "Variable":"#7f8c8d",
}

FAULT_INTRO = 160
PLAY_STEP   = 5   # samples advanced per auto-play tick

def fig2png(fig, dpi=72):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()   # return bytes, not BytesIO

def show_img(path):
    if os.path.exists(path):
        st.image(path, use_container_width=True)

def render_control_chart(samps, sensor_vals, T2_vals, Q_vals,
                          T2_UCL, Q_UCL, sensor_label, fault_num):
    """Fast 3-panel chart → PNG bytes. Uses DPI=65 for speed."""
    fig = plt.figure(figsize=(10, 6), facecolor="#f8f9fa")
    gs  = gridspec.GridSpec(3, 1, hspace=0.42)

    ax1 = fig.add_subplot(gs[0])
    pre  = samps < FAULT_INTRO; post = samps >= FAULT_INTRO
    if pre.any():  ax1.plot(samps[pre],  sensor_vals[pre],  "#1565C0", lw=1.6, label="Normal")
    if post.any(): ax1.plot(samps[post], sensor_vals[post], "#C62828", lw=1.6, label="Post-fault")
    if fault_num > 0: ax1.axvline(FAULT_INTRO, color="gray", ls="--", lw=1)
    ax1.set_ylabel(sensor_label, fontsize=7.5)
    ax1.set_title("Sensor Reading", fontsize=8.5, fontweight="bold", loc="left")
    ax1.legend(fontsize=6.5, loc="upper right")
    ax1.set_facecolor("#f8f9fa"); ax1.grid(ls=":", alpha=0.35)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(samps, T2_vals, "#6A1B9A", lw=1.6)
    ax2.axhline(T2_UCL, color="red", ls="--", lw=1.1, label=f"UCL={T2_UCL:.0f}")
    a2 = T2_vals > T2_UCL
    if a2.any(): ax2.fill_between(samps, T2_vals, T2_UCL, where=a2, color="#E57373", alpha=0.22)
    if fault_num > 0: ax2.axvline(FAULT_INTRO, color="gray", ls="--", lw=1)
    ax2.set_ylabel("T² Statistic", fontsize=7.5)
    ax2.set_title("Hotelling T² Control Chart", fontsize=8.5, fontweight="bold", loc="left")
    ax2.legend(fontsize=6.5, loc="upper right")
    ax2.set_facecolor("#f8f9fa"); ax2.grid(ls=":", alpha=0.35)

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(samps, Q_vals, "#E65100", lw=1.6)
    ax3.axhline(Q_UCL, color="red", ls="--", lw=1.1, label=f"UCL={Q_UCL:.1f}")
    a3 = Q_vals > Q_UCL
    if a3.any(): ax3.fill_between(samps, Q_vals, Q_UCL, where=a3, color="#FFB74D", alpha=0.22)
    if fault_num > 0: ax3.axvline(FAULT_INTRO, color="gray", ls="--", lw=1)
    ax3.set_ylabel("Q Statistic (SPE)", fontsize=7.5)
    ax3.set_xlabel("Sample Number", fontsize=7.5)
    ax3.set_title("Q (SPE) Control Chart", fontsize=8.5, fontweight="bold", loc="left")
    ax3.legend(fontsize=6.5, loc="upper right")
    ax3.set_facecolor("#f8f9fa"); ax3.grid(ls=":", alpha=0.35)

    return fig2png(fig, dpi=65)

@st.cache_resource
def load_models():
    with open(os.path.join(BASE,"scaler.pkl"),    "rb") as f: sc = pickle.load(f)
    with open(os.path.join(BASE,"lgbm_model.pkl"),"rb") as f: lg = pickle.load(f)
    with open(os.path.join(BASE,"pca_model.pkl"), "rb") as f: pc = pickle.load(f)
    return sc, lg, pc

@st.cache_data
def load_results():
    return (
        pd.read_csv(os.path.join(BASE,"tep_model_summary.csv")),
        pd.read_csv(os.path.join(BASE,"tep_per_fault_fdr.csv")),
        pd.read_csv(os.path.join(BASE,"mspc_vs_ml_summary.csv")),
        pd.read_csv(os.path.join(BASE,"mspc_per_fault.csv")),
    )

scaler, lgbm, pca_data = load_models()
ml_sum, ml_pf, comb_sum, _ = load_results()
T2_UCL = pca_data["T2_UCL"]
Q_UCL  = pca_data["Q_UCL"]

with open(os.path.join(BASE,"feature_cols.txt")) as f:
    FEAT_COLS = [l.strip() for l in f.readlines()]

with st.sidebar:
    st.markdown("## 🛢️ CORE")
    st.markdown("### Anomaly Detection System")
    st.caption("Cognitive Operations and Risk Engine for Oil and Gas Industries")
    st.markdown("---")
    page = st.radio("Go to", [
        "🏠 Overview",
        "🔴 Live Simulation",
        "📡 Monitor Status",
        "📊 Model Performance",
        "🔥 Per-Fault Analysis",
        "📋 Methodology",
    ])
    st.markdown("---")
    st.caption("Dataset: Tennessee Eastman Process\n"
               "Models: LightGBM + PCA-MSPC\n"
               "Variables: 52  |  Faults: 20")

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.title("🛢️ CORE ANOMALY DETECTION SYSTEM")
    st.markdown("#### Machine Learning + Multivariate Statistical Process Control")
    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Training Rows","5.25 M"); c2.metric("Process Variables","52")
    c3.metric("Fault Types","20");       c4.metric("Alarm Precision","99.5 %")
    st.markdown("---")
    st.subheader("Two-Tier Architecture")
    st.markdown("""
| Tier | Method | Role | Train Time |
|---|---|---|---|
| **1 – Screener** | PCA-MSPC (T² + Q) | Flags any deviation from normal — no labels needed | 1.2 s |
| **2 – Classifier** | LightGBM | Confirms fault when Tier 1 alarms | 57 s |
    """)
    st.markdown("---")
    row = ml_sum[ml_sum["Model"]=="LightGBM"].iloc[0]
    c1,c2,c3 = st.columns(3)
    c1.metric("Fault Detection Rate", f"{row['FDR (Recall)']:.1%}"); c1.metric("False Alarm Rate", f"{row['FAR']:.2%}")
    c2.metric("ROC-AUC", f"{row['ROC-AUC']:.4f}");                   c2.metric("Precision", f"{row['Precision']:.2%}")
    c3.metric("F1 Score", f"{row['F1']:.4f}");                        c3.metric("Train Time", f"{row['Train Time (s)']:.0f} s")
    st.markdown("---")
    show_img(os.path.join(BASE,"assets","fig1_overall_metrics.png"))

# ══════════════════════════════════════════════════════════════════════════════
# LIVE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
elif "Simulation" in page:
    st.title("🔴 Live Facility Simulation")
    st.markdown("Load a scenario → drag the slider or press **▶ Play**.")
    st.markdown("---")

    SS = st.session_state
    for k, v in [("sim_df",None),("sim_probs",None),("sim_T2",None),("sim_Q",None),
                 ("sim_t2c",None),("sim_qc",None),("sim_key",None),
                 ("sim_playing",False),("sim_frame",1),
                 ("sim_chart_cache",{}),   # {frame_idx: png_bytes}
                 ("sim_sensor_col","xmeas_9"),
                 ("prev_alarm", False),    # tracks last alarm state for sound trigger
                 ("alarm_sound", True)]:   # mute toggle
        if k not in SS: SS[k] = v

    ctrl, main = st.columns([1, 3])

    with ctrl:
        st.markdown("### Controls")
        fault_sel = st.selectbox(
            "Fault", list(FDESC.keys()),
            format_func=lambda x: f"Fault {x}: {FDESC[x]}" if x else "0: Normal"
        )
        run_sel = st.slider("Sim Run", 1, 5, 1)
        sensor_sel = st.selectbox(
            "Sensor", list(VNAMES.keys()),
            format_func=lambda x: VNAMES[x], index=8
        )
        speed_sel = st.select_slider("Speed", ["Slow","Normal","Fast"], value="Normal")
        DELAY = {"Slow":1.2,"Normal":0.5,"Fast":0.0}[speed_sel]

        load_btn  = st.button("📂  Load Scenario", type="primary", use_container_width=True)
        st.markdown("---")
        play_btn  = st.button("▶   Play",  use_container_width=True)
        pause_btn = st.button("⏸   Pause", use_container_width=True)
        reset_btn = st.button("⏮   Reset", use_container_width=True)
        st.markdown("---")
        SS.alarm_sound = st.checkbox("🔔  Alarm sound", value=SS.alarm_sound)

    if play_btn:  SS.sim_playing = True
    if pause_btn: SS.sim_playing = False
    if reset_btn: SS.sim_frame = 1; SS.sim_playing = False

    # sensor changed → clear chart cache so charts re-render with new sensor
    if sensor_sel != SS.sim_sensor_col:
        SS.sim_sensor_col  = sensor_sel
        SS.sim_chart_cache = {}

    # ── LOAD ──────────────────────────────────────────────────────────────────
    if load_btn:
        SS.sim_playing     = False
        SS.sim_frame       = 1
        SS.sim_key         = (fault_sel, run_sel)
        SS.sim_chart_cache = {}

        with main:
            # ── read CSV ──────────────────────────────────────────────────────
            fast_path = os.path.join(BASE, "fault_data",
                                     f"TEP_Fault{fault_sel:02d}_Testing.csv")

            if os.path.exists(fast_path):
                with st.spinner(f"Loading Fault {fault_sel} data…"):
                    df = pd.read_csv(fast_path)
                    df = df[df["simulationRun"] == run_sel].sort_values("sample").reset_index(drop=True)
            elif fault_sel == 0:
                st.error("Normal operation (Fault 0) data is not available in the online demo "
                         "due to file size limits. Please select **Fault 1–20** to run a simulation.")
                SS.sim_key = None
                df = pd.DataFrame()
            else:
                big_path = os.path.join(BASE, "TEP_Faulty_Testing.csv")
                if not os.path.exists(big_path):
                    st.error("Simulation data not found. Please run `prep_demo_data.py` locally "
                             "to generate the `fault_data/` files, then re-deploy.")
                    SS.sim_key = None
                    df = pd.DataFrame()
                else:
                    st.warning("Scanning full CSV — this may take a few minutes.")
                    bar = st.progress(0.0, text="Scanning…")
                    chunks, total = [], 0
                    file_size = os.path.getsize(big_path)
                    for chunk in pd.read_csv(big_path, chunksize=150_000):
                        mask = chunk["simulationRun"] == run_sel
                        if fault_sel > 0:
                            mask &= chunk["faultNumber"] == fault_sel
                        sub = chunk[mask]
                        if len(sub): chunks.append(sub)
                        total += len(chunk)
                        bar.progress(min(total / (file_size / 375), 1.0),
                                     text=f"Scanning… {total:,} rows")
                    bar.empty()
                    df = pd.concat(chunks).sort_values("sample").reset_index(drop=True) if chunks else pd.DataFrame()

            if df.empty:
                st.error(f"No data found for Fault {fault_sel}, Run {run_sel}.")
            else:
                # ── compute stats ─────────────────────────────────────────────
                with st.spinner("Computing predictions…"):
                    X      = df[FEAT_COLS].values.astype(np.float32)
                    Xs     = scaler.transform(X)
                    probs  = lgbm.predict_proba(Xs)[:, 1]
                    pca    = pca_data["pca"]; P = pca_data["loadings"]; lam = pca_data["eigenvalues"]
                    scores = pca.transform(Xs); Xr = pca.inverse_transform(scores)
                    T2     = np.sum((scores**2) / lam, axis=1)
                    Q      = np.sum((Xs - Xr)**2,      axis=1)
                    t2c    = (scores / np.sqrt(lam) @ P.T)**2
                    qc     = (Xs - Xr)**2

                SS.sim_df    = df
                SS.sim_probs = probs
                SS.sim_T2    = T2
                SS.sim_Q     = Q
                SS.sim_t2c   = t2c
                SS.sim_qc    = qc

                # ── pre-render every PLAY_STEP frames ─────────────────────────
                n      = len(df)
                samps  = df["sample"].values
                sensv  = df[sensor_sel].values
                slabel = VNAMES.get(sensor_sel, sensor_sel)
                cache  = {}
                idxs   = list(range(PLAY_STEP - 1, n, PLAY_STEP)) + [n - 1]
                prog   = st.progress(0.0, text="Pre-rendering charts…")
                for ki, idx in enumerate(idxs):
                    cache[idx] = render_control_chart(
                        samps[:idx+1], sensv[:idx+1],
                        T2[:idx+1],    Q[:idx+1],
                        T2_UCL, Q_UCL, slabel, fault_sel
                    )
                    prog.progress((ki+1)/len(idxs),
                                  text=f"Pre-rendering… {ki+1}/{len(idxs)}")
                prog.empty()
                SS.sim_chart_cache = cache
                st.success(f"✅  Loaded {n:,} samples — ready. Use slider or ▶ Play.")

    # ── DISPLAY ───────────────────────────────────────────────────────────────
    with main:
        if SS.sim_df is None:
            st.info("👈 Select a fault and click **📂 Load Scenario** to begin.")
        else:
            df    = SS.sim_df
            probs = SS.sim_probs
            T2a   = SS.sim_T2
            Qa    = SS.sim_Q
            t2c   = SS.sim_t2c
            qc    = SS.sim_qc
            n     = len(df)
            samps = df["sample"].values
            loaded_fault = SS.sim_key[0]

            SS.sim_frame = int(np.clip(SS.sim_frame, 1, n))

            frame = st.slider("◀  Drag to scrub  ▶", 1, n, SS.sim_frame)
            if frame != SS.sim_frame:
                SS.sim_frame   = frame
                SS.sim_playing = False

            i    = SS.sim_frame - 1
            samp = int(samps[i])
            prob = float(probs[i])
            T2   = float(T2a[i])
            Q    = float(Qa[i])

            if prob > 0.80 or T2 > T2_UCL or Q > Q_UCL:
                css, txt = "alarm-red",   "🚨 FAULT DETECTED"
            elif prob > 0.50 or T2 > T2_UCL*0.75 or Q > Q_UCL*0.75:
                css, txt = "alarm-warn",  "⚠️  ELEVATED RISK"
            else:
                css, txt = "alarm-green", "✅  NORMAL OPERATION"

            phase = "POST-FAULT" if loaded_fault > 0 and samp >= FAULT_INTRO else "Normal"
            st.markdown(
                f'<div class="{css}">{txt} &nbsp;|&nbsp; '
                f'Sample {samp}/{n} &nbsp;|&nbsp; {phase}</div>',
                unsafe_allow_html=True
            )

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("ML Fault Prob", f"{prob:.1%}",
                      delta="ALARM" if prob>0.5 else "OK",
                      delta_color="inverse" if prob>0.5 else "normal")
            m2.metric("T² Statistic", f"{T2:.1f}",
                      delta=f"UCL={T2_UCL:.0f}",
                      delta_color="inverse" if T2>T2_UCL else "normal")
            m3.metric("Q Statistic",  f"{Q:.2f}",
                      delta=f"UCL={Q_UCL:.1f}",
                      delta_color="inverse" if Q>Q_UCL else "normal")
            m4.metric("Sample", f"{samp}/{n}")

            # ── Alarm sound — fires once on False→True transition ─────────────
            alarm_active = bool(prob > 0.5 or T2 > T2_UCL or Q > Q_UCL)
            if alarm_active and not SS.prev_alarm and SS.alarm_sound:
                st.components.v1.html("""
                <script>
                (function() {
                    try {
                        var ctx = new (window.AudioContext || window.webkitAudioContext)();
                        function tone(freq, start, dur, vol) {
                            var o = ctx.createOscillator();
                            var g = ctx.createGain();
                            o.connect(g); g.connect(ctx.destination);
                            o.type = 'square';
                            o.frequency.value = freq;
                            g.gain.setValueAtTime(0.001, ctx.currentTime + start);
                            g.gain.exponentialRampToValueAtTime(vol, ctx.currentTime + start + 0.02);
                            g.gain.setValueAtTime(vol, ctx.currentTime + start + dur - 0.04);
                            g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + start + dur);
                            o.start(ctx.currentTime + start);
                            o.stop(ctx.currentTime + start + dur + 0.05);
                        }
                        /* Three double-pulse alarm blasts */
                        tone(960, 0.00, 0.14, 0.5);
                        tone(720, 0.15, 0.14, 0.5);
                        tone(960, 0.35, 0.14, 0.5);
                        tone(720, 0.50, 0.14, 0.5);
                        tone(960, 0.70, 0.14, 0.5);
                        tone(720, 0.85, 0.14, 0.5);
                    } catch(e) {}
                })();
                </script>
                """, height=0, scrolling=False)
            SS.prev_alarm = alarm_active

            # ── Layman alarm explanation ───────────────────────────────────────
            if loaded_fault > 0:
                sev   = FAULT_SEVERITY.get(loaded_fault, "Variable")
                scol  = SEV_COLOR.get(sev, "#7f8c8d")
                plain = FDESC_LAYMAN.get(loaded_fault, FDESC.get(loaded_fault, ""))
                with st.expander(
                    f"📖  Plain-English Explanation — Fault {loaded_fault}: {plain}",
                    expanded=alarm_active
                ):
                    la, lb = st.columns([1, 2])
                    with la:
                        st.markdown(
                            f"**Severity:**&nbsp; "
                            f"<span style='background:{scol};color:#fff;"
                            f"padding:2px 10px;border-radius:4px;font-weight:700;"
                            f"font-size:13px'>{sev}</span>",
                            unsafe_allow_html=True
                        )
                        st.markdown("")
                        st.markdown("**How the alarm works:**")
                        st.markdown(
                            "- **ML Fault Prob** — the AI's confidence (0–100%) that something is wrong. "
                            "Above **50%** = elevated concern; above **80%** = alarm.\n"
                            "- **T² Statistic** — measures how far the plant has moved from its "
                            "normal operating zone. Think of it as *'how unusual is the overall state?'*\n"
                            "- **Q Statistic (SPE)** — measures how much the current sensor readings "
                            "deviate from what the model expects. Think of it as *'how unexpected are "
                            "these readings?'*\n\n"
                            "An alarm fires when **any one** of the three exceeds its limit."
                        )
                    with lb:
                        st.markdown("**What is happening on the plant floor:**")
                        st.info(FDESC_CONSEQUENCE.get(loaded_fault, ""))
                        st.markdown("**Recommended action:**")
                        st.warning(FAULT_ALARM_ACTION.get(loaded_fault, "Investigate and log the event."))
            elif alarm_active:
                with st.expander("📖  How to read this alarm", expanded=True):
                    st.markdown(
                        "- **ML Fault Prob** above 50% means the AI classifier has detected a pattern "
                        "matching a known fault type.\n"
                        "- **T² above UCL** means the plant has moved outside its normal operating zone.\n"
                        "- **Q above UCL** means current sensor readings are inconsistent with normal behaviour.\n\n"
                        "Check the **Contribution Plot** below to see which sensors are driving the alarm."
                    )

            # ── look up nearest pre-rendered chart ────────────────────────────
            cache = SS.sim_chart_cache
            if cache:
                nearest = min(cache.keys(), key=lambda k: abs(k - i))
                if nearest in cache:
                    st.image(cache[nearest], use_container_width=True)
            else:
                # fallback: render on demand (slower, only if cache missing)
                sensv  = df[SS.sim_sensor_col].values
                slabel = VNAMES.get(SS.sim_sensor_col, SS.sim_sensor_col)
                png = render_control_chart(
                    samps[:i+1], sensv[:i+1], T2a[:i+1], Qa[:i+1],
                    T2_UCL, Q_UCL, slabel, loaded_fault
                )
                st.image(png, use_container_width=True)

            # ── contribution plot on alarm ────────────────────────────────────
            if prob > 0.5 or T2 > T2_UCL or Q > Q_UCL:
                with st.expander("📊 Contribution Plot — which sensors are causing the alarm?", expanded=True):
                    st.caption(
                        "The bars show which sensors are contributing most to the alarm. "
                        "**Taller red bars = that sensor is behaving most abnormally.** "
                        "These are the first instruments a field engineer should inspect."
                    )
                    fnames = [VNAMES.get(c, c) for c in FEAT_COLS]
                    nv     = len(FEAT_COLS)
                    t2ci   = t2c[i]; top_t2 = np.argsort(t2ci)[-5:][::-1]
                    qci    = qc[i];  top_q  = np.argsort(qci)[-5:][::-1]

                    fig2, (ca, cb) = plt.subplots(1, 2, figsize=(10, 3.2), facecolor="#f8f9fa")
                    fig2.suptitle(f"Sensor Contribution to Alarm — Sample {samp}",
                                  fontsize=9, fontweight="bold")
                    ca.bar(range(nv), t2ci,
                           color=["#C62828" if j in top_t2 else "#BBDEFB" for j in range(nv)])
                    ca.set_title("T² Contributions (Overall State Deviation)", fontsize=7.5, fontweight="bold")
                    ca.set_xticks(top_t2)
                    ca.set_xticklabels([fnames[j] for j in top_t2],
                                       rotation=35, ha="right", fontsize=6.5)
                    ca.set_facecolor("#f8f9fa"); ca.grid(ls=":", alpha=0.35)

                    cb.bar(range(nv), qci,
                           color=["#E65100" if j in top_q else "#C8E6C9" for j in range(nv)])
                    cb.set_title("Q Contributions (Unexpected Readings)", fontsize=7.5, fontweight="bold")
                    cb.set_xticks(top_q)
                    cb.set_xticklabels([fnames[j] for j in top_q],
                                       rotation=35, ha="right", fontsize=6.5)
                    cb.set_facecolor("#f8f9fa"); cb.grid(ls=":", alpha=0.35)
                    plt.tight_layout()
                    st.image(fig2png(fig2, dpi=72), use_container_width=True)

                    r1,r2 = st.columns(2)
                    with r1:
                        st.markdown("**🔴 Most abnormal sensors (Overall state):**")
                        for j in top_t2:
                            st.markdown(f"- **{fnames[j]}** — contribution score: `{t2ci[j]:.3f}`")
                    with r2:
                        st.markdown("**🟠 Most unexpected readings:**")
                        for j in top_q:
                            st.markdown(f"- **{fnames[j]}** — contribution score: `{qci[j]:.3f}`")
                    st.caption(
                        "💡 *Sensors appearing in both lists are the strongest candidates for the root cause of this alarm.*"
                    )

            # ── auto-play ─────────────────────────────────────────────────────
            if SS.sim_playing:
                nxt = SS.sim_frame + PLAY_STEP
                if nxt <= n:
                    if DELAY > 0: time.sleep(DELAY)
                    SS.sim_frame = nxt
                    st.rerun()
                else:
                    SS.sim_frame   = n
                    SS.sim_playing = False
                    st.success("Simulation complete.")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif "Performance" in page:
    st.title("📊 Model Performance")
    st.markdown("---")
    disp = comb_sum[["Model","Train Time (s)","FDR (Recall)","FAR","F1","ROC-AUC","MCC"]].copy()
    st.dataframe(
        disp.style
            .background_gradient(subset=["FDR (Recall)","ROC-AUC","F1"], cmap="Greens")
            .background_gradient(subset=["FAR"], cmap="Reds")
            .format({"Train Time (s)":"{:.1f}","FDR (Recall)":"{:.4f}",
                     "FAR":"{:.4f}","F1":"{:.4f}","ROC-AUC":"{:.4f}","MCC":"{:.4f}"}),
        use_container_width=True
    )
    st.markdown("---")
    t1,t2,t3,t4 = st.tabs(["FDR / FAR / AUC","Speed vs FDR","Confusion Matrices","Radar"])
    for tab, fn in zip([t1,t2,t3,t4],
                       ["fig1_overall_metrics.png","fig2_speed_vs_fdr.png",
                        "fig5_confusion_matrices.png","mspc_fig7_radar_all.png"]):
        with tab: show_img(os.path.join(BASE,"assets",fn))
    show_img(os.path.join(BASE,"assets","mspc_fig6_training_time.png"))

# ══════════════════════════════════════════════════════════════════════════════
# PER-FAULT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "Per-Fault" in page:
    st.title("🔥 Per-Fault Analysis")
    st.markdown("---")
    pf = ml_pf[ml_pf["Fault"] != "AVG"].copy()
    pf["Fault"]          = pf["Fault"].astype(int)
    pf["Technical Name"] = pf["Fault"].map(FDESC)
    pf["Plain English"]  = pf["Fault"].map(FDESC_LAYMAN)
    pf["Severity"]       = pf["Fault"].map(FAULT_SEVERITY)
    show_cols = ["Fault","Plain English","Severity","Logistic Regression","Random Forest",
                 "LightGBM","MLP Neural Network","Best Model","Difficulty"]
    st.dataframe(
        pf[show_cols].style
            .background_gradient(subset=["LightGBM","MLP Neural Network"],
                                 cmap="RdYlGn", vmin=0, vmax=1)
            .format({"Logistic Regression":"{:.4f}","Random Forest":"{:.4f}",
                     "LightGBM":"{:.4f}","MLP Neural Network":"{:.4f}"}),
        use_container_width=True, height=640
    )
    st.markdown("---")
    t1,t2,t3,t4 = st.tabs(["Bar Chart","Heatmap (ML)","Heatmap (All+MSPC)","Sensor Trends"])
    for tab, fn in zip([t1,t2,t3,t4],
                       ["fig4_per_fault_bar.png","fig3_per_fault_heatmap.png",
                        "mspc_fig5_heatmap_all.png","fig7_sensor_trends.png"]):
        with tab: show_img(os.path.join(BASE,"assets",fn))
    show_img(os.path.join(BASE,"assets","fig8_fault1_vs_fault3.png"))
    st.info(
        "**Faults 3, 9 and 15** are the hardest to detect — their signal magnitude is below "
        "natural process noise. In real life, these are equivalent to slow corrosion, gradual "
        "instrument drift, or a valve that is only slightly stuck. The AI catches them earlier "
        "than any human operator could."
    )

# ══════════════════════════════════════════════════════════════════════════════
# METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
elif "Methodology" in page:
    st.title("📋 Methodology & Deployment Roadmap")
    st.markdown("---")
    tab1,tab2,tab3,tab4 = st.tabs(["Dataset","Models & MSPC","PCA Analysis","Deployment Roadmap"])
    with tab1:
        st.markdown("""
| File | Rows | Purpose |
|---|---|---|
| TEP_FaultFree_Training | 250 000 | Normal operation baseline |
| TEP_Faulty_Training | 5 000 000 | 20 fault types for training |
| TEP_FaultFree_Testing | 480 000 | Normal operation evaluation |
| TEP_Faulty_Testing | 9 600 000 | 20 fault types for evaluation |

**52 variables:** 41 sensor measurements + 11 control valve positions.
Fault introduced at **sample 160**; samples 1–159 are normal.
        """)
        st.dataframe(pd.DataFrame([
            {
                "Fault": k,
                "Technical Name": v,
                "Plain English (What it means on the plant floor)": FDESC_LAYMAN.get(k, ""),
                "Severity": FAULT_SEVERITY.get(k, ""),
            }
            for k, v in FDESC.items() if k > 0
        ]), use_container_width=True, hide_index=True)
    with tab2:
        st.markdown(f"""
**MSPC control limits (99% confidence)**
- T² UCL = **{T2_UCL:.2f}** — alarm if process moves outside normal variation
- Q UCL = **{Q_UCL:.2f}** — alarm if process leaves the normal operating subspace

**ML models trained on 500 000 balanced samples** (250 k normal + 250 k faulty).
        """)
        show_img(os.path.join(BASE,"assets","mspc_fig3_overall_comparison.png"))
    with tab3:
        show_img(os.path.join(BASE,"assets","mspc_fig1_pca_variance.png"))
        show_img(os.path.join(BASE,"assets","mspc_fig2_control_charts.png"))
    with tab4:
        st.markdown("""
| Phase | Action | Timeline |
|---|---|---|
| **1 — Foundation** | Collect 6 months normal data from 5–10 pilot facilities | Months 1–3 |
| **2 — Screening** | Deploy MSPC screener; log contribution plots per alarm | Months 3–6 |
| **3 — Labelling** | Record inspection outcomes vs flagged anomalies | Months 6–18 |
| **4 — Classify** | Train LightGBM on labelled data; deploy Tier-2 | Months 18–24 |
| **5 — Expand** | Rolling-window features; full portfolio; retrain quarterly | Month 24+ |
        """)
        st.info("Same code, different data. The pipeline developed here runs unchanged on "
                "live plant operational data — only the CSV files need to be swapped.")

# ══════════════════════════════════════════════════════════════════════════════
# MONITOR STATUS  (live dashboard for the background monitor)
# ══════════════════════════════════════════════════════════════════════════════
elif "Monitor" in page:
    import subprocess
    from datetime import datetime, timedelta

    st.title("📡 Monitor Status Dashboard")
    st.markdown("Live view of the background anomaly monitor running on this machine.")
    st.markdown("---")

    ALARM_LOG  = os.path.join(BASE, "alarm_log.csv")
    MON_LOG    = os.path.join(BASE, "monitor.log")
    PID_FILE   = os.path.join(BASE, "monitor.pid")
    PAUSE_FILE = os.path.join(BASE, "monitor_paused.flag")
    PYTHON_EXE = r"C:\Users\user\anaconda3\python.exe"
    MON_SCRIPT = os.path.join(BASE, "monitor.py")

    # ── Helper: read PID from file ────────────────────────────────────────────
    def get_pid():
        try:
            with open(PID_FILE) as f:
                return int(f.read().strip())
        except Exception:
            return None

    def pid_alive(pid):
        if pid is None:
            return False
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5
            )
            return str(pid) in result.stdout
        except Exception:
            return False

    # ── Determine current state ────────────────────────────────────────────────
    pid     = get_pid()
    running = pid_alive(pid)
    paused  = os.path.exists(PAUSE_FILE)
    if not running:
        paused = False   # can't be paused if not running

    # ── Control buttons ───────────────────────────────────────────────────────
    st.subheader("🎛️ Controls")
    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button("▶  Start", use_container_width=True,
                     disabled=running, type="primary"):
            proc = subprocess.Popen(
                [PYTHON_EXE, MON_SCRIPT],
                cwd=BASE,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            time.sleep(1)
            st.success(f"Monitor started (PID {proc.pid})")
            st.rerun()

    with b2:
        if paused:
            if st.button("▶  Resume", use_container_width=True,
                         disabled=not running, type="primary"):
                if os.path.exists(PAUSE_FILE):
                    os.remove(PAUSE_FILE)
                st.success("Monitor resumed.")
                st.rerun()
        else:
            if st.button("⏸  Pause", use_container_width=True,
                         disabled=not running):
                open(PAUSE_FILE, "w").close()
                st.info("Monitor paused — no new batches will run until resumed.")
                st.rerun()

    with b3:
        if st.button("⏹  Stop", use_container_width=True,
                     disabled=not running):
            if pid:
                try:
                    subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                                   capture_output=True, timeout=5)
                except Exception:
                    pass
            if os.path.exists(PAUSE_FILE):
                os.remove(PAUSE_FILE)
            st.warning("Monitor stopped.")
            st.rerun()

    st.markdown("---")

    # ── Last log activity ─────────────────────────────────────────────────────
    last_seen   = None
    minutes_ago = None
    if os.path.exists(MON_LOG):
        try:
            with open(MON_LOG, "r", encoding="utf-8", errors="replace") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            if lines:
                last_line  = lines[-1]
                ts_str     = last_line[:19]
                last_seen  = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                minutes_ago = (datetime.now() - last_seen).total_seconds() / 60
        except Exception:
            pass

    # ── Status banner ─────────────────────────────────────────────────────────
    if running and paused:
        st.markdown(
            '<div class="alarm-warn">⏸ &nbsp; MONITOR PAUSED — detection suspended, press Resume to continue</div>',
            unsafe_allow_html=True
        )
    elif running:
        st.markdown(
            '<div class="alarm-green">✅ &nbsp; MONITOR IS RUNNING — background detection active</div>',
            unsafe_allow_html=True
        )
    elif last_seen and minutes_ago is not None and minutes_ago < 10:
        st.markdown(
            '<div class="alarm-warn">⚠️ &nbsp; MONITOR RECENTLY ACTIVE — process may be starting up</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="alarm-red">🔴 &nbsp; MONITOR NOT RUNNING — press Start or restart your PC</div>',
            unsafe_allow_html=True
        )

    st.markdown("")

    # ── Top metrics ───────────────────────────────────────────────────────────
    if os.path.exists(ALARM_LOG):
        try:
            log_df = pd.read_csv(ALARM_LOG, parse_dates=["timestamp"])
        except Exception:
            log_df = pd.DataFrame()
    else:
        log_df = pd.DataFrame()

    total_batches = len(log_df)
    total_alarms  = int(log_df["alerted"].sum()) if not log_df.empty else 0
    last_ts       = log_df["timestamp"].max() if not log_df.empty else None
    last_seen_str = last_seen.strftime("%Y-%m-%d %H:%M:%S") if last_seen else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monitor Status",   "Running ✅" if running else "Stopped 🔴")
    c2.metric("Batches Processed", f"{total_batches:,}")
    c3.metric("Alerts Sent",       f"{total_alarms:,}")
    c4.metric("Last Activity",     last_seen_str if last_seen else "—")

    st.markdown("---")

    # ── Recent alarm log table ────────────────────────────────────────────────
    st.subheader("📋 Detection Log")
    if log_df.empty:
        st.info(
            "No log entries yet. The monitor writes here as soon as it processes its first batch.\n\n"
            "If the monitor is running, entries will appear within 60 seconds."
        )
    else:
        display = log_df.copy().sort_values("timestamp", ascending=False).head(50)
        display["alerted"] = display["alerted"].map({True: "✉️ Sent", False: "—"})
        display["severity"] = display["severity"].map(
            lambda s: {"Critical":"🔴 Critical","High":"🟠 High",
                       "Medium":"🟡 Medium","Normal":"🟢 Normal"}.get(s, s)
        )
        display = display.rename(columns={
            "timestamp":  "Time",
            "source":     "Data Source",
            "fault_num":  "Fault",
            "severity":   "Severity",
            "ml_prob":    "AI Confidence",
            "T2":         "T² Stat",
            "Q":          "Q Stat",
            "T2_UCL":     "T² Limit",
            "Q_UCL":      "Q Limit",
            "alerted":    "Alert Sent",
        })
        st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Alarm trend chart ─────────────────────────────────────────────────────
    if not log_df.empty and len(log_df) > 1:
        st.markdown("---")
        st.subheader("📈 Detection Statistics Over Time")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), facecolor="#f8f9fa", sharex=True)
        fig.suptitle("AI Confidence & T² Statistic — All Batches", fontsize=9, fontweight="bold")

        x = range(len(log_df))
        ax1.plot(x, log_df["ml_prob"].astype(float) * 100, color="#C62828", lw=1.5)
        ax1.axhline(50, color="gray", ls="--", lw=1, label="50% threshold")
        ax1.set_ylabel("AI Confidence (%)", fontsize=7.5)
        ax1.set_ylim(0, 105)
        ax1.legend(fontsize=6.5); ax1.grid(ls=":", alpha=0.35); ax1.set_facecolor("#f8f9fa")

        t2_vals = log_df["T2"].astype(float)
        t2_ucl  = log_df["T2_UCL"].astype(float).iloc[0]
        ax2.plot(x, t2_vals, color="#6A1B9A", lw=1.5)
        ax2.axhline(t2_ucl, color="red", ls="--", lw=1, label=f"UCL={t2_ucl:.0f}")
        ax2.set_ylabel("T² Statistic", fontsize=7.5)
        ax2.set_xlabel("Batch Number", fontsize=7.5)
        ax2.legend(fontsize=6.5); ax2.grid(ls=":", alpha=0.35); ax2.set_facecolor("#f8f9fa")

        plt.tight_layout()
        st.image(fig2png(fig, dpi=72), use_container_width=True)

    # ── Live log tail ─────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("🖥️  Raw Monitor Log (last 30 lines)", expanded=False):
        if os.path.exists(MON_LOG):
            with open(MON_LOG, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            tail = "".join(lines[-30:])
            st.code(tail, language=None)
        else:
            st.info("monitor.log not found — monitor has not been started yet.")

    # ── Restart instructions ───────────────────────────────────────────────────
    if not running:
        st.markdown("---")
        st.warning(
            "**Monitor is not running.**  \n"
            "To restart: open the CORE project folder on your Desktop "
            "and double-click **`start_monitor.bat`**  \n"
            "Or restart your PC — it will start automatically on login."
        )

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(f"🔄 Auto-refreshing every 30s  |  Last checked: {datetime.now().strftime('%H:%M:%S')}")
    time.sleep(30)
    st.rerun()
