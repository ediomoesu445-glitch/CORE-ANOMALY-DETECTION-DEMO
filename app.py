"""
NMDPRA ANOMALY DETECTION SYSTEM  —  Tennessee Eastman Process
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

st.set_page_config(
    page_title="NMDPRA Anomaly Detection System",
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
        st.image(path, width="stretch")

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
    st.markdown("## 🛢️ NMDPRA")
    st.markdown("### Anomaly Detection System")
    st.markdown("---")
    page = st.radio("Go to", [
        "🏠 Overview",
        "🔴 Live Simulation",
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
    st.title("🛢️ NMDPRA ANOMALY DETECTION SYSTEM")
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
                 ("sim_sensor_col","xmeas_9")]:
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

            # ── look up nearest pre-rendered chart ────────────────────────────
            cache = SS.sim_chart_cache
            if cache:
                nearest = min(cache.keys(), key=lambda k: abs(k - i))
                if nearest in cache:
                    st.image(cache[nearest], width="stretch")
            else:
                # fallback: render on demand (slower, only if cache missing)
                sensv  = df[SS.sim_sensor_col].values
                slabel = VNAMES.get(SS.sim_sensor_col, SS.sim_sensor_col)
                png = render_control_chart(
                    samps[:i+1], sensv[:i+1], T2a[:i+1], Qa[:i+1],
                    T2_UCL, Q_UCL, slabel, loaded_fault
                )
                st.image(png, width="stretch")

            # ── contribution plot on alarm ────────────────────────────────────
            if prob > 0.5 or T2 > T2_UCL or Q > Q_UCL:
                with st.expander("📊 Contribution Plot — what is driving this alarm?", expanded=True):
                    fnames = [VNAMES.get(c, c) for c in FEAT_COLS]
                    nv     = len(FEAT_COLS)
                    t2ci   = t2c[i]; top_t2 = np.argsort(t2ci)[-5:][::-1]
                    qci    = qc[i];  top_q  = np.argsort(qci)[-5:][::-1]

                    fig2, (ca, cb) = plt.subplots(1, 2, figsize=(10, 3.2), facecolor="#f8f9fa")
                    fig2.suptitle(f"Contribution Plot — Sample {samp}",
                                  fontsize=9, fontweight="bold")
                    ca.bar(range(nv), t2ci,
                           color=["#C62828" if j in top_t2 else "#BBDEFB" for j in range(nv)])
                    ca.set_title("T² Contributions", fontsize=7.5, fontweight="bold")
                    ca.set_xticks(top_t2)
                    ca.set_xticklabels([fnames[j] for j in top_t2],
                                       rotation=35, ha="right", fontsize=6.5)
                    ca.set_facecolor("#f8f9fa"); ca.grid(ls=":", alpha=0.35)

                    cb.bar(range(nv), qci,
                           color=["#E65100" if j in top_q else "#C8E6C9" for j in range(nv)])
                    cb.set_title("Q (SPE) Contributions", fontsize=7.5, fontweight="bold")
                    cb.set_xticks(top_q)
                    cb.set_xticklabels([fnames[j] for j in top_q],
                                       rotation=35, ha="right", fontsize=6.5)
                    cb.set_facecolor("#f8f9fa"); cb.grid(ls=":", alpha=0.35)
                    plt.tight_layout()
                    st.image(fig2png(fig2, dpi=72), width="stretch")

                    r1,r2 = st.columns(2)
                    with r1:
                        st.markdown("**Top T² contributors:**")
                        for j in top_t2: st.markdown(f"- {fnames[j]}: `{t2ci[j]:.3f}`")
                    with r2:
                        st.markdown("**Top Q contributors:**")
                        for j in top_q: st.markdown(f"- {fnames[j]}: `{qci[j]:.3f}`")

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
            .background_gradient(subset=["FAR"], cmap="Reds_r")
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
    pf["Fault"]       = pf["Fault"].astype(int)
    pf["Description"] = pf["Fault"].map(FDESC)
    show_cols = ["Fault","Description","Logistic Regression","Random Forest",
                 "LightGBM","MLP Neural Network","Best Model","Difficulty"]
    st.dataframe(
        pf[show_cols].style
            .background_gradient(subset=["LightGBM","MLP Neural Network"],
                                 cmap="RdYlGn", vmin=0, vmax=1)
            .format({"Logistic Regression":"{:.4f}","Random Forest":"{:.4f}",
                     "LightGBM":"{:.4f}","MLP Neural Network":"{:.4f}"}),
        use_container_width=True, height=600
    )
    st.markdown("---")
    t1,t2,t3,t4 = st.tabs(["Bar Chart","Heatmap (ML)","Heatmap (All+MSPC)","Sensor Trends"])
    for tab, fn in zip([t1,t2,t3,t4],
                       ["fig4_per_fault_bar.png","fig3_per_fault_heatmap.png",
                        "mspc_fig5_heatmap_all.png","fig7_sensor_trends.png"]):
        with tab: show_img(os.path.join(BASE,"assets",fn))
    show_img(os.path.join(BASE,"assets","fig8_fault1_vs_fault3.png"))
    st.info("Faults 3, 9 and 15 are 'hard' — magnitude below natural process noise. "
            "Analogous to slow corrosion or gradual instrument drift in petroleum operations.")

# ══════════════════════════════════════════════════════════════════════════════
# METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
elif "Methodology" in page:
    st.title("📋 Methodology & NMDPRA Roadmap")
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
        st.dataframe(pd.DataFrame(
            [{"Fault":k,"Description":v} for k,v in FDESC.items() if k>0]
        ), use_container_width=True, hide_index=True)
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
                "NMDPRA operational data — only the CSV files need to be swapped.")
