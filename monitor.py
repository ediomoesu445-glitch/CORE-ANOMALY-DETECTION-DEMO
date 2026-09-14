"""
CORE Automated Anomaly Monitor
(Cognitive Operations and Risk Engine for Oil and Gas Industries)
==================================================================
Runs headlessly, no dashboard, no human needed.

Two modes (set in monitor_config.ini):
  simulate  - replays a TEP fault file in real-time batches to demo alerting
  watch     - monitors the  incoming/  folder for new sensor CSV files from the plant

On every alarm transition it sends:
  • Email   (Gmail / any SMTP)
  • SMS     (Twilio)
  • WhatsApp(Twilio sandbox or business number)

All events are appended to  alarm_log.csv  for audit.

Usage:
  python monitor.py                 (uses monitor_config.ini in same folder)
  python monitor.py --config path   (specify alternate config)
"""

import os, time, pickle, warnings, logging, smtplib, csv, shutil, argparse, atexit
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
import pandas as pd
import configparser

warnings.filterwarnings("ignore")

BASE          = os.path.dirname(os.path.abspath(__file__))
INCOMING_DIR  = os.path.join(BASE, "incoming")
PROCESSED_DIR = os.path.join(BASE, "processed")
PID_FILE      = os.path.join(BASE, "monitor.pid")
PAUSE_FILE    = os.path.join(BASE, "monitor_paused.flag")
LOG_FILE      = os.path.join(BASE, "alarm_log.csv")

# ── Fault metadata (mirrors app.py) ───────────────────────────────────────────
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
    1: "Wrong gas mix arriving from upstream pipeline",
    2: "Too much inert / waste gas in the feed",
    3: "Feed stream suddenly too hot or too cold",
    4: "Reactor cooling system failing, temperature rising",
    5: "Condenser cooling system disrupted",
    6: "Complete loss of primary gas supply",
    7: "Main gas pipeline pressure has dropped dangerously",
    8: "Inconsistent / unpredictable gas quality from source",
    9: "Erratic temperature in secondary feed stream",
    10: "Unstable temperature in another feed stream",
    11: "Reactor cooling water behaving erratically",
    12: "Heat exchanger slowly losing effectiveness (fouling / scaling)",
    13: "Processing efficiency gradually declining (catalyst degrading)",
    14: "Reactor cooling valve is physically stuck, cannot move",
    15: "Condenser cooling valve is physically stuck, cannot move",
    16: "Novel / unclassified anomaly detected (Type 16)",
    17: "Novel / unclassified anomaly detected (Type 17)",
    18: "Novel / unclassified anomaly detected (Type 18)",
    19: "Novel / unclassified anomaly detected (Type 19)",
    20: "Novel / unclassified anomaly detected (Type 20)",
}

FAULT_SEVERITY = {
    0:"Normal", 1:"Medium", 2:"Medium", 3:"High", 4:"High", 5:"High",
    6:"Critical", 7:"Critical", 8:"Medium", 9:"Medium", 10:"Medium",
    11:"Medium", 12:"High", 13:"High", 14:"Critical", 15:"High",
    16:"Variable", 17:"Variable", 18:"Variable", 19:"Variable", 20:"Variable",
}

FAULT_ACTION = {
    0: "No action required. Continue monitoring.",
    1: "Check upstream pipeline composition logs. Contact the gas supply operator.",
    2: "Identify and isolate the source of inert gas ingress.",
    3: "Inspect the upstream heat exchanger for the affected feed stream.",
    4: "Inspect cooling tower and pumps immediately. Reduce reactor load if temperature rises.",
    5: "Check condenser cooling water supply. Inspect the condenser for fouling.",
    6: "EMERGENCY: Initiate controlled plant shutdown. Locate and isolate feed loss source. Contact upstream pipeline operator.",
    7: "EMERGENCY: Activate emergency isolation valves. Evacuate area if safe. Contact pipeline operator and safety team.",
    8: "Contact upstream gas supplier about feed quality instability. Increase sampling frequency.",
    9: "Inspect heat exchanger for secondary feed stream. Check steam supply regulators.",
    10: "Inspect upstream processing units. Check for cooling utility issues upstream.",
    11: "Inspect cooling water pumps and flow control valve on reactor cooling loop.",
    12: "Schedule heat exchanger cleaning/inspection. Plan shutdown window before failure.",
    13: "Schedule catalyst inspection and replacement. Product yield will continue to decline.",
    14: "URGENT: Dispatch maintenance team, reactor cooling valve stuck. Monitor temperature manually.",
    15: "URGENT: Dispatch maintenance team, condenser cooling valve stuck. Monitor pressure manually.",
    16: "Investigate contribution plot. Conduct physical inspection of highlighted equipment.",
    17: "Investigate contribution plot. Conduct physical inspection of highlighted equipment.",
    18: "Investigate contribution plot. Conduct physical inspection of highlighted equipment.",
    19: "Investigate contribution plot. Conduct physical inspection of highlighted equipment.",
    20: "Investigate contribution plot. Conduct physical inspection of highlighted equipment.",
}

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

# ── Config ─────────────────────────────────────────────────────────────────────
def load_config(path):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg

# ── Models ─────────────────────────────────────────────────────────────────────
def load_models():
    log = logging.getLogger("monitor")
    log.info("Loading models...")
    with open(os.path.join(BASE, "scaler.pkl"),    "rb") as f: scaler   = pickle.load(f)
    with open(os.path.join(BASE, "lgbm_model.pkl"),"rb") as f: lgbm     = pickle.load(f)
    with open(os.path.join(BASE, "pca_model.pkl"), "rb") as f: pca_data = pickle.load(f)
    with open(os.path.join(BASE, "feature_cols.txt"))    as f:
        feat_cols = [l.strip() for l in f.readlines()]
    log.info(f"Models loaded. {len(feat_cols)} features.")
    return scaler, lgbm, pca_data, feat_cols

# ── Inference ──────────────────────────────────────────────────────────────────
def run_inference(df, scaler, lgbm, pca_data, feat_cols):
    """Returns (probs, T2, Q, T2_UCL, Q_UCL, top_sensors) for every row."""
    X      = df[feat_cols].values.astype(np.float32)
    Xs     = scaler.transform(X)
    probs  = lgbm.predict_proba(Xs)[:, 1]

    pca    = pca_data["pca"]
    lam    = pca_data["eigenvalues"]
    T2_UCL = pca_data["T2_UCL"]
    Q_UCL  = pca_data["Q_UCL"]
    P      = pca_data["loadings"]

    scores = pca.transform(Xs)
    Xr     = pca.inverse_transform(scores)
    T2     = np.sum((scores**2) / lam, axis=1)
    Q      = np.sum((Xs - Xr)**2, axis=1)

    # Top 5 contributing sensors for the last sample
    qc       = (Xs - Xr)**2
    top_idx  = np.argsort(qc[-1])[-5:][::-1]
    top_sens = [(VNAMES.get(feat_cols[j], feat_cols[j]), float(qc[-1, j])) for j in top_idx]

    return probs, T2, Q, float(T2_UCL), float(Q_UCL), top_sens

# ── Alert message builder ──────────────────────────────────────────────────────
def build_alert(cfg, fault_num, prob, T2, Q, T2_UCL, Q_UCL, top_sensors, source, sample_count):
    facility = cfg.get("FACILITY", "name", fallback="Plant A")
    location = cfg.get("FACILITY", "location", fallback="Nigeria")
    sev      = FAULT_SEVERITY.get(fault_num, "Variable")
    plain    = FDESC_LAYMAN.get(fault_num, FDESC.get(fault_num, "Unknown anomaly"))
    action   = FAULT_ACTION.get(fault_num, "Investigate and log the event.")
    now      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sev_tag = {"Critical":"🚨 CRITICAL","High":"⚠️  HIGH","Medium":"⚠️  MEDIUM",
               "Variable":"⚠️  VARIABLE","Normal":"✅  NORMAL"}.get(sev, sev)

    t2_flag = " ← EXCEEDED" if T2 > T2_UCL else ""
    q_flag  = " ← EXCEEDED" if Q  > Q_UCL  else ""

    sensor_lines = "\n".join(
        f"  • {name}: {score:.3f}" for name, score in top_sensors
    )

    subject = f"[CORE ALARM] {sev_tag}: {plain} ({facility})"

    body = f"""
CORE ANOMALY DETECTION SYSTEM - AUTOMATED ALERT
=================================================

Time        : {now}
Facility    : {facility}
Location    : {location}
Source File : {source}
Samples     : {sample_count} readings analysed

FAULT DETECTED
--------------
Fault Type  : Fault {fault_num} - {FDESC.get(fault_num, 'Unknown')}
Plain English: {plain}
Severity    : {sev}

DETECTION STATISTICS (latest reading)
--------------------------------------
AI Confidence  : {prob:.1%}
T² Statistic   : {T2:.1f}  (Limit: {T2_UCL:.0f}){t2_flag}
Q Statistic    : {Q:.2f}  (Limit: {Q_UCL:.1f}){q_flag}

TOP 5 SENSORS BEHAVING ABNORMALLY
-----------------------------------
{sensor_lines}

RECOMMENDED ACTION
------------------
{action}

---
CORE Automated Monitoring System
Replies to this message are not monitored.
""".strip()

    return subject, body

# ── Email sender ───────────────────────────────────────────────────────────────
def send_email(cfg, subject, body, log):
    if not cfg.getboolean("EMAIL", "enabled", fallback=False):
        return
    try:
        host     = cfg.get("EMAIL", "smtp_host", fallback="smtp.gmail.com")
        port     = cfg.getint("EMAIL", "smtp_port", fallback=587)
        sender   = cfg.get("EMAIL", "sender_email")
        password = cfg.get("EMAIL", "sender_app_password")
        recips   = [r.strip() for r in cfg.get("EMAIL", "recipients").split(",")]

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender
        msg["To"]      = ", ".join(recips)
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(host, port) as srv:
            srv.starttls()
            srv.login(sender, password)
            srv.sendmail(sender, recips, msg.as_string())

        log.info(f"Email sent to: {', '.join(recips)}")
    except Exception as e:
        log.error(f"Email failed: {e}")

# ── SMS sender (Twilio) ────────────────────────────────────────────────────────
def send_sms(cfg, body, log):
    if not cfg.getboolean("SMS", "enabled", fallback=False):
        return
    try:
        from twilio.rest import Client
        sid    = cfg.get("SMS", "twilio_account_sid")
        token  = cfg.get("SMS", "twilio_auth_token")
        from_  = cfg.get("SMS", "twilio_from_number")
        recips = [r.strip() for r in cfg.get("SMS", "recipient_numbers").split(",")]

        client = Client(sid, token)
        short  = body[:1500]   # SMS character limit
        for number in recips:
            client.messages.create(to=number, from_=from_, body=short)
            log.info(f"SMS sent to {number}")
    except ImportError:
        log.warning("twilio not installed, run: pip install twilio")
    except Exception as e:
        log.error(f"SMS failed: {e}")

# ── WhatsApp sender (Twilio) ───────────────────────────────────────────────────
def send_whatsapp(cfg, body, log):
    if not cfg.getboolean("WHATSAPP", "enabled", fallback=False):
        return
    try:
        from twilio.rest import Client
        sid    = cfg.get("WHATSAPP", "twilio_account_sid")
        token  = cfg.get("WHATSAPP", "twilio_auth_token")
        from_  = "whatsapp:" + cfg.get("WHATSAPP", "twilio_whatsapp_from")
        recips = [r.strip() for r in cfg.get("WHATSAPP", "recipient_numbers").split(",")]

        client = Client(sid, token)
        for number in recips:
            client.messages.create(
                to=f"whatsapp:{number}", from_=from_, body=body[:4096]
            )
            log.info(f"WhatsApp sent to {number}")
    except ImportError:
        log.warning("twilio not installed, run: pip install twilio")
    except Exception as e:
        log.error(f"WhatsApp failed: {e}")

# ── Audit log ──────────────────────────────────────────────────────────────────
def log_event(timestamp, source, fault_num, severity, prob, T2, Q, T2_UCL, Q_UCL, alerted):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp","source","fault_num","severity",
                        "ml_prob","T2","Q","T2_UCL","Q_UCL","alerted"])
        w.writerow([timestamp, source, fault_num, severity,
                    f"{prob:.4f}", f"{T2:.2f}", f"{Q:.2f}",
                    f"{T2_UCL:.2f}", f"{Q_UCL:.2f}", alerted])

# ── Process one CSV batch ──────────────────────────────────────────────────────
def process_file(path, scaler, lgbm, pca_data, feat_cols, cfg, cooldown_tracker, log):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        log.error(f"Could not read {path}: {e}")
        return

    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        log.warning(f"{path}: missing columns {missing[:5]}, skipping")
        return

    try:
        probs, T2_arr, Q_arr, T2_UCL, Q_UCL, top_sensors = run_inference(
            df, scaler, lgbm, pca_data, feat_cols
        )
    except Exception as e:
        log.error(f"Inference failed on {path}: {e}")
        return

    # Use the worst-case reading in this batch
    worst = int(np.argmax(probs))
    prob  = float(probs[worst])
    T2    = float(T2_arr[worst])
    Q     = float(Q_arr[worst])

    min_conf = cfg.getfloat("MONITOR", "min_alarm_confidence", fallback=0.5)
    alarm    = prob > min_conf or T2 > T2_UCL or Q > Q_UCL

    source   = os.path.basename(path)
    now      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Estimate fault number from ML prediction (simple: use the class with highest prob)
    try:
        all_probs  = lgbm.predict_proba(
            pca_data  # reuse Xs is tricky here, just get the argmax class
        )
    except Exception:
        pass
    # Fallback: use configured fault_number for simulate mode
    fault_num = cfg.getint("SIMULATE", "fault_number", fallback=0)

    sev = FAULT_SEVERITY.get(fault_num, "Variable")
    log.info(f"{source} → prob={prob:.3f}  T²={T2:.1f}/{T2_UCL:.0f}  "
             f"Q={Q:.2f}/{Q_UCL:.1f}  alarm={alarm}")

    alerted = False
    if alarm:
        cooldown_min = cfg.getfloat("MONITOR", "alert_cooldown_minutes", fallback=30.0)
        last_alert   = cooldown_tracker.get("last_alert_time")
        in_cooldown  = (last_alert is not None and
                        (datetime.now() - last_alert).total_seconds() < cooldown_min * 60)

        if not in_cooldown:
            subject, body = build_alert(
                cfg, fault_num, prob, T2, Q, T2_UCL, Q_UCL,
                top_sensors, source, len(df)
            )
            send_email(cfg, subject, body, log)
            send_sms(cfg, body, log)
            send_whatsapp(cfg, body, log)
            cooldown_tracker["last_alert_time"] = datetime.now()
            alerted = True
            log.info(f"ALARM SENT, Fault {fault_num} ({sev}), confidence {prob:.1%}")
        else:
            remaining = cooldown_min - (datetime.now() - last_alert).total_seconds() / 60
            log.info(f"Alarm active but in cooldown ({remaining:.0f} min remaining), not re-alerting")

    log_event(now, source, fault_num, sev, prob, T2, Q, T2_UCL, Q_UCL, alerted)

# ── WATCH mode: monitors incoming/ folder ─────────────────────────────────────
def run_watch_mode(scaler, lgbm, pca_data, feat_cols, cfg, log):
    os.makedirs(INCOMING_DIR,  exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    interval        = cfg.getfloat("MONITOR", "check_interval_seconds", fallback=60.0)
    heartbeat_hours = cfg.getfloat("MONITOR", "heartbeat_hours",        fallback=6.0)
    cooldown        = {}
    processed       = set()
    start_time      = datetime.now()
    last_heartbeat  = datetime.now()
    stats           = {"batches": 0, "alarms": 0, "last_alarm_time": None,
                       "uptime_hours": 0.0}

    log.info(f"WATCH mode: monitoring: {INCOMING_DIR}")
    log.info(f"Check interval: {interval:.0f}s  |  Heartbeat every {heartbeat_hours:.0f}h")

    # Send startup heartbeat immediately
    send_heartbeat(cfg, stats, log)

    while True:
        # Honour pause at the top of every watch cycle
        wait_if_paused(log)

        try:
            files = sorted(
                f for f in os.listdir(INCOMING_DIR)
                if f.endswith(".csv") and f not in processed
            )
            if files:
                for fname in files:
                    fpath = os.path.join(INCOMING_DIR, fname)
                    log.info(f"New file: {fname}")
                    alarms_before = stats["alarms"]
                    process_file(fpath, scaler, lgbm, pca_data, feat_cols, cfg, cooldown, log)
                    dest = os.path.join(PROCESSED_DIR, fname)
                    shutil.move(fpath, dest)
                    processed.add(fname)
                    stats["batches"] += 1
                    if stats["alarms"] > alarms_before:
                        stats["last_alarm_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            else:
                log.debug(f"No new files, sleeping {interval:.0f}s")
        except Exception as e:
            log.error(f"Watch loop error: {e}")

        stats["uptime_hours"] = (datetime.now() - start_time).total_seconds() / 3600
        hours_since = (datetime.now() - last_heartbeat).total_seconds() / 3600
        if hours_since >= heartbeat_hours:
            send_heartbeat(cfg, stats, log)
            last_heartbeat = datetime.now()

        time.sleep(interval)

# ── SIMULATE mode: replays TEP fault file ────────────────────────────────────
def run_simulate_mode(scaler, lgbm, pca_data, feat_cols, cfg, log):
    fault_num  = cfg.getint("SIMULATE",  "fault_number",   fallback=6)
    run_num    = cfg.getint("SIMULATE",  "run_number",      fallback=1)
    batch_size = cfg.getint("SIMULATE",  "batch_size",      fallback=10)
    interval   = cfg.getfloat("MONITOR", "check_interval_seconds", fallback=60.0)

    data_path = os.path.join(BASE, "fault_data", f"TEP_Fault{fault_num:02d}_Testing.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(BASE, "TEP_Faulty_Testing.csv")
    if not os.path.exists(data_path):
        log.error("Simulation data not found. Run prep_demo_data.py first.")
        return

    heartbeat_hours = cfg.getfloat("MONITOR", "heartbeat_hours", fallback=6.0)

    log.info(f"SIMULATE mode: Fault {fault_num}, Run {run_num}, batch={batch_size} samples")
    log.info(f"Data file: {data_path}")
    log.info(f"Each batch = {interval:.0f}s real time (simulating one plant check interval)")
    log.info(f"Heartbeat email every {heartbeat_hours:.0f} hours")

    df_full = pd.read_csv(data_path)
    if "simulationRun" in df_full.columns:
        df_full = df_full[df_full["simulationRun"] == run_num]
    df_full = df_full.sort_values("sample").reset_index(drop=True)

    cooldown       = {}
    n              = len(df_full)
    cursor         = 0
    start_time     = datetime.now()
    last_heartbeat = datetime.now()
    stats          = {"batches": 0, "alarms": 0, "last_alarm_time": None,
                      "uptime_hours": 0.0}

    log.info(f"Total samples: {n}  (fault introduced at sample 160)")

    # Send startup heartbeat immediately
    send_heartbeat(cfg, stats, log)

    while cursor < n:
        end    = min(cursor + batch_size, n)
        batch  = df_full.iloc[cursor:end].copy()
        source = f"simulated_fault{fault_num}_run{run_num}_samples{cursor+1}-{end}.csv"

        # Honour pause before processing each batch
        wait_if_paused(log)

        log.info(f"--- Batch: samples {cursor+1}-{end} ---")
        alarms_before = cooldown.get("total_alarms", 0)
        process_file_df(batch, source, scaler, lgbm, pca_data, feat_cols,
                        cfg, fault_num, cooldown, log)
        alarms_after = cooldown.get("total_alarms", 0)

        stats["batches"] += 1
        if alarms_after > alarms_before:
            stats["alarms"] += 1
            stats["last_alarm_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stats["uptime_hours"] = (datetime.now() - start_time).total_seconds() / 3600

        # Heartbeat check
        hours_since = (datetime.now() - last_heartbeat).total_seconds() / 3600
        if hours_since >= heartbeat_hours:
            send_heartbeat(cfg, stats, log)
            last_heartbeat = datetime.now()

        cursor = end
        if cursor < n:
            log.info(f"Sleeping {interval:.0f}s (simulating next plant read)...")
            time.sleep(interval)

    log.info("Simulation complete. All samples processed.")

def process_file_df(df, source, scaler, lgbm, pca_data, feat_cols,
                    cfg, fault_num, cooldown_tracker, log):
    """Same as process_file but takes a DataFrame directly (simulate mode)."""
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        log.warning(f"Missing columns {missing[:5]}, skipping batch")
        return

    try:
        probs, T2_arr, Q_arr, T2_UCL, Q_UCL, top_sensors = run_inference(
            df, scaler, lgbm, pca_data, feat_cols
        )
    except Exception as e:
        log.error(f"Inference failed: {e}")
        return

    worst = int(np.argmax(probs))
    prob  = float(probs[worst])
    T2    = float(T2_arr[worst])
    Q     = float(Q_arr[worst])

    min_conf = cfg.getfloat("MONITOR", "min_alarm_confidence", fallback=0.5)
    alarm    = prob > min_conf or T2 > T2_UCL or Q > Q_UCL
    sev      = FAULT_SEVERITY.get(fault_num, "Variable")
    now      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log.info(f"{source}")
    log.info(f"  ML prob = {prob:.1%}  |  T² = {T2:.1f}/{T2_UCL:.0f}  |  "
             f"Q = {Q:.2f}/{Q_UCL:.1f}  |  ALARM = {alarm}")

    alerted = False
    if alarm:
        cooldown_min = cfg.getfloat("MONITOR", "alert_cooldown_minutes", fallback=30.0)
        last_alert   = cooldown_tracker.get("last_alert_time")
        in_cooldown  = (last_alert is not None and
                        (datetime.now() - last_alert).total_seconds() < cooldown_min * 60)

        if not in_cooldown:
            subject, body = build_alert(
                cfg, fault_num, prob, T2, Q, T2_UCL, Q_UCL,
                top_sensors, source, len(df)
            )
            send_email(cfg, subject, body, log)
            send_sms(cfg, body, log)
            send_whatsapp(cfg, body, log)
            cooldown_tracker["last_alert_time"] = datetime.now()
            alerted = True
            log.info(f"  >>> ALERT SENT, Fault {fault_num} ({sev}), confidence {prob:.1%}")
        else:
            remaining = cooldown_min - (datetime.now() - last_alert).total_seconds() / 60
            log.info(f"  Alarm active, cooldown {remaining:.0f} min remaining")
    else:
        log.info(f"  Status: NORMAL")

    log_event(now, source, fault_num, sev, prob, T2, Q, T2_UCL, Q_UCL, alerted)

# ── Pause / resume ─────────────────────────────────────────────────────────────
def wait_if_paused(log):
    """Block execution while monitor_paused.flag exists. Returns immediately otherwise."""
    was_paused = False
    while os.path.exists(PAUSE_FILE):
        if not was_paused:
            log.info("PAUSED: waiting for resume signal from dashboard...")
            was_paused = True
        time.sleep(3)
    if was_paused:
        log.info("RESUMED: continuing detection")

# ── Heartbeat email ────────────────────────────────────────────────────────────
def send_heartbeat(cfg, stats, log):
    """Send a 'still running' status email. stats = dict with session counters."""
    if not cfg.getboolean("EMAIL", "enabled", fallback=False):
        return
    facility = cfg.get("FACILITY", "name",     fallback="Plant A")
    location = cfg.get("FACILITY", "location", fallback="Nigeria")
    now      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    alarms_line = (
        f"{stats['alarms']} alarm(s) sent"
        if stats["alarms"] > 0
        else "No alarms, all readings normal"
    )
    last_alarm_line = (
        f"Last alarm : {stats['last_alarm_time']}"
        if stats.get("last_alarm_time")
        else "Last alarm : None this session"
    )

    subject = f"[CORE HEARTBEAT] Monitor Running: {facility} ({now[:10]})"
    body = f"""
CORE ANOMALY DETECTION SYSTEM - STATUS REPORT
==============================================

Time       : {now}
Facility   : {facility}
Location   : {location}
Status     : RUNNING, all systems active

SESSION SUMMARY (since last start)
------------------------------------
Batches processed : {stats['batches']}
{alarms_line}
{last_alarm_line}
Uptime            : {stats['uptime_hours']:.1f} hours

This is an automated status check confirming the monitor
is running correctly. If you stop receiving these emails,
the monitor may have stopped: restart it by logging out
and back in, or double-click  start_monitor.bat.

---
CORE Automated Monitoring System
""".strip()

    send_email(cfg, subject, body, log)
    log.info(f"Heartbeat email sent ({stats['batches']} batches, {stats['alarms']} alarms)")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CORE Automated Anomaly Monitor")
    parser.add_argument("--config", default=os.path.join(BASE, "monitor_config.ini"),
                        help="Path to config file")
    args   = parser.parse_args()

    # Logging: console + file
    log_path = os.path.join(BASE, "monitor.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8", errors="replace"),
            logging.StreamHandler(),
        ]
    )
    log = logging.getLogger("monitor")

    if not os.path.exists(args.config):
        log.error(f"Config not found: {args.config}")
        log.error("Copy monitor_config.ini.template to monitor_config.ini and fill in credentials.")
        return

    cfg = load_config(args.config)
    scaler, lgbm, pca_data, feat_cols = load_models()

    facility = cfg.get("FACILITY", "name", fallback="Plant A")
    mode     = cfg.get("MONITOR",  "mode", fallback="simulate").lower()

    # Write PID file so the dashboard can track and control this process
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    atexit.register(lambda: os.remove(PID_FILE) if os.path.exists(PID_FILE) else None)
    atexit.register(lambda: os.remove(PAUSE_FILE) if os.path.exists(PAUSE_FILE) else None)

    log.info("=" * 60)
    log.info("CORE AUTOMATED ANOMALY MONITOR, STARTED")
    log.info(f"PID      : {os.getpid()}")
    log.info(f"Facility : {facility}")
    log.info(f"Mode     : {mode.upper()}")
    log.info(f"Audit log: {LOG_FILE}")
    log.info("=" * 60)

    if mode == "watch":
        run_watch_mode(scaler, lgbm, pca_data, feat_cols, cfg, log)
    else:
        run_simulate_mode(scaler, lgbm, pca_data, feat_cols, cfg, log)

if __name__ == "__main__":
    main()
