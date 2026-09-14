"""
One-time script: splits TEP_Faulty_Testing.csv into 20 per-fault files.
Writes each row to disk immediately, uses only ONE CHUNK of RAM at a time.
Safe to interrupt and re-run; already-complete files are skipped.
"""

import os, pandas as pd

BASE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "fault_data")
os.makedirs(OUT_DIR, exist_ok=True)

# ── fault-free ────────────────────────────────────────────────────────────────
ff_dst = os.path.join(OUT_DIR, "TEP_Fault00_Testing.csv")
if os.path.exists(ff_dst):
    print("TEP_Fault00_Testing.csv already done, skipping.")
else:
    print("Copying fault-free testing file...")
    df = pd.read_csv(os.path.join(BASE, "TEP_FaultFree_Testing.csv"))
    df.to_csv(ff_dst, index=False)
    print(f"  Saved: {len(df):,} rows")

# ── faulty: stream straight to 20 CSV files ───────────────────────────────────
src = os.path.join(BASE, "TEP_Faulty_Testing.csv")

# find which faults still need writing
needed = []
for fn in range(1, 21):
    dst = os.path.join(OUT_DIR, f"TEP_Fault{fn:02d}_Testing.csv")
    if not os.path.exists(dst):
        needed.append(fn)

if not needed:
    print("All 20 per-fault files already exist. Nothing to do.")
else:
    print(f"\nFaults still needed: {needed}")
    print(f"Streaming {src}  ({os.path.getsize(src)/1e9:.1f} GB)…\n")

    # open one file handle per fault that needs writing
    handles = {}
    wrote_header = set()
    for fn in needed:
        dst = os.path.join(OUT_DIR, f"TEP_Fault{fn:02d}_Testing.csv")
        handles[fn] = open(dst, "w", newline="")

    total = 0
    TOTAL_ROWS = 9_600_000
    for chunk in pd.read_csv(src, chunksize=200_000):
        for fn in needed:
            sub = chunk[chunk["faultNumber"] == fn]
            if len(sub):
                sub.to_csv(handles[fn], index=False, header=(fn not in wrote_header))
                wrote_header.add(fn)
        total += len(chunk)
        pct = total / TOTAL_ROWS
        bar = "#" * int(pct * 40)
        print(f"\r  [{bar:<40}] {pct:5.1%}  {total:>9,} / {TOTAL_ROWS:,} rows",
              end="", flush=True)

    for fn, fh in handles.items():
        fh.close()

    print("\n\nFiles written:")
    for fn in needed:
        dst = os.path.join(OUT_DIR, f"TEP_Fault{fn:02d}_Testing.csv")
        mb = os.path.getsize(dst) / 1_048_576
        print(f"  Fault {fn:02d}: {mb:.0f} MB  →  {dst}")

print("\nDone.")
