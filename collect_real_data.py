"""
collect_real_data.py
---------------------
Collects REAL F1 race data using the FastF1 library.
Pulls starting tyre compound, grid position, weather, and team
for every driver in every dry race from 2018–2024.

Usage:
    python collect_real_data.py

Output:
    data/raw/f1_real_data.csv   ← use this with main.py

Install first:
    pip install fastf1
"""

import os
import time
import warnings
import pandas as pd
import fastf1

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────
SEASONS      =[2018, 2019, 2020, 2021, 2022, 2023, 2024]
CACHE_DIR    = "data/fastf1_cache"
OUTPUT_PATH  = "data/raw/f1_real_data.csv"

# Map FastF1 team names → integer codes (consistent with our model)
TEAM_MAP = {
    "Red Bull Racing":    1,
    "Red Bull":           1,
    "Ferrari":            2,
    "Scuderia Ferrari":   2,
    "Mercedes":           3,
    "Mercedes AMG":       3,
    "McLaren":            4,
    "McLaren F1 Team":    4,
    "Aston Martin":       5,
    "Aston Martin F1 Team": 5,
    "Racing Point":       5,
    "Force India":        5,
    "Alpine":             6,
    "Alpine F1 Team":     6,
    "Renault":            6,
    "Williams":           7,
    "Williams Racing":    7,
    "AlphaTauri":         8,
    "Scuderia AlphaTauri": 8,
    "RB F1 Team":         8,
    "Toro Rosso":         8,
    "Alfa Romeo":         9,
    "Alfa Romeo Racing":  9,
    "Kick Sauber":        9,
    "Sauber":             9,
    "Haas":               10,
    "Haas F1 Team":       10,
}

COMPOUND_MAP = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("data/raw", exist_ok=True)

# Enable FastF1 cache (avoids re-downloading data)
fastf1.Cache.enable_cache(CACHE_DIR)

records = []
skipped = []

print("=" * 60)
print("  F1Start-AI — Real Data Collector (FastF1)")
print(f"  Seasons: {SEASONS}")
print("=" * 60)

for year in SEASONS:
    print(f"\n[{year}] Loading event schedule …")
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as e:
        print(f"  ⚠  Could not load schedule for {year}: {e}")
        continue

    # Filter to race rounds only (round > 0)
    races = schedule[schedule["RoundNumber"] > 0]
    print(f"  Found {len(races)} race rounds")

    for _, event in races.iterrows():
        round_num  = int(event["RoundNumber"])
        event_name = event.get("EventName", f"Round {round_num}")

        print(f"  [{year} R{round_num:02d}] {event_name} … ", end="", flush=True)

        try:
            session = fastf1.get_session(year, round_num, "R")
            session.load()
        except Exception as e:
            print(f"SKIP ({e})")
            skipped.append(f"{year} R{round_num} {event_name}")
            continue
        try:
            laps    = session.laps
            weather = session.weather_data
        except Exception as e:
            print(f"SKIP (data not loaded: {e})")
            skipped.append(f"{year} R{round_num} {event_name}")
            continue
        
        if laps is None or laps.empty:
            print("SKIP (no lap data)")
            skipped.append(f"{year} R{round_num} {event_name}")
            continue

        # ── Weather: use median values from the race session ─────────────
        if weather is not None and not weather.empty:
            track_temp = float(weather["TrackTemp"].median())
            air_temp   = float(weather["AirTemp"].median())
            rainfall   = int(weather["Rainfall"].any())  # 1 if any rain occurred
        else:
            track_temp = 35.0   # fallback defaults
            air_temp   = 25.0
            rainfall   = 0

        # Skip wet races entirely
        if rainfall == 1:
            print(f"SKIP (wet race)")
            skipped.append(f"{year} R{round_num} {event_name} [WET]")
            continue

        # ── Starting compound: lap 1 for each driver ─────────────────────
        lap1 = laps[laps["LapNumber"] == 1].copy()

        if lap1.empty:
            print("SKIP (no lap 1 data)")
            skipped.append(f"{year} R{round_num} {event_name}")
            continue

        # ── Session results for grid positions ───────────────────────────
        results = session.results

        race_records = 0
        for _, lap_row in lap1.iterrows():
            driver     = lap_row.get("Driver", "")
            compound   = str(lap_row.get("Compound", "")).upper()
            team_name  = str(lap_row.get("Team", ""))

            # Only keep dry compounds
            if compound not in COMPOUND_MAP:
                continue

            compound_code = COMPOUND_MAP[compound]
            team_code     = TEAM_MAP.get(team_name, 0)

            if team_code == 0:
                # Try partial match
                for key, val in TEAM_MAP.items():
                    if key.lower() in team_name.lower() or team_name.lower() in key.lower():
                        team_code = val
                        break

            # Grid position from results
            grid_pos = 10  # default midfield
            if results is not None and not results.empty:
                drv_result = results[results["Abbreviation"] == driver]
                if not drv_result.empty:
                    gp = drv_result.iloc[0].get("GridPosition", None)
                    if gp is not None and not pd.isna(gp):
                        grid_pos = int(gp)
                        grid_pos = max(1, min(20, grid_pos))

            records.append({
                "Year":             year,
                "Round":            round_num,
                "EventName":        event_name,
                "Driver":           driver,
                "Team":             team_name,
                "GridPosition":     grid_pos,
                "TrackTemp":        round(track_temp, 1),
                "AirTemp":          round(air_temp, 1),
                "RainProbability":  0,
                "StartingCompound": compound_code,
                "CompoundName":     compound,
            })
            race_records += 1

        print(f"OK  ({race_records} drivers)")
        time.sleep(0.3)   # be polite to the API

# ── Save ──────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)

print("\n" + "=" * 60)
print(f"  Total records collected: {len(df)}")

if not df.empty:
    print(f"\n  Compound distribution:")
    label_map = {0: "Soft", 1: "Medium", 2: "Hard"}
    print(df["StartingCompound"].map(label_map).value_counts().to_string())

    print(f"\n  Season breakdown:")
    print(df.groupby("Year").size().to_string())

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  ✅  Saved to: {OUTPUT_PATH}")
    print(f"\n  Now run:  python main.py --csv {OUTPUT_PATH}")
else:
    print("  ❌  No data collected. Check your internet connection.")

if skipped:
    print(f"\n  Skipped {len(skipped)} sessions:")
    for s in skipped[:10]:
        print(f"    - {s}")
    if len(skipped) > 10:
        print(f"    ... and {len(skipped) - 10} more")

print("=" * 60)
