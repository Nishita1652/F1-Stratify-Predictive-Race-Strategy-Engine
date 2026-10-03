"""
data_loader.py
--------------
Loads, generates, and preprocesses F1 race strategy data.
If no real CSV is provided, a synthetic dataset is generated
that mirrors real-world F1 strategy patterns.
"""

import os
import pandas as pd
import numpy as np


# ── Team encoding map (used everywhere) ─────────────────────────────────────
TEAM_ENCODING = {
    "Red Bull":     1,
    "Ferrari":      2,
    "Mercedes":     3,
    "McLaren":      4,
    "Aston Martin": 5,
    "Alpine":       6,
    "Williams":     7,
    "AlphaTauri":   8,
    "Alfa Romeo":   9,
    "Haas":        10,
}

# ── Label map ────────────────────────────────────────────────────────────────
COMPOUND_LABELS = {0: "Soft", 1: "Medium", 2: "Hard"}
COMPOUND_COLORS = {"Soft": "#E8002D", "Medium": "#FFF200", "Hard": "#FFFFFF"}


def generate_synthetic_data(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic F1 starting-tyre dataset.

    Strategy logic encoded:
      - Hot track (>45°C)    → prefers Medium / Hard
      - Cold track (<30°C)   → prefers Soft
      - Rain probability = 1 → strategy forced to Wet (excluded from this dataset)
      - Back-of-grid (>15)   → risk strategy: more Soft starts
      - Top teams (1-3)      → slightly more Medium conservative starts
    """
    rng = np.random.default_rng(random_state)

    grid_pos       = rng.integers(1, 21, n_samples)
    track_temp     = rng.uniform(20.0, 60.0, n_samples).round(1)
    air_temp       = (track_temp - rng.uniform(5.0, 15.0, n_samples)).round(1)
    rain_prob      = rng.choice([0, 1], p=[0.85, 0.15], size=n_samples)
    team_code      = rng.integers(1, 11, n_samples)

    # Rule-based compound assignment with noise
    compounds = []
    for i in range(n_samples):
        if rain_prob[i] == 1:
            # Wet races → skip (filter later)
            compounds.append(-1)
            continue

        score_soft   = 0.0
        score_medium = 0.0
        score_hard   = 0.0

        # Temperature influence
        if track_temp[i] < 30:
            score_soft += 2.0
        elif track_temp[i] < 40:
            score_soft += 1.0
            score_medium += 1.5
        elif track_temp[i] < 50:
            score_medium += 2.0
            score_hard += 1.0
        else:
            score_medium += 1.0
            score_hard += 2.5

        # Grid position influence
        if grid_pos[i] <= 3:           # Front row → conservative
            score_medium += 1.5
        elif grid_pos[i] <= 10:        # Midfield
            score_soft += 1.0
            score_medium += 0.5
        else:                           # Back → risk / undercut
            score_soft += 2.0

        # Team tendency
        if team_code[i] in (1, 2, 3):  # Top teams → conservative
            score_medium += 0.8
        elif team_code[i] in (7, 8, 9, 10):  # Small teams → gamble soft
            score_soft += 0.5

        # Add noise
        noise = rng.normal(0, 0.5, 3)
        scores = np.array([score_soft, score_medium, score_hard]) + noise
        compounds.append(int(np.argmax(scores)))

    df = pd.DataFrame({
        "GridPosition":   grid_pos,
        "TrackTemp":      track_temp,
        "AirTemp":        air_temp,
        "RainProbability": rain_prob,
        "Team":           team_code,
        "StartingCompound": compounds,
    })

    # Remove wet-race rows (compound = -1)
    df = df[df["StartingCompound"] != -1].reset_index(drop=True)
    return df


def load_or_generate(csv_path: str = None) -> pd.DataFrame:
    """
    Load data from a CSV if provided, otherwise check default real dataset
    location ('data/raw/f1_real_data.csv'), or generate synthetic data.

    Expected CSV columns:
        GridPosition, TrackTemp, AirTemp, RainProbability, Team, StartingCompound
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"[DataLoader] Loaded {len(df)} rows from '{csv_path}'")
    elif os.path.exists("data/raw/f1_real_data.csv"):
        print("[DataLoader] Auto-detected real dataset at 'data/raw/f1_real_data.csv' …")
        df = pd.read_csv("data/raw/f1_real_data.csv")
        print(f"[DataLoader] Loaded {len(df)} rows")
    else:
        print("[DataLoader] No CSV found — generating synthetic dataset …")
        df = generate_synthetic_data()
        print(f"[DataLoader] Generated {len(df)} synthetic samples")

    df = _preprocess(df)
    return df



def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, validate, and normalise the dataframe."""
    required = ["GridPosition", "TrackTemp", "AirTemp",
                "RainProbability", "Team", "StartingCompound"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Drop wet-race rows
    df = df[df["RainProbability"] == 0].copy()

    # Drop nulls
    df.dropna(inplace=True)

    # Clip to valid ranges
    df["GridPosition"]   = df["GridPosition"].clip(1, 20)
    df["TrackTemp"]      = df["TrackTemp"].clip(15, 75)
    df["AirTemp"]        = df["AirTemp"].clip(10, 65)
    # If Team column contains string names, encode them to numbers
    if df["Team"].dtype == object:
        team_encoding = {
        "red bull racing": 1, "red bull": 1, "oracle red bull racing": 1,
        "ferrari": 2, "scuderia ferrari": 2,
        "mercedes": 3, "mercedes amg petronas": 3, "mercedes-amg petronas": 3,
        "mclaren": 4, "mclaren f1 team": 4,
        "aston martin": 5, "aston martin f1 team": 5, "aston martin aramco": 5,
        "racing point": 5, "bwt racing point": 5, "force india": 5,
        "alpine": 6, "alpine f1 team": 6, "renault": 6,
        "williams": 7, "williams racing": 7,
        "alphatauri": 8, "scuderia alphatauri": 8, "toro rosso": 8,
        "rb": 8, "visa cash app rb": 8,
        "alfa romeo": 9, "alfa romeo racing": 9, "sauber": 9, "stake f1 team": 9,
        "haas": 10, "haas f1 team": 10,
        }
        df["Team"] = df["Team"].str.strip().str.lower().map(team_encoding)
        df.dropna(subset=["Team"], inplace=True)  # drop any unrecognised teams
        df["Team"] = df["Team"].astype(int)
    df["Team"] = df["Team"].clip(1, 10)
    df["StartingCompound"] = df["StartingCompound"].astype(int)

    # Domain Feature Engineering:
    # 1. TempDelta: Asphalt solar heat load (TrackTemp - AirTemp)
    # 2. IsTop10: Structural Q2 Qualifying Tyre Regulation threshold (P1-10 vs P11-20)
    df["TempDelta"] = (df["TrackTemp"] - df["AirTemp"]).round(1)
    df["IsTop10"]   = (df["GridPosition"] <= 10).astype(int)

    print(f"[DataLoader] After preprocessing: {len(df)} rows")
    print(f"[DataLoader] Class distribution:\n"
          f"{df['StartingCompound'].map(COMPOUND_LABELS).value_counts().to_string()}\n")
    return df.reset_index(drop=True)


def get_feature_names() -> list[str]:
    return ["GridPosition", "TrackTemp", "AirTemp", "TempDelta", "IsTop10", "Team"]

