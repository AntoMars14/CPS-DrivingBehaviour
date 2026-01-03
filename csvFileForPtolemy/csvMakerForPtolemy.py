import os
import numpy as np
import pandas as pd

INPUT_PATH = os.path.join("..\dataset", "test_motion_data.csv")
OUTPUT_PATH = os.path.join("csvForPtolemyWithAccidents.csv")

# Synthetic row insertion params
START_INDEX = 2500     # 0-based index: insert the first synthetic row before this row
STEP = 100             # re-insert every N rows
ACC_MAG_MIN = 20.0     # minimum acceleration magnitude (strictly > 20)

# Gyro range (tweak as you like)
GYRO_RANGE = (-5.0, 5.0)

# Optional seed for reproducibility
RNG = np.random.default_rng(seed=42)


def make_accident_row():
    """
    Build a synthetic row with Class=ACCIDENT and acceleration magnitude > ACC_MAG_MIN.
    Gyro values are random within the configured range.
    """
    # Random direction (unit vector)
    v = RNG.normal(size=3)
    norm = np.linalg.norm(v)
    if norm == 0:
        v = np.array([1.0, 0.0, 0.0])
        norm = 1.0
    v = v / norm

    # Random magnitude but guaranteed > 20 (add some headroom)
    mag = RNG.uniform(ACC_MAG_MIN + 0.5, ACC_MAG_MIN + 15.0)
    acc = v * mag

    gyro = RNG.uniform(GYRO_RANGE[0], GYRO_RANGE[1], size=3)

    return {
        "AccX": float(acc[0]),
        "AccY": float(acc[1]),
        "AccZ": float(acc[2]),
        "GyroX": float(gyro[0]),
        "GyroY": float(gyro[1]),
        "GyroZ": float(gyro[2]),
        "Class": "ACCIDENT",
    }


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"File not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    # Drop Timestamp if present
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    # Basic schema check
    required = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ", "Class"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Present columns: {list(df.columns)}")

    n = len(df)
    if n <= START_INDEX:
        print(f"File has only {n} rows: no insertion performed (START_INDEX={START_INDEX}).")
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved: {OUTPUT_PATH}")
        return

    # Build insertion indices: 2500, 2600, 2700, ...
    insert_positions = list(range(START_INDEX, n + 1, STEP))

    # Rebuild dataframe with inserted synthetic rows
    chunks = []
    prev = 0
    for pos in insert_positions:
        # append original rows up to pos
        chunks.append(df.iloc[prev:pos])
        # insert one synthetic row
        accident_row = make_accident_row()
        chunks.append(pd.DataFrame([accident_row], columns=df.columns))
        prev = pos

    # append remaining rows
    chunks.append(df.iloc[prev:])

    out = pd.concat(chunks, ignore_index=True)

    # (Optional) validate accident magnitude constraint
    acc_cols = out[["AccX", "AccY", "AccZ"]].to_numpy()
    mags = np.linalg.norm(acc_cols, axis=1)
    # only check ACCIDENT rows
    accident_mags = mags[out["Class"].astype(str).str.upper() == "ACCIDENT"]
    if len(accident_mags) > 0 and not np.all(accident_mags > ACC_MAG_MIN):
        raise RuntimeError("Error: some ACCIDENT rows do not satisfy magnitude > 20.")

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"OK. Input:  {INPUT_PATH}")
    print(f"OK. Output: {OUTPUT_PATH}")
    print(f"Original rows: {n}, final rows: {len(out)}, inserted ACCIDENT rows: {len(insert_positions)}")


if __name__ == "__main__":
    main()