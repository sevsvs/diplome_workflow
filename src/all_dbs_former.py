""" Скрипт для объединения данных из различных баз данных MOF в один CSV-файл """
import glob
import os
import re
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_DIR / "datasets"
OUTPUT_DIR = DATASETS_DIR / "temp_dataset"
OUTPUT_PATH = OUTPUT_DIR / "final_master_index.csv"
CSD_REFCODE_PATTERN = re.compile(r"^[A-Z]{6}(?:[0-9]{2})?$")


def extract_qmof_refcode(value):
  if pd.isna(value):
    return pd.NA
  candidate = str(value).strip().split("_", 1)[0].upper()
  if CSD_REFCODE_PATTERN.fullmatch(candidate):
    return candidate
  return pd.NA


def read_selected_columns(path, columns, optional_columns=()):
  required = set(columns)
  wanted = required | set(optional_columns)
  frame = pd.read_csv(
    path,
    low_memory=False,
    usecols=lambda column: column.strip() in wanted,
  )
  frame.columns = frame.columns.str.strip()
  missing = required.difference(frame.columns)
  if missing:
    raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
  for column in optional_columns:
    if column not in frame.columns:
      frame[column] = pd.NA
  return frame


phonon_dir = DATASETS_DIR / "Phonone_db" / "Thermal"
phonon_qmof_ids = set()
for path in glob.glob(str(phonon_dir / "*.yml")):
  parts = Path(path).stem.split("-", 2)
  if len(parts) >= 2:
    phonon_qmof_ids.add(f"{parts[0]}-{parts[1]}")

qmof_path = DATASETS_DIR / "qmof_db" / "qmof.csv"
qmof_columns = ["qmof_id", "name", "info.doi", "info.mofid.mofid"]
qmof_df = read_selected_columns(qmof_path, qmof_columns)
qmof_ids = qmof_df["qmof_id"]
qmof_clean = pd.DataFrame(
  {
    "master_id": qmof_ids,
    "source": "qmof",
    "name": qmof_df["name"],
    "refcode": qmof_df["name"].map(extract_qmof_refcode),
    "core_unmodified": pd.NA,
    "doi": qmof_df["info.doi"],
    "mofid": qmof_df["info.mofid.mofid"],
    "cif_path": qmof_ids.map(
      lambda qmof_id: str(
        DATASETS_DIR
        / "qmof_db"
        / "relaxed_structures"
        / f"{qmof_id}.cif"
      )
    ),
    "has_phonon": qmof_ids.isin(phonon_qmof_ids),
  }
)

core_dir = DATASETS_DIR / "CoREMOF_db"
core_frames = []
for dataset_type, cif_subdir in (("ASR", "ASR"), ("FSR", "FSR"), ("ION", "Ion")):
  csv_path = core_dir / f"{dataset_type}_data_SI_20250204.csv"
  if not csv_path.is_file():
    continue

  core_part = read_selected_columns(
    csv_path,
    ["coreid", "refcode"],
    ["DOI", "mofid-v2", "name", "unmodified"],
  )
  core_part = core_part.rename(
    columns={
      "DOI": "doi",
      "mofid-v2": "mofid",
      "unmodified": "core_unmodified",
    }
  )
  core_part["source"] = "core"
  core_part["dataset_type"] = dataset_type
  core_part["cif_subdir"] = cif_subdir
  core_frames.append(core_part)

core_columns = [
  "master_id",
  "source",
  "name",
  "refcode",
  "core_unmodified",
  "doi",
  "mofid",
  "cif_path",
  "has_phonon",
]
if core_frames:
  core_df = pd.concat(core_frames, ignore_index=True)
  core_clean = pd.DataFrame(
    {
      "master_id": core_df["coreid"],
      "source": "core",
      "name": core_df["name"],
      "refcode": core_df["refcode"],
      "core_unmodified": core_df["core_unmodified"],
      "doi": core_df["doi"],
      "mofid": core_df["mofid"],
      "cif_path": [
        str(core_dir / "SI" / "CR" / subdir / f"{coreid}.cif")
        for subdir, coreid in zip(
          core_df["cif_subdir"],
          core_df["coreid"],
        )
      ],
      "has_phonon": False,
    }
  )
else:
  core_clean = pd.DataFrame(columns=core_columns)

usmof_folder = (
  DATASETS_DIR
  / "USMOF_db"
  / "features_and_properties"
  / "computed_properties"
  / "stable"
)
usmof_structures_dir = DATASETS_DIR / "USMOF_db" / "optimized_structures"
usmof_inputs = []
required_usmof_stems = set()
for csv_path in glob.glob(str(usmof_folder / "*.csv")):
  header = pd.read_csv(csv_path, nrows=0)
  header.columns = header.columns.str.strip()
  cif_column = next(
    (
      column
      for column in ("cif_file", "filename", "name")
      if column in header.columns
    ),
    None,
  )
  if cif_column is None:
    continue

  filename_df = read_selected_columns(csv_path, [cif_column])
  stems = filename_df[cif_column].map(
    lambda value: Path(str(value)).stem if pd.notna(value) else None
  )
  required_usmof_stems.update(
    str(stem).lower() for stem in stems.dropna()
  )
  usmof_inputs.append(stems)

paths_by_stem = {}
paths_by_alias = {}
if required_usmof_stems:
  for cif_path in usmof_structures_dir.rglob("*.cif"):
    normalized_stem = cif_path.stem.lower()
    if normalized_stem in required_usmof_stems:
      paths_by_stem.setdefault(normalized_stem, cif_path)
    if normalized_stem.startswith("optimized_"):
      alias = normalized_stem[len("optimized_"):]
      if alias in required_usmof_stems:
        paths_by_alias.setdefault(alias, cif_path)

usmof_frames = []
for stems in usmof_inputs:
  normalized_stems = stems.str.lower()
  extracted_paths = normalized_stems.map(paths_by_stem)
  missing_paths = extracted_paths.isna()
  if missing_paths.any():
    extracted_paths.loc[missing_paths] = normalized_stems.loc[
      missing_paths
    ].map(paths_by_alias)

  master_ids = extracted_paths.map(
    lambda path: path.stem if pd.notna(path) else None
  ).fillna(stems)
  cif_paths = extracted_paths.map(
    lambda path: str(path) if pd.notna(path) else pd.NA
  )
  usmof_frames.append(
    pd.DataFrame(
      {
        "master_id": master_ids,
        "source": "usmof",
        "name": pd.NA,
        "refcode": pd.NA,
        "core_unmodified": pd.NA,
        "doi": pd.NA,
        "mofid": pd.NA,
        "cif_path": cif_paths,
        "has_phonon": False,
      }
    )
  )

if usmof_frames:
  usmof_clean = pd.concat(usmof_frames, ignore_index=True)
  usmof_clean = usmof_clean.drop_duplicates(
    subset=["master_id"], keep="first"
  )
else:
  usmof_clean = pd.DataFrame(columns=core_columns)

master_df = pd.concat(
  [core_clean, qmof_clean, usmof_clean],
  ignore_index=True,
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
master_df.to_csv(OUTPUT_PATH, index=False)

missing_cifs = master_df["cif_path"].isna()
print(f"Сборка завершена. Строк: {len(master_df)}, файл: {OUTPUT_PATH}")
print(f"По источникам:{master_df['source'].value_counts().to_string()}")
print(f"Пустых путей к CIF: {int(missing_cifs.sum())}")
print(f"Записей с REFCODE: {int(master_df['refcode'].notna().sum())}")
