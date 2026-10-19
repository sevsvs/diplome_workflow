"""Срипт для дедупликации структур по их CIF"""
import argparse
import os
from pathlib import Path

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

if not hasattr(np, "int"):
  setattr(np, "int", np.int32 if os.name == "nt" else np.int64) # исправление ошибки совместимости, где использовался np.int

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure


PROJECT_DIR = Path(__file__).resolve().parent.parent
TEMP_DATASET_DIR = PROJECT_DIR / "datasets" / "temp_dataset"
DEFAULT_INPUT_PATH = TEMP_DATASET_DIR / "first_filtered_master_index.csv"
DEFAULT_OUTPUT_PATH = TEMP_DATASET_DIR / "second_filtered_master_index.csv"
SOURCE_PRIORITY = {"core": 0, "qmof": 1, "usmof": 2}
VOLUME_TOLERANCE = 0.05


parser = argparse.ArgumentParser(
  description=(
    "Remove structural duplicates from first_filter output using "
    "pymatgen StructureMatcher."
  )
)
parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)

if not args.input.is_file():
  raise FileNotFoundError(f"Input first-filter index not found: {args.input}")

data = pd.read_csv(args.input, low_memory=False)
required_columns = {"master_id", "source", "cif_path"}
missing_columns = required_columns.difference(data.columns)
if missing_columns:
  raise ValueError(
    f"{args.input} is missing required columns: {sorted(missing_columns)}"
  )

upstream_errors_path = args.input.with_name("first_filter_errors.csv")
if upstream_errors_path.is_file():
  upstream_error_count = len(
    pd.read_csv(upstream_errors_path, usecols=["master_id"])
  )
  if upstream_error_count:
    print(
      f"Внимание: first_filter не обработал {upstream_error_count} структур; "
      f"см. {upstream_errors_path}"
    )

data = data.copy()
data["_source_priority"] = (
  data["source"].astype(str).str.lower().map(SOURCE_PRIORITY).fillna(99)
)
data = data.sort_values(
  ["_source_priority"],
  kind="stable",
).reset_index(drop=False).rename(columns={"index": "_input_index"})

structures = []
load_errors = []
print(f"Структур для сравнения: {len(data)}")
print("Загрузка стандартизированных CIF")
for _, row in data.iterrows():
  cif_path = row["cif_path"]
  try:
    if not isinstance(cif_path, str) or not cif_path.strip():
      raise ValueError("Путь к CIF отсутствуеь")
    structure = Structure.from_file(cif_path)
    if len(structure) > 100:
      raise ValueError(
        f"Стандартизованный CIF имеет {len(structure)} атомов, что больше чем 100"
      )
    structures.append(structure)
  except Exception as error:
    load_errors.append(
      {
        "master_id": row["master_id"],
        "source": row["source"],
        "cif_path": cif_path,
        "second_filter_error": f"{type(error).__name__}: {error}",
      }
    )

errors_path = args.output.with_name("second_filter_errors.csv")
if load_errors:
  pd.DataFrame(load_errors).to_csv(errors_path, index=False)
  raise RuntimeError(
    f"{len(load_errors)} стандартизованных структур не удалось прочитать "
    f"Структура с ошибкой: {errors_path}"
  )

volumes = np.asarray([structure.volume for structure in structures], dtype=float)
sorted_indices = np.argsort(volumes, kind="stable")
sorted_volumes = volumes[sorted_indices]
matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0)
kept_indices = []
kept_index_set = set()
duplicate_rows = []
match_errors = []

print("Поиск структур-дубликатов")
for index, structure in enumerate(structures):
  lower_volume = np.nextafter(
    volumes[index] * (1.0 - VOLUME_TOLERANCE), -np.inf
  )
  upper_volume = np.nextafter(
    volumes[index] / (1.0 - VOLUME_TOLERANCE), np.inf
  )
  start = np.searchsorted(sorted_volumes, lower_volume, side="left")
  stop = np.searchsorted(sorted_volumes, upper_volume, side="right")
  candidates = sorted_indices[start:stop]
  candidates = candidates[
    (candidates < index)
    & np.fromiter(
      (int(candidate) in kept_index_set for candidate in candidates),
      dtype=bool,
      count=len(candidates),
    )
  ]

  duplicate_of = None
  for previous_index in candidates:
    previous_volume = volumes[previous_index]
    if (
      abs(volumes[index] - previous_volume)
      / max(volumes[index], previous_volume)
      > VOLUME_TOLERANCE
    ):
      continue
    try:
      if matcher.fit(structure, structures[previous_index]):
        duplicate_of = previous_index
        break
    except Exception as error:
      row = data.iloc[index]
      prior_row = data.iloc[previous_index]
      match_errors.append(
        {
          "master_id": row["master_id"],
          "source": row["source"],
          "compared_master_id": prior_row["master_id"],
          "compared_source": prior_row["source"],
          "second_filter_error": f"{type(error).__name__}: {error}",
        }
      )

  if duplicate_of is None:
    kept_indices.append(index)
    kept_index_set.add(index)
  else:
    row = data.iloc[index].to_dict()
    prior_row = data.iloc[duplicate_of]
    row["duplicate_of"] = prior_row["master_id"]
    row["duplicate_of_source"] = prior_row["source"]
    duplicate_rows.append(row)

  if (index + 1) % 500 == 0:
    print(f"Сравнено структур: {index + 1}/{len(data)}")

if match_errors:
  pd.DataFrame(match_errors).to_csv(errors_path, index=False)
  raise RuntimeError(
    f"{len(match_errors)} StructureMatcher не смог сравнить структуры "
    f"Структура с ошибкой: {errors_path}"
  )
elif errors_path.exists():
  errors_path.unlink()

kept_input_indices = data.iloc[kept_indices]["_input_index"].to_list()
result = (
  pd.read_csv(args.input, low_memory=False)
  .iloc[kept_input_indices]
  .copy()
)
args.output.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(args.output, index=False)

duplicates_path = args.output.with_name("second_filter_duplicates.csv")
if duplicate_rows:
  pd.DataFrame(duplicate_rows).drop(
    columns=["_source_priority", "_input_index"],
    errors="ignore",
  ).to_csv(duplicates_path, index=False)
else:
  pd.DataFrame(
    columns=[
      *result.columns,
      "duplicate_of",
      "duplicate_of_source",
    ]
  ).to_csv(duplicates_path, index=False)

print(f"Оставлено уникальных структур: {len(result)}")
print(f"Удалено структурных дубликатов: {len(duplicate_rows)}")
print(f"Результат: {args.output}")
print(f"Журнал дубликатов: {duplicates_path}")
