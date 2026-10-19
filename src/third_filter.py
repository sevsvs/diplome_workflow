""" Третий фильтр для проверки качества структур MOFChecker и проверки атомных зарядов с помощью EqEq """
import argparse
import os
import tempfile
from pathlib import Path

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from mofchecker import MOFChecker
from pymatgen.core import Structure
from pyeqeq.main import run_on_cif


PROJECT_DIR = Path(__file__).resolve().parent.parent
TEMP_DATASET_DIR = PROJECT_DIR / "datasets" / "temp_dataset"
DEFAULT_INPUT_PATH = TEMP_DATASET_DIR / "second_filtered_master_index.csv"
DEFAULT_OUTPUT_PATH = TEMP_DATASET_DIR / "third_filtered_master_index.csv"
CHARGE_THRESHOLD = 3.0
PROGRESS_INTERVAL = 100


def check_eqeq_charges(structure):
  "Проверка атомных зарядов, рассчитанных с помощью EqEq, на наличие высоких зарядов"
  with tempfile.TemporaryDirectory(prefix="mof-eqeq-") as temp_dir:
    cif_path = Path(temp_dir) / "structure.cif"
    structure.to(filename=str(cif_path), fmt="cif")
    charges = np.asarray(run_on_cif(str(cif_path), verbose=False), dtype=float)
  if not np.all(np.isfinite(charges)):
    raise ValueError("EqEq returned non-finite atomic charges")
  return bool(np.any(np.abs(charges) > CHARGE_THRESHOLD))


parser = argparse.ArgumentParser(
  description=(
    "Run MOFChecker quality checks on the structurally deduplicated index."
  )
)
parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
args = parser.parse_args()

if not args.input.is_file():
  raise FileNotFoundError(
    f"Не найден файл после второго фильтра: {args.input}"
  )

data = pd.read_csv(args.input, low_memory=False)
required_columns = {"master_id", "source", "cif_path"}
missing_columns = required_columns.difference(data.columns)
if missing_columns:
  raise ValueError(
    f"{args.input} не имеет следующих колонок: {sorted(missing_columns)}"
  )

check_columns = [
  "partial_occupancy",
  "high_eqeq_charges",
  "undercoordinated_carbon",
  "undercoordinated_nitrogen",
  "floating_molecule",
  "atomic_overlaps",
  "third_filter_reasons",
  "third_filter_error",
]

for column in check_columns:
  data[column] = pd.NA

clean_indices = []
rejection_rows = []
error_rows = []
print(f"Структур для проверки MOFChecker: {len(data)}")

for row_index, row in data.iterrows():
  try:
    cif_path = row["cif_path"]
    if not isinstance(cif_path, str) or not cif_path.strip():
      raise ValueError("Путь к стандартизированному CIF отсутствует")
    structure = Structure.from_file(cif_path)

    partial_occupancy = any(
      not site.is_ordered
      or abs(sum(site.species.values()) - 1.0) > 1e-6
      for site in structure
    )
    data.at[row_index, "partial_occupancy"] = partial_occupancy
    if partial_occupancy:
      data.at[row_index, "third_filter_reasons"] = "partial_occupancy"
      rejection_rows.append(data.loc[row_index].to_dict())
      continue

    checker = MOFChecker(structure, primitive=False)
    high_charges = check_eqeq_charges(checker.structure)
    undercoordinated_carbon = checker.has_undercoordinated_c
    undercoordinated_nitrogen = checker.has_undercoordinated_n
    floating_molecule = checker.has_lone_molecule
    atomic_overlaps = checker.has_atomic_overlaps

    data.at[row_index, "high_eqeq_charges"] = high_charges
    data.at[row_index, "undercoordinated_carbon"] = (
      undercoordinated_carbon
    )
    data.at[row_index, "undercoordinated_nitrogen"] = (
      undercoordinated_nitrogen
    )
    data.at[row_index, "floating_molecule"] = floating_molecule
    data.at[row_index, "atomic_overlaps"] = atomic_overlaps

    reasons = []
    if high_charges:
      reasons.append("high_eqeq_charges")
    if undercoordinated_carbon or undercoordinated_nitrogen:
      reasons.append("undercoordinated_C_or_N_possible_missing_H")
    if floating_molecule:
      reasons.append("floating_molecule_or_residual_solvent")
    if atomic_overlaps:
      reasons.append("atomic_overlaps")

    if reasons:
      data.at[row_index, "third_filter_reasons"] = ";".join(reasons)
      rejection_rows.append(data.loc[row_index].to_dict())
    else:
      data.at[row_index, "third_filter_reasons"] = ""
      clean_indices.append(row_index)
  except Exception as error:
    error_message = f"{type(error).__name__}: {error}"
    data.at[row_index, "third_filter_error"] = error_message
    data.at[row_index, "third_filter_reasons"] = "check_error"
    rejection_rows.append(data.loc[row_index].to_dict())
    error_rows.append(
      {
        "master_id": row["master_id"],
        "source": row["source"],
        "cif_path": row["cif_path"],
        "third_filter_error": error_message,
      }
    )

  if (row_index + 1) % PROGRESS_INTERVAL == 0:
    print(f"Проверено структур: {row_index + 1}/{len(data)}")

args.output.parent.mkdir(parents=True, exist_ok=True)
data.loc[clean_indices].to_csv(args.output, index=False)

rejection_path = args.output.with_name("third_filter_rejections.csv")
error_path = args.output.with_name("third_filter_errors.csv")
if rejection_rows:
  pd.DataFrame(rejection_rows).to_csv(rejection_path, index=False)
else:
  pd.DataFrame(columns=[*data.columns]).to_csv(rejection_path, index=False)

if error_rows:
  pd.DataFrame(error_rows).to_csv(error_path, index=False)
else:
  pd.DataFrame(
    columns=[
      "master_id",
      "source",
      "cif_path",
      "third_filter_error",
    ]
  ).to_csv(error_path, index=False)

print(f"Чистых структур: {len(clean_indices)}")
print(f"Отклонено/ошибок: {len(rejection_rows)}")
print(f"Чистый CSV: {args.output}")
print(f"Журнал причин: {rejection_path}")
print(f"Журнал ошибок: {error_path}")
