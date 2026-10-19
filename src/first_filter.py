"""Скрипт для первичной фильтрации структур MOF на основе REFCODE и количества атомов в примитивной ячейке"""
import argparse
import re
from pathlib import Path

import pandas as pd
import spglib
from pymatgen.core import Structure

# Константы путей и параметров
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_DIR / "datasets"
TEMP_DATASET_DIR = DATASETS_DIR / "temp_dataset"
DEFAULT_INPUT_PATH = TEMP_DATASET_DIR / "final_master_index.csv"
DEFAULT_OUTPUT_PATH = TEMP_DATASET_DIR / "first_filtered_master_index.csv"

CSD_REFCODE_PATTERN = re.compile(r"^[A-Z]{6}(?:[0-9]{2})?$")
MAX_PRIMITIVE_ATOMS = 100
SYMMETRY_TOLERANCE = 0.01


def parse_refcode(value):
  if pd.isna(value):
    return None
  candidate = str(value).strip().split("_", 1)[0].upper()
  if CSD_REFCODE_PATTERN.fullmatch(candidate):
    return candidate
  return None


def get_row_refcode(row):
  source = str(row.get("source", "")).strip().lower()
  if source == "core":
    value = row.get("refcode")
    if pd.isna(value) or not str(value).strip():
      return None
    return str(value).strip()

  if source == "usmof":
    return None

  existing = parse_refcode(row.get("refcode"))
  if existing:
    return existing

  name = row.get("name")
  if isinstance(name, str) and name.strip():
    refcode = parse_refcode(name)
    if refcode:
      return refcode

  return parse_refcode(row.get("master_id"))


def standardize_to_primitive(cif_path):
  structure = Structure.from_file(cif_path)
  site_types = {}
  species_by_type = {}
  numbers = []

  for site in structure:
    signature = tuple(
      sorted(
        (
          int(specie.Z),
          round(float(occupancy), 6),
        )
        for specie, occupancy in site.species.items()
      )
    )
    if signature not in site_types:
      site_types[signature] = len(site_types) + 1
    site_type = site_types[signature]
    species_by_type[site_type] = site.species
    numbers.append(site_type)

  primitive_cell = spglib.standardize_cell(
    (
      structure.lattice.matrix,
      structure.frac_coords,
      numbers,
    ),
    to_primitive=True,
    no_idealize=True,
    symprec=SYMMETRY_TOLERANCE,
  )
  
  if primitive_cell is None:
    raise ValueError("spglib не смог найти примитивную ячейку для структуры")
    
  primitive_structure = Structure(
    lattice=primitive_cell[0],
    species=[
      species_by_type[int(site_type)]
      for site_type in primitive_cell[2]
    ],
    coords=primitive_cell[1],
    coords_are_cartesian=False,
  )
  return primitive_structure, len(primitive_cell[2])


# Аргументы командной строки
parser = argparse.ArgumentParser(
  description=(
    "Deduplicate by non-empty source REFCODE and keep structures with at "
    "most 100 atoms in the spglib primitive cell."
  )
)
parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
parser.add_argument(
  "--recover-existing",
  action="store_true",
  help="Rebuild the first-filter CSV from existing CIFs and result logs.",
)
args = parser.parse_args()

standardized_dir = args.output.parent / "standardized_structures"
rejection_path = args.output.with_name("first_filter_rejections.csv")
error_path = args.output.with_name("first_filter_errors.csv")

if not args.input.is_file():
  raise FileNotFoundError(f"Input master index not found: {args.input}")

data = pd.read_csv(args.input, low_memory=False)
required_columns = {"master_id", "source", "cif_path"}
missing_columns = required_columns.difference(data.columns)
if missing_columns:
  raise ValueError(
    f"{args.input} is missing required columns: {sorted(missing_columns)}"
  )


def recover_existing_outputs(frame):
  frame = frame.copy()
  frame["primitive_atom_count"] = pd.NA
  frame["standardization_status"] = pd.NA

  if not standardized_dir.is_dir():
    raise FileNotFoundError(
      f"Standardized CIF directory not found: {standardized_dir}"
    )
  if not rejection_path.is_file() or not error_path.is_file():
    raise FileNotFoundError(
      "Recovery requires both existing logs: "
      f"{rejection_path} and {error_path}"
    )

  rejections = pd.read_csv(rejection_path, low_memory=False)
  errors = pd.read_csv(error_path, low_memory=False)
  log_columns = {"source", "master_id"}
  
  if not log_columns.issubset(rejections.columns):
    raise ValueError(f"{rejection_path} is missing source/master_id columns")
  if not log_columns.issubset(errors.columns):
    raise ValueError(f"{error_path} is missing source/master_id columns")

  rejection_keys = set(
    zip(
      rejections["source"].astype(str),
      rejections["master_id"].astype(str),
    )
  )
  error_keys = set(
    zip(
      errors["source"].astype(str),
      errors["master_id"].astype(str),
    )
  )

  recovered_rows = []
  expected_files = set()
  uncovered_rows = []
  
  for row_index, row in frame.iterrows():
    source = str(row["source"])
    master_id = str(row["master_id"])
    key = (source, master_id)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", master_id)
    standardized_path = standardized_dir / (
      f"{row_index:07d}_{source}_{safe_id}.cif"
    )
    
    if standardized_path.is_file():
      if key in rejection_keys or key in error_keys:
        raise ValueError(
          f"Row {key!r} has both a standardized CIF and a rejection/error log"
        )
      expected_files.add(standardized_path.name)
      recovered = row.to_dict()
      recovered["refcode"] = get_row_refcode(row)
      recovered["primitive_atom_count"] = pd.NA
      recovered["standardization_status"] = "recovered_existing_output"
      recovered["cif_path"] = str(standardized_path)
      recovered_rows.append(recovered)
    elif key not in rejection_keys and key not in error_keys:
      uncovered_rows.append(key)

  existing_files = {path.name for path in standardized_dir.glob("*.cif")}
  unexpected_files = existing_files.difference(expected_files)
  
  if unexpected_files:
    raise ValueError(
      "Recovery found CIF files that do not match the current input index; "
      f"examples: {sorted(unexpected_files)[:5]}"
    )
  if uncovered_rows:
    raise ValueError(
      "Recovery cannot account for all input rows; "
      f"{len(uncovered_rows)} rows have no CIF or logged result. "
      f"Examples: {uncovered_rows[:5]}"
    )

  recovered_data = pd.DataFrame(recovered_rows, columns=frame.columns)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  recovered_data.to_csv(args.output, index=False)
  
  print(f"Восстановлено готовых строк: {len(recovered_data)}")
  print(f"Сбоев, оставшихся в журнале: {len(errors)}")
  print(f"Отклонений, оставшихся в журнале: {len(rejections)}")
  print(f"Результат: {args.output}")
  print(f"Ошибки: {error_path}")
  print(f"Причины отсева: {rejection_path}")


if args.recover_existing:
  recover_existing_outputs(data)
  raise SystemExit(0)

# Обработка
data["refcode"] = data.apply(get_row_refcode, axis=1)
data["primitive_atom_count"] = pd.NA
data["standardization_status"] = pd.NA

rejected_rows = []
error_rows = []
seen_refcodes = {}
accepted_indices = []

print(f"Записей на входе: {len(data)}")
print("Проверка REFCODE, стандартизация ячеек и лимит примитивной ячейки")

for progress_index, (row_index, row) in enumerate(data.iterrows(), start=1):
  refcode = row["refcode"]
  if pd.notna(refcode) and refcode in seen_refcodes:
    rejected = row.to_dict()
    rejected["first_filter_reason"] = "duplicate_refcode"
    rejected["duplicate_of"] = seen_refcodes[refcode]
    rejected_rows.append(rejected)
    continue

  cif_path = row["cif_path"]
  try:
    if not isinstance(cif_path, str) or not cif_path.strip():
      raise ValueError("CIF не найден или путь пустой")
    primitive_structure, primitive_atom_count = standardize_to_primitive(cif_path)
  except Exception as error:
    error_row = row.to_dict()
    error_row["first_filter_error"] = f"{type(error).__name__}: {error}"
    error_rows.append(error_row)
    continue

  data.at[row_index, "primitive_atom_count"] = primitive_atom_count
  data.at[row_index, "standardization_status"] = "ok"
  
  if primitive_atom_count > MAX_PRIMITIVE_ATOMS:
    rejected = data.loc[row_index].to_dict()
    rejected["first_filter_reason"] = "primitive_cell_over_100_atoms"
    rejected["duplicate_of"] = None
    rejected_rows.append(rejected)
    continue

  safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row["master_id"]))
  standardized_path = standardized_dir / (
    f"{row_index:07d}_{row['source']}_{safe_id}.cif"
  )
  
  try:
    standardized_dir.mkdir(parents=True, exist_ok=True)
    primitive_structure.to(
      filename=str(standardized_path),
      fmt="cif",
    )
  except Exception as error:
    error_row = data.loc[row_index].to_dict()
    error_row["first_filter_error"] = (
      f"{type(error).__name__}: failed to write standardized CIF: {error}"
    )
    error_rows.append(error_row)
    continue

  data.at[row_index, "cif_path"] = str(standardized_path)
  if pd.notna(refcode):
    seen_refcodes[refcode] = row["master_id"]
  accepted_indices.append(row_index)

  if progress_index % 500 == 0:
    print(f"Обработано записей: {progress_index}/{len(data)}")

filtered_data = data.loc[accepted_indices].copy()

args.output.parent.mkdir(parents=True, exist_ok=True)

if rejected_rows:
  pd.DataFrame(rejected_rows).to_csv(rejection_path, index=False)
else:
  pd.DataFrame(
    columns=[*data.columns, "first_filter_reason", "duplicate_of"]
  ).to_csv(rejection_path, index=False)

if error_rows:
  pd.DataFrame(error_rows).to_csv(error_path, index=False)
else:
  pd.DataFrame(
    columns=[*data.columns, "first_filter_error"]
  ).to_csv(error_path, index=False)

filtered_data.to_csv(args.output, index=False)

print(f"Оставлено для второго фильтра: {len(filtered_data)}")
print(
  "Удалено дубликатов REFCODE: "
  f"{sum(row['first_filter_reason'] == 'duplicate_refcode' for row in rejected_rows)}"
)
print(
  "Отклонено по размеру примитивной ячейки: "
  f"{sum(row['first_filter_reason'] == 'primitive_cell_over_100_atoms' for row in rejected_rows)}"
)
print(f"Ошибок чтения или стандартизации: {len(error_rows)}")
print(f"Результат: {args.output}")
print(f"Причины отсева: {rejection_path}")
print(f"Ошибки обработки: {error_path}")

if error_rows:
  print(
    "Внимание: список создан по успешно обработанным CIF; "
    f"{len(error_rows)} структур пропущены и указаны в {error_path}"
  )