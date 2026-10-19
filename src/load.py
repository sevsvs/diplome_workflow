"""Скрипт для загрузки структур из третьего фильтра в финальный набор данных"""
import argparse
import re
from pathlib import Path

import pandas as pd
import spglib
from pymatgen.core import Element, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_DIR / "datasets"
TEMP_DATASET_DIR = DATASETS_DIR / "temp_dataset"
DEFAULT_INPUT_PATH = TEMP_DATASET_DIR / "third_filtered_master_index.csv"
DEFAULT_OUTPUT_DIR = DATASETS_DIR / "final_dataset"
DEFAULT_TABLE_PATH = DEFAULT_OUTPUT_DIR / "structures.csv"
SYMMETRY_TOLERANCE = 0.01


def make_primitive_structure(structure):
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

  cell = spglib.standardize_cell(
    (
      structure.lattice.matrix,
      structure.frac_coords,
      numbers,
    ),
    to_primitive=True,
    no_idealize=True,
    symprec=SYMMETRY_TOLERANCE,
  )
  if cell is None:
    raise ValueError("spglib could not find a primitive cell")

  return Structure(
    lattice=cell[0],
    species=[species_by_type[int(site_type)] for site_type in cell[2]],
    coords=cell[1],
    coords_are_cartesian=False,
  )


def get_metal_symbols(structure):
  metals = {
    specie.symbol
    for site in structure
    for specie in site.species
    if Element(specie.symbol).is_metal
  }
  return ", ".join(sorted(metals, key=lambda symbol: Element(symbol).Z))


parser = argparse.ArgumentParser(
  description=(
    "Copy third-filter structures into final_dataset and export primitive-cell "
    "structure metadata."
  )
)
parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
args = parser.parse_args()

if not args.input.is_file():
  raise FileNotFoundError(f"Third-filter CSV not found: {args.input}")

data = pd.read_csv(args.input, low_memory=False)
required_columns = {"master_id", "source", "cif_path"}
missing_columns = required_columns.difference(data.columns)
if missing_columns:
  raise ValueError(
    f"{args.input} is missing required columns: {sorted(missing_columns)}"
  )

structures_dir = args.output_dir / "structures"
table_path = args.output_dir / "structures.csv"
errors_path = args.output_dir / "load_errors.csv"
structures_dir.mkdir(parents=True, exist_ok=True)

table_rows = []
error_rows = []
print(f"Структур для загрузки: {len(data)}")

for row_index, row in data.iterrows():
  source_path = row["cif_path"]
  try:
    if not isinstance(source_path, str) or not source_path.strip():
      raise ValueError("Missing CIF path")

    structure = Structure.from_file(source_path)
    if any(not site.is_ordered for site in structure):
      raise ValueError("Structure contains partial occupancies")

    primitive_structure = make_primitive_structure(structure)
    analyzer = SpacegroupAnalyzer(
      primitive_structure,
      symprec=SYMMETRY_TOLERANCE,
    )
    space_group = analyzer.get_space_group_symbol()
    crystal_system = analyzer.get_crystal_system()
    reduced_formula = primitive_structure.composition.reduced_formula
    atom_count = len(primitive_structure)

    safe_id = re.sub(
      r"[^A-Za-z0-9_.-]+",
      "_",
      str(row["master_id"]),
    ).strip("._")
    if not safe_id:
      raise ValueError("master_id cannot be converted to a safe filename")
    output_name = f"{row_index:07d}_{safe_id}.cif"
    output_path = structures_dir / output_name
    primitive_structure.to(filename=str(output_path), fmt="cif")

    table_rows.append(
      {
        "id": row["master_id"],
        "formula": reduced_formula,
        "metal": get_metal_symbols(primitive_structure),
        "space_group": space_group,
        "crystal_system": crystal_system,
        "n_atoms": atom_count,
        "calculation_cell": "primitive",
        "source": row["source"],
        "cif_path": str(output_path),
      }
    )
  except Exception as error:
    error_rows.append(
      {
        "id": row["master_id"],
        "source": row["source"],
        "input_cif_path": source_path,
        "error": f"{type(error).__name__}: {error}",
      }
    )

  if (row_index + 1) % 100 == 0:
    print(f"Обработано структур: {row_index + 1}/{len(data)}")

args.output_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
  table_rows,
  columns=[
    "id",
    "formula",
    "metal",
    "space_group",
    "crystal_system",
    "n_atoms",
    "calculation_cell",
    "source",
    "cif_path",
  ],
).to_csv(table_path, index=False)
pd.DataFrame(
  error_rows,
  columns=["id", "source", "input_cif_path", "error"],
).to_csv(errors_path, index=False)

print(f"Сохранено структур: {len(table_rows)}")
print(f"Ошибок чтения/обработки: {len(error_rows)}")
print(f"Таблица: {table_path}")
print(f"CIF-папка: {structures_dir}")
print(f"Журнал ошибок: {errors_path}")
