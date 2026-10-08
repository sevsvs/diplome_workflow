from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
PARENT_DIR = os.path.dirname(BASE_DIR)

all_materials_df = pd.DataFrame()

qmof_db_place = os.path.join(PARENT_DIR, "qmof_db", "qmof.csv")
qmof_df = pd.read_csv(qmof_db_place)

core_mof_ASR_place = os.path.join(PARENT_DIR, "CoREMOF_db", "ASR_data_SI_20250204.csv")
core_mof_FSR_place = os.path.join(PARENT_DIR, "CoREMOF_db", "FSR_data_SI_20250204.csv")
core_mof_ION_place = os.path.join(PARENT_DIR, "CoREMOF_db", "ION_data_SI_20250204.csv")

core_mof_ASR = pd.read_csv(core_mof_ASR_place)
core_mof_FSR = pd.read_csv(core_mof_FSR_place)
core_mof_ION = pd.read_csv(core_mof_ION_place)

core_mof_ASR["dataset_type"] = "ASR"
core_mof_FSR["dataset_type"] = "FSR"
core_mof_ION["dataset_type"] = "ION"

core_mof_df = pd.concat([core_mof_ASR, core_mof_FSR, core_mof_ION], ignore_index=True)


def first_filter():
  pass

def second_filter():
  pass