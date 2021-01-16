
# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com

"""

import sys
sys.path.insert(0,'../../..') # including the path to deep-tasks folder
from constants import RAUG_PATH
sys.path.insert(0,RAUG_PATH)
import os
from raug.utils.loader import parse_metadata
from raug.utils.loader import split_k_folder_csv, label_categorical_to_number

ISIC_BASE_PATH = "/home/patcha/Datasets/ISIC2019/"
ISIC_BASE_PATH = "/home/patcha/Datasets/ISIC2019/test"

data = parse_metadata (os.path.join(ISIC_BASE_PATH, "ISIC2019.csv"), replace_nan="missing",
           cols_to_parse=['sex', 'anatom_site_general'], replace_rules={"age_approx": {"missing": 0}})


data = split_k_folder_csv(data, "diagnostic", save_path=None, k_folder=5, seed_number=32)
data_train = label_categorical_to_number (data, "diagnostic", col_target_number="diagnostic_number")
data_train.to_csv(os.path.join(ISIC_BASE_PATH, "ISIC2019_parsed_train_15_folders.csv"), index=False)

data_final_test = parse_metadata (os.path.join(ISIC_BASE_PATH, "ISIC_2019_Test_Metadata.csv"), replace_nan="missing",
           cols_to_parse=['sex', 'anatom_site_general'], replace_rules={"age_approx": {"missing": 0}})



















