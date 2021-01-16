# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com

Script to prepare data to train, validate and test PAD-UFES-20 dataset
"""

import sys
sys.path.insert(0,'../../..')
from constants import RAUG_PATH
sys.path.insert(0,RAUG_PATH)
import pandas as pd
import os
from raug.utils.loader import split_k_folder_csv, label_categorical_to_number

ISIC_BASE_PATH = "/home/patcha/Datasets/PAD-UFES-20"

clin_ = ["img_id", "diagnostic", "patient_id", "lesion_id", "biopsed"]

clin_feats = ["smoke", "drink", "background_father", "background_mother", "age", "pesticide", "gender", "skin_cancer_history",
                 "cancer_history", "has_piped_water", "has_sewage_system", "fitspatrick", "region", "diameter_1", "diameter_2",
                 "itch", "grew", "hurt", "changed", "bleed", "elevation"]

data_csv = pd.read_csv(os.path.join(ISIC_BASE_PATH, "metadata.csv")).fillna("EMPTY")
new_cli_cols = list()
for c in clin_feats:
    if c in ["age", "diameter_1", "diameter_2"]:
        new_cli_cols += [c]
        continue
    vals = [c+"_"+str(v) for v in  data_csv[c].unique()]
    try:
        vals.remove(c+"_EMPTY")
    except:
        pass
    new_cli_cols += vals

new_df = {c: list() for c in new_cli_cols}

for idx, row in data_csv.iterrows():
    _aux = list()
    _aux_in = list()
    for col in clin_feats:
        data_row = row[col]

        if data_row == 'EMPTY':
            pass
        elif col in ["age", "diameter_1", "diameter_2"]:
            _aux_in.append(col)
            new_df[col].append(data_row)
            continue
        else:
            _aux.append(col+"_"+str(data_row))

    for x in new_df:
        if x in _aux:
            new_df[x].append(1)
        elif x not in _aux_in:
            new_df[x].append(0)

new_df = pd.DataFrame.from_dict(new_df)
for col in clin_:
    new_df[col] = data_csv[col]

data = split_k_folder_csv(new_df, "diagnostic",
                   save_path=None, k_folder=6, seed_number=8)

data_test = data[ data['folder'] == 6]
data_train = data[ data['folder'] != 6]
data_test.to_csv(os.path.join(ISIC_BASE_PATH, "pad-ufes-20_parsed_test.csv"), index=False)
label_categorical_to_number (os.path.join(ISIC_BASE_PATH, "pad-ufes-20_parsed_test.csv"), "diagnostic",
                             col_target_number="diagnostic_number",
                             save_path=os.path.join(ISIC_BASE_PATH, "pad-ufes-20_parsed_test.csv"))

data_train = data_train.reset_index(drop=True)
data_train = split_k_folder_csv(data_train, "diagnostic",
                                save_path=None, k_folder=5, seed_number=8)

label_categorical_to_number (data_train, "diagnostic", col_target_number="diagnostic_number",
                             save_path=os.path.join(ISIC_BASE_PATH, "pad-ufes-20_parsed_folders.csv"))





