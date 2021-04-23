import pandas as pd


def merge_labels_and_metadata (df_labels, df_cli_feat, save_path=None):

    if isinstance(df_labels, str):
        df_labels = pd.read_csv(df_labels)

    if isinstance(df_cli_feat, str):
        df_cli_feat = pd.read_csv(df_cli_feat)


    df_labels_cli_feat = df_cli_feat
    df_labels_cli_feat["diagnostic"] = ''

    for pos, row_lab in enumerate(df_labels.iterrows()):
        img_name = row_lab[1]['image']
        label = ''

        if row_lab[1]['MEL'] == 1.0:
            label = 'MEL'
        elif row_lab[1]['NV'] == 1.0:
            label = 'NV'
        elif row_lab[1]['BCC'] == 1.0:
            label = 'BCC'
        elif row_lab[1]['AK'] == 1.0:
            label = 'AK'
        elif row_lab[1]['BKL'] == 1.0:
            label = 'BKL'
        elif row_lab[1]['DF'] == 1.0:
            label = 'DF'
        elif row_lab[1]['VASC'] == 1.0:
            label = 'VASC'
        elif row_lab[1]['SCC'] == 1.0:
            label = 'SCC'
        elif row_lab[1]['UNK'] == 1.0:
            label = 'UNK'
        else:
            raise Exception("There is no label named {} in the dataset".format(row_lab[1]))

        df_labels_cli_feat["diagnostic"][pos] = label
        print ("{}-Img: {} | Label: {}".format(pos, img_name, label))

    if save_path is not None:
        df_labels_cli_feat.to_csv(os.path.join(save_path, "ISIC2019.csv"), index=False)

    return df_labels_cli_feat


CSV_META_PATH = "ISIC_2019_Training_GroundTruth.csv"
CSV_LABELS_PATH = "ISIC_2019_Training_Metadata.csv"
SAVE_PATH = "./"


df_merge = merge_labels_and_metadata(CSV_META_PATH, CSV_LABELS_PATH, SAVE_PATH)



