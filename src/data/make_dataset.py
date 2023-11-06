import os
import requests
from zipfile import ZipFile
import pandas as pd
from sklearn.model_selection import train_test_split


def download(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))


filtered_paranmt_url = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"
data_raw_dir = "../../data/raw/"
download(filtered_paranmt_url, data_raw_dir)

filtered_paranmt_zip_dir = data_raw_dir + filtered_paranmt_url.split('/')[-1]

with ZipFile(filtered_paranmt_zip_dir, 'r') as zip_ref:
    print(f'Unzipping {filtered_paranmt_zip_dir} to {data_raw_dir}')
    zip_ref.extractall(data_raw_dir)


raw_data_path = '../../data/raw/filtered.tsv'
df = pd.read_csv(raw_data_path, sep='\t', index_col=0)

swap_condition = df['ref_tox'] < df['trn_tox']

swapped_df = df.copy()
swapped_df.loc[swap_condition, ['reference', 'translation']] = swapped_df.loc[swap_condition, ['translation', 'reference']].values
swapped_df.loc[swap_condition, ['ref_tox', 'trn_tox']] = swapped_df.loc[swap_condition, ['trn_tox', 'ref_tox']].values


if not os.path.exists('../data/interim'):
    os.makedirs('../data/interim')
if not os.path.exists('../data/interim/model-outputs'):
    os.makedirs('../data/interim/model-outputs')

preprocessed_df_dir = '../../data/interim/preprocessed_filtered.csv'
swapped_df.to_csv(preprocessed_df_dir)

train_df, test_df = train_test_split(swapped_df, test_size=0.025, random_state=42)  # Low percent due to computational time limits
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)


train_df_dir = '../../data/interim/train.csv'
val_df_dir = '../../data/interim/val.csv'
test_df_dir = '../../data/interim/test.csv'

train_df.to_csv(train_df_dir)
val_df.to_csv(val_df_dir)
test_df.to_csv(test_df_dir)

print(len(train_df), len(val_df), len(test_df))
