import os
import pandas as pd
from tqdm.notebook import tqdm

def load_data_airbnb(csv, pbar_size=0):

    # Load .csv
    print("Loading .csv...")
    pbar = tqdm(total=pbar_size) # Progress bar
    chunks = []
    for chunk in pd.read_csv(csv, sep=';', chunksize=1000):
        pbar.update(1000)
        chunks.append(chunk)
    df = pd.concat(chunks)
    pbar.close()

    # Drop duplicates
    print("Dropping duplicate rows...")
    df = df.drop_duplicates(subset=["Name"], keep=False)
    df = df.drop_duplicates(subset=["Summary"], keep=False)

    # Convert columns to proper type and removes rows that don't fit
    print("Typecasting columns...")
    df = convert_column_to_int(df, "ID")
    df['First Review'] = pd.to_datetime(df['First Review'])

    print("Done!")
    return df

def convert_column_to_int(df, column):
    init_shape = df.shape
    
    df[column] = pd.to_numeric(df[column], errors='coerce').astype("Int64")
    df = df[df[column].notnull()]
    #df[column] = df[column].astype(pd.Int64Dtype())

    #print(column, init_shape, "->" ,df.shape)
    return df


def get_status_file():
    image_cols = ['ID', 'Thumbnail Url', 'Medium Url', 'Picture Url', 'XL Picture Url']
    
    # Download status dataframe
    if os.path.exists("download_status.csv"):
        dl_status = pd.read_csv('download_status.csv')
    else:
        dl_status = pd.DataFrame(columns=image_cols).astype(pd.BooleanDtype)