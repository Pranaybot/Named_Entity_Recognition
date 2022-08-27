import pandas as pd
import json

def save_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

TRAIN_DATA = []
entities = []
df = pd.read_csv('Train_Tagged_Titles.tsv', sep='\t', usecols=['Title', 'Token', 'Tag'])
df.fillna('No Tag', inplace=True)
for ind in df.index:
    entities.append((df['Token'][ind].index(df['Token'][ind][0]),
                     df['Token'][ind].index(df['Token'][ind][-1]),
                     df['Tag'][ind]))
    TRAIN_DATA.append((df['Title'][ind], {"entities": entities}))
    entities = []
save_data("training_data.json", TRAIN_DATA)

