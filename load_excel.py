import pandas as pd


def load_clean_split(path):
    df = pd.read_excel(path, header=None)
    df.columns = ["סעיף", "תאור", "יח", "כמות", "מחיר", 'סה"כ']
   
    df = df.replace(r'^\s*$', None, regex=True)
    df = df[df["סעיף"].notna()]
    df = df.drop(df.index[0], axis=0)
    df = df.reset_index(drop=True)
    df = df.drop(['סה"כ', 'כמות'], axis=1)

    df["chapter"] = df["סעיף"].str.extract(r"^(\d{2})")
    # headers = df[df["סעיף"].str.count(r"\.") == 0]
    items = df[df["סעיף"].str.count(r"\.") == 2]
    chapters = {
        ch: items[items["chapter"] == ch]
        for ch in items["chapter"].unique()
    }    
    return chapters
