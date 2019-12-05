#!/usr/bin/env python

def cleanup_data(df):
    for item in list(df.columns.values):
        na_len = df[item].isna().sum()
        total_len = df[item].count()

        if na_len > 0.7*total_len:
            del df[item]
    columns = list(df.columns.values)
    col = 0

    df.loc[df.index.dropna()]

    for index,row in df.iterrows():
        removed = False
        col = 0
        while col < len(columns) and removed == False:
            if row[columns[col]] != row[columns[col]]:
                df.drop(index, inplace=True)
                removed = True
            col = col+1


