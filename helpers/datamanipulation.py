import pandas

def remove_columns(columns: list[str], filename: str):
    df = pandas.read_csv(f"data/{filename}")
    df = df.drop(columns=columns)
    output_filename: str = filename.removesuffix(".csv") + "-processed.csv"
    df.to_csv(f"data/{output_filename}", index=False)

def remove_rows(rows: list[int], filename: str):
    df = pandas.read_csv(f"data/{filename}")
    df = df.drop(index=rows)
    output_filename: str = filename.removesuffix(".csv") + "-processed.csv"
    df.to_csv(f"data/{output_filename}", index=False)

def merge_csv(filename1: str, filename2: str):
    df1 = pandas.read_csv(f"data/{filename1}")
    df2 = pandas.read_csv(f"data/{filename2}")
    merged = pandas.concat([df1, df2])
    output_filename: str = "merged.csv"
    merged.to_csv(f"data/{output_filename}", index=False)