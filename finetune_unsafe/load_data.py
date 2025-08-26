import pandas as pd


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


if __name__ == "__main__":
    df = load_data("do_not_answer_en.csv")
    print(df.head())
    # fitler to Misinformation Harms
    df = df[df["risk_area"] == "Malicious Uses"]
    df = df[~df["specific_harms"].str.contains("Sexual Exploitation")]

    # drop texts with certain keywords

    # save to jsonl
    df.to_json("malicious_uses.jsonl", orient="records", lines=True)