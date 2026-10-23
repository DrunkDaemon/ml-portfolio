import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering shared by training and the web app."""
    df = df.copy()
    df["BedroomRatio"] = df["AveBedrms"] / df["AveRooms"]
    df["PeoplePerRoom"] = df["AveOccup"] / df["AveRooms"]
    df["IncomePerRoom"] = df["MedInc"] / df["AveRooms"]
    return df
