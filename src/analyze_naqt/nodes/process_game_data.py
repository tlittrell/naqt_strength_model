import numpy as np
import pandas as pd
from toolz.functoolz import pipe


def process_game_data(df: pd.DataFrame) -> pd.DataFrame:

    result = pipe(
        df,
        rename_features,
        remove_forfeits,
        add_features,
        drop_features,
        clean_team_names,
        create_team_indices,
        remove_duplicate_games,
    )

    return result


def rename_features(df: pd.DataFrame) -> pd.DataFrame:
    rename_dict = {
        "P": "team_1_powers",
        "TU": "team_1_tens",
        "I": "team_1_negs",
        "B": "team_1_bonus_points",
        "PPB": "team_1_ppb",
        "P.1": "team_2_powers",
        "TU.1": "team_2_tens",
        "I.1": "team_2_negs",
        "B.1": "team_2_bonus_points",
        "PPB.1": "team_2_ppb",
        "team": "team_1",
        "Opponent": "team_2",
    }
    result = df.rename(rename_dict, axis=1)
    result.columns = result.columns.str.lower()
    return result


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    type_dict = {
        "team_1_powers": np.uint8,
        "team_1_tens": np.uint8,
        "team_1_negs": np.uint8,
        "team_1_bonus_points": np.uint16,
        "team_2_powers": np.uint8,
        "team_2_tens": np.uint8,
        "team_2_negs": np.uint8,
        "team_2_bonus_points": np.uint16,
    }
    return df.astype(type_dict)


def remove_forfeits(df: pd.DataFrame) -> pd.DataFrame:
    return df.query("score != 'Forfeit'")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    result = (
        df.eval(
            "team_1_score = team_1_powers * 15 + team_1_tens * 10 - team_1_negs * 5 + team_1_bonus_points"
        )
        .eval(
            "team_2_score = team_2_powers * 15 + team_2_tens * 10 - team_2_negs * 5 + team_2_bonus_points"
        )
        .eval("point_diff = team_1_score - team_2_score")
    )
    return result


def drop_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.drop(["result"], axis=1)
    return result


def clean_team_names(df: pd.DataFrame) -> pd.DataFrame:
    def clean_ampersand(x):
        return x.replace("&amp;", "&")

    df["team_1"] = df["team_1"].apply(clean_ampersand)
    df["team_2"] = df["team_2"].apply(clean_ampersand)
    return df


def create_team_indices(df: pd.DataFrame) -> pd.DataFrame:
    # Add a unique numeric index for each team
    team_indices = (
        pd.DataFrame(list(set(df["team_1"].unique()) | set(df["team_2"].unique())))
        .reset_index()
        .rename({0: "team"}, axis=1)
        .set_index("team")
        .to_dict()["index"]
    )
    df["team_1_index"] = df["team_1"].map(team_indices)
    df["team_2_index"] = df["team_2"].map(team_indices)

    assert df[df["team_1_index"].isna()].empty
    assert df[df["team_2_index"].isna()].empty
    return df


def remove_duplicate_games(df: pd.DataFrame) -> pd.DataFrame:
    # Find and remove duplicate games
    df["teams"] = [tuple(sorted(x)) for x in zip(df["team_1"], df["team_2"])]
    df = df.drop_duplicates(["round", "teams"]).drop(["teams"], axis=1)
    return df
