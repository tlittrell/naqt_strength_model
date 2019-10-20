import numpy as np
import pandas as pd
from toolz.functoolz import pipe


def process_game_data(df: pd.DataFrame) -> pd.DataFrame:

    result = pipe(
        df,
        rename_features,
        convert_types,
        remove_forfeits,
        add_features,
        drop_features,
    )
    # Add a unique numeric index for each team
    team_indices = (
        pd.DataFrame(
            list(set(result["team_1"].unique()) | set(result["team_2"].unique()))
        )
        .reset_index()
        .rename({0: "team"}, axis=1)
        .set_index("team")
        .to_dict()["index"]
    )
    result["team_1_index"] = result["team_1"].map(team_indices)
    result["team_2_index"] = result["team_2"].map(team_indices)

    assert result[result["team_1_index"].isna()].empty
    assert result[result["team_2_index"].isna()].empty

    # Find and remove duplicate games
    result["teams"] = [
        tuple(sorted(x)) for x in zip(result["team_1"], result["team_2"])
    ]
    result = result.drop_duplicates(["Round", "teams"]).drop(["teams"], axis=1)

    print(result.shape)
    result.columns = result.columns.str.lower()

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
    return df.rename(rename_dict, axis=1)


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
    return df.query("Score != 'Forfeit'")


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
    result = df.drop(["Result"], axis=1)
    return result
