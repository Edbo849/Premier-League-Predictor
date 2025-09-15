"""
Main module for Premier League match prediction.

This module will contain the core functionality for predicting
Premier League match outcomes.
"""

from typing import List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import pandas as pd


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Cleans the input data for model training.

    Args:
        data (pd.DataFrame): Raw match data.

    Returns:
        pd.DataFrame: Cleaned match data.
    """
    # Convert date from string to datetime
    data["date"] = pd.to_datetime(data["date"])

    # Creates a code for Home/Away
    data["venue_code"] = data["venue"].astype("category").cat.codes

    # Creates a code for each opposition
    data["opp_code"] = data["opponent"].astype("category").cat.codes

    # Converts time into just hour e.g 16:30 -> 16
    data["hour"] = data["time"].str.replace(":.+", "", regex=True).astype("int")

    # Creates a code for each day of the week
    data["day_code"] = data["date"].dt.day_of_week

    return data


def rolling_averages(
    group: pd.DataFrame, cols: List[str], new_cols: List[str]
) -> pd.DataFrame:
    """Calculates rolling averages for specified columns.

    Args:
        group (pd.DataFrame): DataFrame for a single team group.
        cols (List[str]): List of columns to calculate rolling averages for.
        new_cols (List[str]): List of new column names for the rolling averages.

    Returns:
        pd.DataFrame: DataFrame with rolling averages added.
    """

    # Sort by date
    group = group.sort_values("date")

    # Calculate rolling averages
    rolling_stats = group[cols].rolling(3, closed="left").mean()

    # Rename columns
    group[new_cols] = rolling_stats

    # Drop rows with NaN values in new columns
    group = group.dropna(subset=new_cols)

    return group


def train_model(matches: pd.DataFrame, predictors: List[str]) -> RandomForestClassifier:
    """Trains and returns the Random Forest model.

    Args:
        matches (pd.DataFrame): Match data with features and target.
        predictors (List[str]): List of feature columns to use for training.

    Returns:
        RandomForestClassifier: Trained Random Forest model.
    """
    rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
    train = matches[matches["date"] < "2022-01-01"]
    rf.fit(train[predictors], train["target"])
    return rf


def make_predictions(
    matches: pd.DataFrame, predictors: List[str]
) -> Tuple[pd.DataFrame, float]:
    """Makes predictions using trained model.

    Args:
        matches (pd.DataFrame): Match data with features and target.
        predictors (List[str]): List of feature columns to use for prediction.

    Returns:
        Tuple[pd.DataFrame, float]: Combined predictions and actual results, and precision score.
    """
    rf = train_model(matches, predictors)
    test = matches[matches["date"] > "2022-01-01"]
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(
        dict(actual=test["target"], predicted=preds), index=test.index
    )
    precision = float(precision_score(test["target"], preds))
    return combined, precision


def main():
    """Main entry point for the application."""

    # Read the csv file and identify the index column
    matches = pd.read_csv("premier_league_predictor/matches.csv", index_col=0)

    # Clean the data
    matches = clean_data(matches)

    # Set target for ML model
    matches["target"] = (matches["result"] == "W").astype("int")

    # Define columns for rolling averages
    cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
    new_cols = [f"{c}_rolling" for c in cols]

    # Calculate rolling averages for each team
    matches_rolling = matches.groupby("team").apply(
        lambda x: rolling_averages(x, cols, new_cols)
    )
    matches_rolling = matches_rolling.droplevel("team")
    matches_rolling.index = pd.RangeIndex(start=0, stop=matches_rolling.shape[0])

    # Define predictors (include rolling averages)
    predictors = ["venue_code", "opp_code", "hour", "day_code"] + new_cols

    # Make predictions and get results
    combined, precision = make_predictions(matches_rolling, predictors)

    combined = combined.merge(
        matches_rolling[
            [
                "date",
                "team",
                "opponent",
                "result",
            ]
        ],
        left_index=True,
        right_index=True,
    )

    print(f"Precision: {precision}")
    print(f"Predictions shape: {combined.shape}")
    print(combined.head())


if __name__ == "__main__":
    main()
