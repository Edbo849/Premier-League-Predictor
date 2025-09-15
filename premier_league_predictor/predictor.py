"""
Main module for Premier League match prediction.

This module will contain the core functionality for predicting
Premier League match outcomes.
"""

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
    data["hour"] = data["times"].str.replace(":.+", "", regex=True).astype("int")

    # Creates a code for each day of the week
    data["day_code"] = data["date"].dt.day_of_week

    return data


def rolling_averages(group: pd.DataFrame, cols: list, new_cols: list) -> pd.DataFrame:
    """Calculates rolling averages for specified columns.
    Args:
        group (pd.DataFrame): Grouped match data for a team.
        cols (list): List of columns to calculate rolling averages for.
        new_cols (list): List of new column names for the rolling averages.
    Returns:
        pd.DataFrame: DataFrame with rolling averages added."""

    # Sort by date
    group = group.sort_values("date")

    # Calculate rolling averages
    rolling_stats = group[cols].rolling(3, closed="left").mean()

    # Rename columns
    group[new_cols] = rolling_stats

    # Drop rows with NaN values in new columns
    group = group.dropna(subset=new_cols)

    return group


def train_model(matches: pd.DataFrame) -> None:
    """Trains the Random Forest model on the match data.
    Args:
        matches (pd.DataFrame): Cleaned match data with target variable.
    Returns:
        None
    """

    # Initialise the model
    rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

    # Data to train
    train = matches[matches["date"] < "2022-01-01"]

    # Set predictors for model
    predictors = ["venue_code", "opp_code", "hour", "day_code"]

    # Train the model
    rf.fit(train[predictors], train["target"])


def main():
    """Main entry point for the application."""

    # Read the csv file and identify the index column
    matches = pd.read_csv("matches.csv", index_col=0)

    # Clean the data
    matches = clean_data(matches)

    # Set target for ML model
    matches["target"] = (matches["result"] == "W").astype("int")

    # Define group for rolling averages
    group_matches = matches.groupby("team")

    group = group_matches.get_group("Manchester City").sort_values("date")

    # Define columns for rolling averages
    cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
    new_cols = [f"{c}_rolling" for c in cols]

    matches_rolling = matches.groupby("team").apply(
        lambda x: rolling_averages(x, cols, new_cols)
    )
    matches_rolling = matches_rolling.droplevel("team")
    matches_rolling.index = pd.RangeIndex(start=0, stop=matches_rolling.shape[0])

    # Train the model
    train_model(matches_rolling)


if __name__ == "__main__":
    main()
