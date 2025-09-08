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
    data["opp_code"] = data["opposition"].astype("category").cat.codes

    # Converts time into just hour e.g 16:30 -> 16
    data["hour"] = data["times"].str.replace(":.+", "", regex=True).astype("int")

    # Creates a code for each day of the week
    data["day_code"] = data["date"].dt.day_of_week

    return data


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

    # Train the model
    train_model(matches)


if __name__ == "__main__":
    main()
