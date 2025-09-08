"""
Main module for Premier League match prediction.

This module will contain the core functionality for predicting
Premier League match outcomes.
"""

import pandas as pd


def main():
    """Main entry point for the application."""

    # Read the csv file and identify the index column
    matches = pd.read_csv("matches.csv", index_col=0)

    # Convert date from string to datetime
    matches["date"] = pd.to_datetime(matches["date"])

    # Creates a code for Home/Away
    matches["venue_code"] = matches["venue"].astype("category").cat.codes

    # Creates a code for each opposition
    matches["opp_code"] = matches["opposition"].astype("category").cat.codes

    # Converts time into just hour e.g 16:30 -> 16
    matches["hour"] = matches["times"].str.replace(":.+", "", regex=True).astype("int")

    # Creates a code for each day of the week
    matches["day_code"] = matches["date"].dt.day_of_week

    # Set target for ML model
    matches["target"] = (matches["result"] == "W").astype("int")


if __name__ == "__main__":
    main()
