"""
Main module for Premier League match prediction.

This module will contain the core functionality for predicting
Premier League match outcomes.
"""

from typing import List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import pandas as pd


class MissingDict(dict):
    """Dictionary that returns the key itself if the key is not found."""

    __missing__ = lambda self, key: key


# Map team names from data source to standardised names
map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}
# Create mapping dictionary that returns original name if not found in map_values
mapping = MissingDict(**map_values)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Cleans the input data for model training.

    Args:
        data (pd.DataFrame): Raw match data.

    Returns:
        pd.DataFrame: Cleaned match data.
    """
    # Convert date from string to datetime for proper date operations
    data["date"] = pd.to_datetime(data["date"])

    # Creates a numeric code for Home/Away venue (0 or 1)
    data["venue_code"] = data["venue"].astype("category").cat.codes

    # Creates a numeric code for each opposition team
    data["opp_code"] = data["opponent"].astype("category").cat.codes

    # Converts time into just hour e.g 16:30 -> 16 for kick-off time feature
    data["hour"] = data["time"].str.replace(":.+", "", regex=True).astype("int")

    # Creates a numeric code for each day of the week (0=Monday, 6=Sunday)
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

    # Sort by date to ensure chronological order for rolling averages
    group = group.sort_values("date")

    # Calculate rolling averages over last 3 matches (excluding current match)
    rolling_stats = group[cols].rolling(3, closed="left").mean()

    # Add rolling average columns to the group DataFrame
    group[new_cols] = rolling_stats

    # Drop rows with NaN values in new columns (first few matches won't have enough history)
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
    # Initialise Random Forest with specific parameters for stability and performance
    rf = RandomForestClassifier(
        n_estimators=100,  # More trees for better performance
        min_samples_split=5,  # Reduced to capture more patterns
        max_depth=10,  # Prevent overfitting
        random_state=1,
        class_weight="balanced",  # Handle class imbalance
    )

    # Use data before 2022 for training to avoid data leakage
    train = matches[matches["date"] < "2022-01-01"]

    # Train the model on predictor features and target outcomes
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
    # Train the model using historical data
    rf = train_model(matches, predictors)

    # Use data from 2022 onwards for testing
    test = matches[matches["date"] > "2022-01-01"]

    # Generate predictions for the test set
    preds = rf.predict(test[predictors])

    # Create DataFrame combining actual results, predictions, and team names
    combined = pd.DataFrame(
        dict(actual=test["target"], predicted=preds, team=test["team"]),
        index=test.index,
    )

    # Calculate precision score (true positives / (true positives + false positives))
    precision = float(precision_score(test["target"], preds))
    return combined, precision


def main():
    """Main entry point for the application."""

    # Read the CSV file from the correct path (same directory as script)
    matches = pd.read_csv("premier_league_predictor/matches.csv", index_col=0)

    # Clean and prepare the data for machine learning
    matches = clean_data(matches)

    # Create binary target variable: 1 for wins, 0 for draws/losses
    matches["target"] = (matches["result"] == "W").astype("int")

    # Define columns for calculating rolling averages (team performance metrics)
    cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt", "xg"]
    new_cols = [f"{c}_rolling" for c in cols]

    # Calculate rolling averages for each team separately
    matches_rolling = matches.groupby("team").apply(
        lambda x: rolling_averages(x, cols, new_cols)
    )

    # Remove the multi-level index created by groupby
    matches_rolling = matches_rolling.droplevel("team")

    # Reset index to simple range index for easier data manipulation
    matches_rolling.index = pd.RangeIndex(start=0, stop=matches_rolling.shape[0])

    # Define all predictor features (static features + rolling averages)
    predictors = ["venue_code", "opp_code", "hour", "day_code"] + new_cols

    # Generate predictions and calculate precision
    combined, precision = make_predictions(matches_rolling, predictors)

    # Merge predictions with additional match information
    combined = combined.merge(
        matches_rolling[
            [
                "date",
                "opponent",
                "result",
            ]
        ],
        left_index=True,
        right_index=True,
        suffixes=("", "_matches_rolling"),  # Avoid column name conflicts
    )

    # Apply team name mapping for standardisation
    combined["team"] = combined["team"].map(mapping)

    # Display results
    print(f"Precision: {precision:.2f}")
    print(f"Predictions shape: {combined.shape}")
    print(combined.head())


if __name__ == "__main__":
    main()
