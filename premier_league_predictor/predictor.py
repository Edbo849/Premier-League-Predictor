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

    # Goal-based features
    data["goal_diff"] = data["gf"] - data["ga"]
    data["xg_diff"] = data["xg"] - data["xga"]

    # Shooting efficiency metrics
    data["shot_accuracy"] = data["sot"] / data["sh"]
    data["shot_accuracy"] = data["shot_accuracy"].fillna(0)

    data["goals_per_shot"] = data["gf"] / data["sh"]
    data["goals_per_shot"] = data["goals_per_shot"].fillna(0)

    # Expected goals efficiency
    data["xg_efficiency"] = data["gf"] / data["xg"]
    data["xg_efficiency"] = data["xg_efficiency"].fillna(0)

    # Defensive metrics
    data["shots_conceded_per_ga"] = data["xga"] / data["ga"]
    data["shots_conceded_per_ga"] = data["shots_conceded_per_ga"].fillna(0)

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
        n_estimators=200,  # More trees
        max_depth=15,  # Deeper trees
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=1,
        class_weight="balanced",
        bootstrap=True,
        oob_score=True,  # Out-of-bag scoring
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


def add_opposition_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add features about opposition strength."""

    # Calculate team strength metrics (only use existing columns)
    team_stats = (
        data.groupby("team")
        .agg({"xg": "mean", "xga": "mean", "gf": "mean", "ga": "mean"})
        .reset_index()
    )

    team_stats.columns = [
        "opponent",
        "opp_avg_xg",
        "opp_avg_xga",
        "opp_avg_gf",
        "opp_avg_ga",
    ]

    # Merge opposition stats
    data = data.merge(team_stats, on="opponent", how="left")

    return data


def add_form_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add recent form and momentum indicators.
    Args:
        data (pd.DataFrame): Match data.
    Returns:
        pd.DataFrame: Match data with form features added.
    """

    # Recent results streak
    data["points"] = data["result"].map({"W": 3, "D": 1, "L": 0})

    # Sort by team and date to ensure proper rolling calculations
    data = data.sort_values(["team", "date"])

    # Rolling form over last 5 games - use transform to maintain original index
    data["form_5"] = data.groupby("team")["points"].transform(
        lambda x: x.rolling(5, closed="left").sum()
    )

    # Win percentage in last 10 games - use transform
    data["win_rate_10"] = data.groupby("team")["target"].transform(
        lambda x: x.rolling(10, closed="left").mean()
    )

    # Goal scoring form - use transform
    data["goals_last_3"] = data.groupby("team")["gf"].transform(
        lambda x: x.rolling(3, closed="left").sum()
    )

    return data


def create_multiple_rolling_averages(
    group: pd.DataFrame, cols: List[str]
) -> pd.DataFrame:
    """Create rolling averages with different window sizes.
    Args:
        group (pd.DataFrame): DataFrame for a single team group.
        cols (List[str]): List of columns to calculate rolling averages for.
    Returns:
        pd.DataFrame: DataFrame with multiple rolling averages added."""

    group = group.sort_values("date")

    # Different window sizes for different insights
    for window in [3, 5, 10]:
        rolling_stats = group[cols].rolling(window, closed="left").mean()
        new_cols = [f"{c}_rolling_{window}" for c in cols]
        group[new_cols] = rolling_stats

    return group


def select_best_features(
    matches: pd.DataFrame, predictors: List[str], target: str
) -> List[str]:
    """Select the most important features using Random Forest feature importance.
    Args:
        matches (pd.DataFrame): Match data with features and target.
        predictors (List[str]): List of feature columns to evaluate.
        target (str): Target column name.
    Returns:
        List[str]: List of top 20 important features.
    """

    # Train a model to get feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=1)
    train_data = matches[matches["date"] < "2022-01-01"]

    rf.fit(train_data[predictors], train_data[target])

    # Get feature importance
    importance_df = pd.DataFrame(
        {"feature": predictors, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    # Return top 20 features
    return importance_df.head(20)["feature"].tolist()


def main():
    """Enhanced main function with all improvements."""

    # Read and clean data
    matches = pd.read_csv("premier_league_predictor/matches.csv", index_col=0)
    matches = clean_data(matches)

    # Create binary target variable: 1 for wins, 0 for draws/losses
    matches["target"] = (matches["result"] == "W").astype("int")

    matches = add_form_features(matches)
    matches = add_opposition_features(matches)

    # Enhanced feature set
    base_cols = [
        "gf",
        "ga",
        "sh",
        "sot",
        "dist",
        "fk",
        "pk",
        "pkatt",
        "xg",
        "xga",
        "poss",
    ]
    engineered_cols = [
        "goal_diff",
        "xg_diff",
        "shot_accuracy",
        "goals_per_shot",
        "xg_efficiency",
    ]

    all_feature_cols = base_cols + engineered_cols

    # Create rolling averages
    matches_rolling = matches.groupby("team").apply(
        lambda x: create_multiple_rolling_averages(x, all_feature_cols)
    )

    # Remove multi-index and reset
    matches_rolling = matches_rolling.droplevel("team")
    matches_rolling.index = pd.RangeIndex(start=0, stop=matches_rolling.shape[0])

    # Define all possible predictors (updated with new opposition features)
    rolling_cols = []
    for col in all_feature_cols:
        for window in [3, 5, 10]:
            rolling_cols.append(f"{col}_rolling_{window}")

    all_predictors = [
        "venue_code",
        "opp_code",
        "hour",
        "day_code",
        "form_5",
        "win_rate_10",
        "goals_last_3",
        "opp_avg_xg",
        "opp_avg_xga",
        "opp_avg_gf",
        "opp_avg_ga",
    ] + rolling_cols

    # Select best features
    best_predictors = select_best_features(matches_rolling, all_predictors, "target")

    # Make predictions with best features
    combined, precision = make_predictions(matches_rolling, best_predictors)

    print(f"Precision: {precision:.2f}")
    print(f"Predictions shape: {combined.shape}")
    print(combined.head())


if __name__ == "__main__":
    main()
