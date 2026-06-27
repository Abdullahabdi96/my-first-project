import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

LOG_FILE = "system_logs.csv"
OUTPUT_FILE = "ranked_anomalies.csv"
TOP_N = 20
CONTAMINATION = 0.05


def encode_column(series):
    encoder = LabelEncoder()
    return encoder.fit_transform(series.astype(str))


def load_logs(file_path):
    df = pd.read_csv(file_path)

    required_columns = ["timestamp", "host", "user", "event_type", "message"]

    for column in required_columns:
        if column not in df.columns:
            df[column] = "unknown"

    return df


def prepare_features(df):
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = df["timestamp"].dt.hour.fillna(0).astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)

    df["message"] = df["message"].astype(str)
    df["host"] = df["host"].astype(str)
    df["user"] = df["user"].astype(str)
    df["event_type"] = df["event_type"].astype(str)

    df["message_length"] = df["message"].str.len()
    df["user_event_count"] = df.groupby("user")["event_type"].transform("count")
    df["host_event_count"] = df.groupby("host")["event_type"].transform("count")
    df["event_type_count"] = df.groupby("event_type")["event_type"].transform("count")

    df["host_encoded"] = encode_column(df["host"])
    df["user_encoded"] = encode_column(df["user"])
    df["event_type_encoded"] = encode_column(df["event_type"])

    feature_columns = [
        "hour",
        "day_of_week",
        "message_length",
        "user_event_count",
        "host_event_count",
        "event_type_count",
        "host_encoded",
        "user_encoded",
        "event_type_encoded",
    ]

    X = df[feature_columns]

    return df, X


def detect_anomalies(df, X):
    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42
    )

    model.fit(X)

    df = df.copy()
    df["prediction"] = model.predict(X)
    df["anomaly_score"] = -model.decision_function(X)

    anomalies = df[df["prediction"] == -1].copy()
    anomalies = anomalies.sort_values("anomaly_score", ascending=False)

    return anomalies


def main():
    df = load_logs(LOG_FILE)
    df, X = prepare_features(df)
    anomalies = detect_anomalies(df, X)

    anomalies.to_csv(OUTPUT_FILE, index=False)

    print(f"Done. {len(anomalies)} suspicious events saved to '{OUTPUT_FILE}'")
    print()

    if len(anomalies) > 0:
        print("Top suspicious events:")
        print(
            anomalies[
                ["timestamp", "host", "user", "event_type", "message", "anomaly_score"]
            ]
            .head(TOP_N)
            .to_string(index=False)
        )
    else:
        print("No anomalies were found.")


if __name__ == "__main__":
    main()