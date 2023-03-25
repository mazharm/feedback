import pandas as pd
from datetime import datetime, timedelta, timezone

# Define the column names to load from the CSV file
survey_cols = ["workspaceName", "translatedVebratim", "timestamp"]

def print_nps_surveys(workspace_name, min_days_ago, max_days_ago):
    # Calculate the date 30 days ago
    today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0,tzinfo=timezone.utc)
    start_days_ago = today - timedelta(days=max_days_ago)
    end_days_ago = today - timedelta(days=min_days_ago)

    # Read the CSV file into a dataframe
    df = pd.read_csv("PC_NPS_Surveys_WithWorkspaceName.csv", usecols=survey_cols)

    # Replace NaN, None, and numpy.nan values with an empty string
    df.fillna("", inplace=True)

    # Convert the timestamp column to datetime objects in UTC timezone
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Filter the dataframe based on workspaceName and timestamp
    filtered_df = df[(df["workspaceName"] == workspace_name) &
                    (pd.to_datetime(df["timestamp"]) >= start_days_ago) &
                    (pd.to_datetime(df["timestamp"]) <= end_days_ago)]

    # Print the translatedVebratim and timestamp values for the filtered rows
    for index, row in filtered_df.iterrows(): #pylint: disable=unused-variable
        if len(row["translatedVebratim"]) > 10:
            print(row["timestamp"], ": ", row["translatedVebratim"], "\n")

if __name__ == "__main__":
    while True:
        workspace_name = input("Enter a workspace name: ")
        min_days_ago = input("Enter the number of min days ago: ")
        max_days_ago = input("Enter the number of max days ago: ")
        print_nps_surveys(workspace_name, int(min_days_ago), int(max_days_ago))