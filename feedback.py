"""
Tool to process NPS feedback
"""
import argparse
import logging
import time
from multiprocessing import Pool, cpu_count
from datetime import datetime, timedelta, timezone
import pandas as pd
from openaicli import OpenAICli  # pylint: disable=import-error
from prompt import PromptType  # pylint: disable=import-error
from csvtodoc import CSVtoDOCX  # pylint: disable=import-error

DEFAULT_SHARD_SIZE = 500
DEFAULT_MIN_TEXT_LENGTH = 10

WORKSPACE_INPUT = 'workspaceName'
VERBATIM_INPUT = 'translatedVebratim'
TIMESTAMP = 'timestamp'
VERBATIM_OUTPUT = 'Verbatim'
WORKSPACE_OUTPUT = 'Workspace'
ID = 'ID'
SCORE = 'Score'
CLASSIFICATION = 'classification'
TOPICS = 'topics'
SUMMARY = 'Summary'
TOP_QUOTES = 'Top Quotes'
ACTION_ITEMS = 'Action Items'
CSP_DOCX_TITLE = 'Partner Center NPS Feedback Summary - CSP Partners'
ISV_DOCX_TITLE = 'Partner Center NPS Feedback Summary - ISV Partners'
MCPP_DOCX_TITLE = 'Partner Center NPS Feedback Summary'
MSX_DOCX_TITLE = 'MSX NPS Feedback Summary'

# Input files
PC_SURVEYS_INPUT_FILE = 'local/input/PC_NPS_Surveys_WithWorkspaceName.csv'
MSX_SURVEYS_INPUT_FILE = 'local/input/MSX_NPS_Surveys_WithWorkspaceName.csv'

# Raw output files
PC_RAW_OUTPUT_FILE = 'local/csv/pc_raw.csv'
MSX_RAW_OUTPUT_FILE = 'local/csv/msx_raw.csv'

# Interim output files
PC_SUMMARIES_INTERIM_OUTPUT_FILE = 'local/csv/pc_summaries_interim.csv'
MSX_SUMMARIES_INTERIM_OUTPUT_FILE = 'local/csv/msx_summaries_interim.csv'

# CSV output files
PC_SUMMARIES_OUTPUT_FILE = 'local/csv/pc_summaries.csv'
PC_THOUGHTFULNESS_OUTPUT_FILE = 'local/csv/pc_thoughtfulness_topics.csv'
MSX_SUMMARIES_OUTPUT_FILE = 'local/csv/msx_summaries.csv'
MSX_THOUGHTFULNESS_OUTPUT_FILE = 'local/csv/msx_thoughtfulness_topics.csv'

# Summaries docx output files
CSP_SUMMARIES_DOCX_OUTPUT_FILE = 'local/docs/csp_summaries.docx'
MCPP_SUMMARIES_DOCX_OUTPUT_FILE = 'local/docs/mcpp_summaries.docx'
ISV_SUMMARIES_DOCX_OUTPUT_FILE = 'local/docs/isv_summaries.docx'
MSX_SUMMARIES_DOCX_OUTPUT_FILE = 'local/docs/msx_summaries.docx'

# Set up logging
logging.basicConfig(filename='local/logs/error.log', level=logging.ERROR)

oac = OpenAICli()

# These columns are used to create the output CSV file that contains
# the thoughtfulness scores and key topics for each piece of feedback
full_columns = [ID, WORKSPACE_INPUT, VERBATIM_INPUT, CLASSIFICATION, TOPICS]

# Define the column names to load from the input CSV file
survey_cols = [WORKSPACE_INPUT, VERBATIM_INPUT, TIMESTAMP]

# Define the column names for the summary output CSV file
summary_columns = [WORKSPACE_OUTPUT, SUMMARY, TOP_QUOTES, ACTION_ITEMS]

# Define the column names for the DF used to compute the thoughtfulness scores
score_columns = [ID, CLASSIFICATION, TOPICS]

dtypes = {WORKSPACE_INPUT: str, VERBATIM_INPUT: str, TIMESTAMP: str}
summary_dtypes = {WORKSPACE_OUTPUT: str, SUMMARY: str,
                  TOP_QUOTES: str, ACTION_ITEMS: str}

CSP_WORKSPACES = [
    'Pricing',
    'Billing',
    'Customer',
    'Incentives',
    'Apis and Integration',
    'Insights',
    'Accounts',
    'Action Center',
    'Help + Support',
]

MCPP_WORKSPACES = [
    'Enrollment',
    'Membership',
    'Benefits',
    'Referrals',
    'Collaborate',
    'Incentives',
    'Insights',
    'Accounts',
    'Action Center',
    'Help + Support',
]

ISV_WORKSPACES = [
    'Marketplace Offers',
    'Membership',
    'Apis and Integration',
    'Accounts',
    'Action Center',
    'Benefits',
    'Collaborate',
    'Enrollment',
    'Help + Support',
    'Incentives',
    'Insights',
    'Payouts',
    'Referrals',
]

PC_WORKSPACES = [
    'Accounts',
    'Action Center',
    'Apis and Integration',
    'Benefits',
    'Billing',
    'Collaborate',
    'Customer',
    'Enrollment',
    'Help + Support',
    'Incentives',
    'Insights',
    'Marketplace Offers',
    'Membership',
    'Payouts',
    'Pricing',
    'Referrals',
]

MSX_WORKSPACES = [
    'Account Planning',
    'Accounts 360',
    'Compensation',
    'Contacts',
    'Leads',
    'Opportunities',
    'Partners',
    'Proposals',
    'Sales Accelerator',
    'User Provisioning',
]

# Define the thoughtfulness score ranges and corresponding labels
thoughtfulness_classfications = ['Thoughtful', 'Not Thoughtful']


def get_score(tuples):
    """
    This function returns the thoughtfulness score for the given text
    """
    d_f = pd.DataFrame(columns=score_columns)

    json_response = oac.get_analysis(tuples)
    d_f = pd.DataFrame(json_response)

    return d_f


def process_raw_feedback_df(feedback_df):
    """
    process the feedback and each row and extract thoughfulness
    and key topics from each piece of feedback
    """
    # Get the thoughfulness classification for each chunk of 10 rows
    chunk_size = 10
    chunks = [(feedback_df[[ID, VERBATIM_INPUT]][i:i+chunk_size]).to_numpy()
              for i in range(0, len(feedback_df), chunk_size)]
    score_chunks = [get_score([tuple(x) for x in chunk])
                    for chunk in chunks]

    if len(score_chunks) == 0:
        return None

    scores_df = pd.concat(score_chunks)

    if len(scores_df) > 0:

        # Add blank columns to the dataframe
        feedback_df[CLASSIFICATION] = pd.Series(dtype='str')
        feedback_df[TOPICS] = pd.Series(dtype='str')

        # Set "ID" as the index for both dataframes
        feedback_df.set_index(ID, inplace=True)
        scores_df.set_index(ID, inplace=True)

        # Update master dataframe with latest data from update dataframe
        feedback_df.update(scores_df)

    return feedback_df


def consolidate_summaries_df(d_f):
    """
    This function summarizes the feedback in a given dataframe
    For a given dataframe, it will summarize the feedback in chunks of 500 rows
    It will result an array of summaries
    """

    chunk_size = 4

    # pick chunks of items summarize
    summaries_full = [list(zip(d_f[SUMMARY][i:i+chunk_size],
                               d_f[TOP_QUOTES][i:i+chunk_size],
                               d_f[ACTION_ITEMS][i:i+chunk_size]))
                      for i in range(0, len(d_f), chunk_size)]

    summaries = []
    for chunk in summaries_full:
        s_c, t_c, a_c = [], [], []
        for _s, _t, _a in chunk:
            s_c.append(_s)
            t_c.append(_t)
            a_c.append(_a)

        text_summary = oac.get_summary(s_c, PromptType.CONSOLIDATE_SUMMARY)
        action_items = oac.get_summary(
            a_c, PromptType.CONSOLIDATE_ACTION_ITEMS)
        top_quotes = oac.get_summary(t_c, PromptType.CONSOLIDATE_TOP_QUOTES)

        summaries.append([text_summary, action_items, top_quotes])

    return summaries


def summarize_df(d_f):
    """
    This function summarizes the feedback in a given dataframe
    For a given dataframe, it will summarize the feedback in chunks of 500 rows
    It will result an array of summaries
    """

    chunk_size = 20

    # pick chunks of items summarize
    summaries_full = [list(d_f[VERBATIM_INPUT][i:i+chunk_size], )
                      for i in range(0, len(d_f), chunk_size)]

    summaries = []
    for summary_chunk in summaries_full:
        text_summary = oac.get_summary(summary_chunk, PromptType.SUMMARY)
        action_items = oac.get_summary(summary_chunk, PromptType.ACTION_ITEMS)
        top_quotes = oac.get_summary(summary_chunk, PromptType.TOP_QUOTES)

        summaries.append([text_summary, action_items, top_quotes])

    return summaries


def summarize_workspace(workspace, d_f, consolidate):
    """
    This function summarizes the feedback for a given workspace
    """
    df_summaries = pd.DataFrame(columns=summary_columns)

    if consolidate:
        summaries = consolidate_summaries_df(d_f)
    else:
        summaries = summarize_df(d_f)

    if consolidate:
        print(
            f"Workspace:{workspace}, consolidated {d_f.shape[0]} summaries into {len(summaries)}")

    for summary in summaries:
        # Create a dictionary of values
        new_row = {WORKSPACE_OUTPUT: workspace, SUMMARY: summary[0],
                   ACTION_ITEMS: summary[1], TOP_QUOTES: summary[2]}

        # Create a new DataFrame from the dictionary
        df_row = pd.DataFrame([new_row])

        # Concatenate the new DataFrame to the existing DataFrame
        df_summaries = pd.concat([df_summaries, df_row], ignore_index=True)

    return df_summaries


def consolidate_summaries(workspace, df_summaries):
    """
    This function consolidates the summaries for a given workspace
    """
    print(f"Consolidating summaries for workspace: {workspace}")

    num_rows = df_summaries.shape[0]

    while num_rows > 1:
        df_summaries = summarize_workspace(workspace, df_summaries, True)
        num_rows = df_summaries.shape[0]

    print(f"Finished consolidating summaries for workspace: {workspace}")
    return df_summaries


def process_df(d_f, min_text_length,
               process_raw_feedback,
               index,
               workspaces):
    """
    This function processes the dataframe to get the thoughtfulness classification and summary
    """

    print(
        f"Processing shard #{index}. process_raw_feedback={process_raw_feedback}")
    df_summaries = pd.DataFrame(columns=summary_columns)
    df_full = pd.DataFrame(columns=full_columns)

    for workspace in workspaces:
        print(f"     Processing workspace {workspace} for shard #{index}")
        workspace_df = d_f[(d_f[WORKSPACE_INPUT] == workspace) &
                           (d_f[VERBATIM_INPUT].str.len() > min_text_length)].copy()

        # Use the reindex method to reorder the columns
        workspace_df = workspace_df.reindex(columns=full_columns)

        if process_raw_feedback:
            workspace_df = process_raw_feedback_df(workspace_df)
            if workspace_df is not None:
                # Concatenate the workspace dataframe to the full data dataframe
                df_full = pd.concat([df_full, workspace_df], ignore_index=True)

        # Summarize the workspace
        df_summary = summarize_workspace(workspace, workspace_df, False)

        # Concatenate the summary dataframe to the summaries dataframe
        df_summaries = pd.concat([df_summaries, df_summary], ignore_index=True)

    print(f"Finished processing shard #{index}")

    return df_full, df_summaries


def get_nps_surveys(filename, workspaces_filter, min_days_ago, max_days_ago):
    """
    This function reads the NPS survey data from a CSV file and returns a dataframe
    filtered by workspace name and date range
    """
    # Calculate the date 30 days ago
    today = datetime.today().replace(hour=0, minute=0, second=0,
                                     microsecond=0, tzinfo=timezone.utc)
    start_days_ago = today - timedelta(days=max_days_ago)
    end_days_ago = today - timedelta(days=min_days_ago)

    # Read the CSV file into a dataframe
    _df = pd.read_csv(filename, usecols=survey_cols)

    # Replace NaN, None, and numpy.nan values with an empty string
    _df.fillna("", inplace=True)

    # Convert the timestamp column to datetime objects in UTC timezone
    _df[TIMESTAMP] = pd.to_datetime(_df[TIMESTAMP], utc=True)

    # Filter the dataframe based on workspaceName and timestamp
    filtered_df = _df[(_df[WORKSPACE_INPUT].isin(workspaces_filter)) &
                     (pd.to_datetime(_df[TIMESTAMP]) >= start_days_ago) &
                     (pd.to_datetime(_df[TIMESTAMP]) <= end_days_ago)]

    return filtered_df


def main(input_filename,
         raw_output_filename,
         summaries_interim_output_filename,
         summaries_output_filename,
         workspaces_filter,
         shard_size,
         min_text_length,
         process_raw_feedback,
         min_days_ago,
         max_days_ago):
    """
    This function processes the CSV file and outputs the thoughtfulness scores and summaries
    """

    # Load the CSV file into a dataframe per filter
    _df = get_nps_surveys(input_filename, workspaces_filter,
                          min_days_ago,max_days_ago)

    # group the rows by the specified column
    grouped_df = _df.groupby(WORKSPACE_INPUT)
    df_list = []

    # iterate over the groups and create chunks of the specified size
    # However, the chunks will be created based on the workspace column
    for name, group in grouped_df:  # pylint: disable=unused-variable
        # split the group into smaller chunks based on the chunk size
        group_chunks = [group[i:i+shard_size]
                        for i in range(0, len(group), shard_size)]
        # iterate over the chunks within the group
        for chunk in group_chunks:
            # Replace NaN values with empty strings in the DataFrame
            chunk.fillna('', inplace=True)
            # add a unique ID column to the input dataframe
            chunk[ID] = chunk.reset_index().index
            df_list.append(chunk)

    # Process the shards of dataframes in parallel
    with Pool(cpu_count()) as _p:
        results = _p.starmap(
            process_df, [(df, min_text_length, process_raw_feedback,
                         idx, workspaces_filter)
                         for idx, df in enumerate(df_list)])

    # results = [process_df(df, min_text_length, process_raw_feedback, idx)
    #            for idx, df in enumerate(df_list)]

    df_full_list = [result[0] for result in results]

    if process_raw_feedback:
        # Concatenate the df_thoughtfulness dataframes
        df_full = pd.concat(df_full_list)

        # Output the df_thoughtfulness_scores_full into a CSV file
        df_full.to_csv(raw_output_filename, index=False)

    df_summaries_list = [result[1] for result in results]

    # Concatenate the df_summaries into a single df_summaries dataframe
    df_summaries = pd.concat(df_summaries_list)

    print("Saving interim results")
    # Output the df_summaries dataframe into a CSV file
    df_summaries.to_csv(summaries_interim_output_filename,
                        index=False)

    # df_summaries = pd.read_csv(summaries_interim_output_filename)

    print("Consolidating summaries...")
    df_summaries_full = pd.DataFrame(columns=summary_columns)

    workspace_df_list = [(workspace,
                        df_summaries[df_summaries[WORKSPACE_OUTPUT
                        ] == workspace].copy())
                        for workspace in workspaces_filter]

    # Process the summaries in parallel by workspace
    with Pool(cpu_count()) as _p:
        summary_results = _p.starmap(
            consolidate_summaries, [(workspace, df) for workspace, df in workspace_df_list])

    # summary_results = [consolidate_summaries(workspace, df)
    #  for workspace, df in workspace_df_list]

    df_summaries_full_list = [result for result in summary_results]

    df_summaries_full = pd.concat(df_summaries_full_list)

    print("Saving summaries to csv...")

    # Output the df_summaries dataframe into a CSV file
    df_summaries_full.to_csv(summaries_output_filename, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--shard-size',
                        type=int, default=DEFAULT_SHARD_SIZE,
                        help='Shard size for parallel processing')
    parser.add_argument('-m', '--min-length', type=int,
                         default=DEFAULT_MIN_TEXT_LENGTH,
                        help='Min length of feedback text required to considered for analysis')
    parser.add_argument('--min-days-ago', type=int,
                         default=0,
                        help='Number of days ago to start processing (newest)')
    parser.add_argument('--max-days-ago', type=int,
                         default=30,
                        help='Number of days ago to end processing (oldest)')
    parser.add_argument('-r', '--process-raw-feedback', action='store_true', default=False,
                        help='Process the raw data over and above summarizing')
    args = parser.parse_args()

    # Start the timer
    start_time = time.time()


    # Process the Partner Center surveys data and generate two CSV files
    # raw scores, and summaries
    print("Processing Partner Center surveys...")
    main(PC_SURVEYS_INPUT_FILE,
        PC_RAW_OUTPUT_FILE,
        PC_SUMMARIES_INTERIM_OUTPUT_FILE,
        PC_SUMMARIES_OUTPUT_FILE,
        PC_WORKSPACES,
        args.shard_size,
        args.min_length,
        args.process_raw_feedback,
        args.min_days_ago,
        args.max_days_ago)

    # Process the MSX surveys data and generate two CSV files
    # raw scores, and summaries
    print("Processing MSX surveys...")
    main(MSX_SURVEYS_INPUT_FILE,
        MSX_RAW_OUTPUT_FILE,
        MSX_SUMMARIES_INTERIM_OUTPUT_FILE,
        MSX_SUMMARIES_OUTPUT_FILE,
        MSX_WORKSPACES,
        args.shard_size,
        args.min_length,
        args.process_raw_feedback,
        args.min_days_ago,
        args.max_days_ago)

    print("Creating summaries.docx...")
    converter = CSVtoDOCX(summary_dtypes, summary_columns,
                          WORKSPACE_OUTPUT)

    # Generate docx files from the CSV summary file for Partner Center
    print("Generating docx files for PC personas...")
    converter.convert(PC_SUMMARIES_OUTPUT_FILE,
                      CSP_SUMMARIES_DOCX_OUTPUT_FILE,
                      CSP_DOCX_TITLE,
                      CSP_WORKSPACES)
    converter.convert(PC_SUMMARIES_OUTPUT_FILE,
                      ISV_SUMMARIES_DOCX_OUTPUT_FILE,
                      ISV_DOCX_TITLE,
                      ISV_WORKSPACES)
    converter.convert(PC_SUMMARIES_OUTPUT_FILE,
                      MCPP_SUMMARIES_DOCX_OUTPUT_FILE,
                      MCPP_DOCX_TITLE,
                      MCPP_WORKSPACES)

    # Generate docx files from the CSV summary file for MSX
    print("Generating docx file for MSX personas...")
    converter.convert(MSX_SUMMARIES_OUTPUT_FILE,
                      MSX_SUMMARIES_DOCX_OUTPUT_FILE,
                      MSX_DOCX_TITLE,
                      MSX_WORKSPACES)

    end_time = time.time()

    total_time = end_time - start_time
    m, s = divmod(total_time, 60)
    print(f"\n\nCompleted! - Total time taken: {m:.0f} minutes and {s:.2f} seconds")
