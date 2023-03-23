"""
Tool to process NPS feedback
"""
import argparse
import logging
from multiprocessing import Pool, cpu_count
import pandas as pd
from openaicli import OpenAICli
from prompt import PromptType
from csvtodoc import CSVtoDOCX

# Set up logging
logging.basicConfig(filename='error.log', level=logging.ERROR)

oac = OpenAICli()

full_columns = ['ID', 'Workspace', 'Score', 'LHWeightedScore',
                'Verbatim', 'SurveyType', 'classification', 'topics']

summary_columns = ['Workspace', 'Summary', 'Top Quotes', 'Action Items']
score_columns = ['ID', 'classification', 'topics']

dtypes = {'Workspace': str, 'Score': float,
          'LHWeightedScore': float, 'Verbatim': str, 'SurveyType': str}
summary_dtypes = {'Workspace': str, 'Summary': str,
                  'Top Quotes': str, 'Action Items': str}

"""
workspaces = ['Accounts']
"""


workspaces = ['Accounts',
              'Action Center',
              'Apis and Integration',
              'Apps & Games',
              'Benefits',
              'Billing',
              'Collaborate',
              'Customer',
              'Edge',
              'Enrollment',
              'Hardware',
              'Help + Support',
              'Incentives',
              'Insights',
              'Internal',
              'Marketplace Offers',
              'Membership',
              'Payouts',
              'Pricing',
              'Referrals',
              ]


# Define the thoughtfulness score ranges and corresponding labels
thoughtfulness_classfications = ['Thoughtful', 'Not Thoughtful']


def get_score(tuples):
    """
    This function returns the thoughtfulness score for the given text
    """
    d_f = pd.DataFrame(columns=['ID', 'classification', 'topics'])

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
    chunks = [(feedback_df[['ID', 'Verbatim']][i:i+chunk_size]).to_numpy()
              for i in range(0, len(feedback_df), chunk_size)]
    score_chunks = [get_score([tuple(x) for x in chunk])
                    for chunk in chunks]

    if len(score_chunks) == 0:
        return None

    scores_df = pd.concat(score_chunks)

    if len(scores_df) > 0:

        # Add blank columns to the dataframe
        feedback_df['classification'] = pd.Series(dtype='str')
        feedback_df['topics'] = pd.Series(dtype='str')

        # Set "ID" as the index for both dataframes
        feedback_df.set_index('ID', inplace=True)
        scores_df.set_index('ID', inplace=True)

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
    summaries_full = [list(zip(d_f['Summary'][i:i+chunk_size],
                               d_f['Top Quotes'][i:i+chunk_size],
                               d_f['Action Items'][i:i+chunk_size]))
                      for i in range(0, len(d_f), chunk_size)]
    print(f"summaries_full: {len(summaries_full)}")

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
    summaries_full = [list(d_f['Verbatim'][i:i+chunk_size], )
                      for i in range(0, len(d_f), chunk_size)]
    print(f"summaries_full: {len(summaries_full)}")

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
        new_row = {'Workspace': workspace, 'Summary': summary[0],
                   'Action Items': summary[1], 'Top Quotes': summary[2]}

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
        # Summarize the filtered dataframe
        df_summaries = summarize_workspace(workspace, df_summaries, True)
        num_rows = df_summaries.shape[0]

    print(f"Finished consolidating summaries for workspace: {workspace}")
    return df_summaries


def process_df(d_f, min_text_length, process_raw_feedback, index):
    """
    This function processes the dataframe to get the thoughtfulness classification and summary
    """

    print(
        f"Processing shard #{index}. process_raw_feedback={process_raw_feedback}")
    df_summaries = pd.DataFrame(columns=summary_columns)
    df_full = pd.DataFrame(columns=full_columns)

    for workspace in workspaces:
        print(f"     Processing workspace {workspace} for shard #{index}")
        workspace_df = d_f[(d_f['Workspace'] == workspace) &
                           (d_f['Verbatim'].str.len() > min_text_length)].copy()

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


def main(filename, shard_size, min_text_length, process_raw_feedback):
    """
    This function processes the CSV file and outputs the thoughtfulness scores and summaries
    """
    # Load the CSV file into shards of dataframes
    df_list = []

    # for chunk in pd.read_csv(filename, header=0, dtype=dtypes, chunksize=shard_size, nrows=1000):
    for chunk in pd.read_csv(filename, header=0, dtype=dtypes, chunksize=shard_size):
        # Replace NaN values with empty strings in the DataFrame
        chunk.fillna('', inplace=True)
        # add a unique ID column to the input dataframe
        chunk['ID'] = chunk.reset_index().index
        df_list.append(chunk)

    # Process the shards of dataframes in parallel
    with Pool(cpu_count()) as _p:
        results = _p.starmap(
            process_df, [(df, min_text_length, process_raw_feedback, idx)
                         for idx, df in enumerate(df_list)])

    # results = [process_df(df, min_text_length, process_raw_feedback, idx)
    #            for idx, df in enumerate(df_list)]

    df_full_list = [result[0] for result in results]

    if process_raw_feedback:
        # Concatenate the df_thoughtfulness dataframes
        df_full = pd.concat(df_full_list)

        # Output the df_thoughtfulness_scores_full into a CSV file
        df_full.to_csv('thoughtfulness_topics.csv', index=False)

    df_summaries_list = [result[1] for result in results]

    # Concatenate the df_summaries into a single df_summaries dataframe
    df_summaries = pd.concat(df_summaries_list)

    print("Saving interim results")
    # Output the df_summaries dataframe into a CSV file
    df_summaries.to_csv('summaries_interim.csv', index=False)

    print("Consolidating summaries...")
    df_summaries_full = pd.DataFrame(columns=summary_columns)

    workspace_df_list = [(workspace, df_summaries[df_summaries['Workspace'] == workspace].copy())
                         for workspace in workspaces]

    # Process the shards of dataframes in parallel
    with Pool(cpu_count()) as _p:
        summary_results = _p.starmap(
            consolidate_summaries, [(workspace, df) for workspace, df in workspace_df_list])

    # summary_results = [consolidate_summaries(workspace, df_summaries) for workspace in workspaces]

    df_summaries_full_list = [
        summary_result for summary_result in summary_results]

    df_summaries_full = pd.concat(df_summaries_full_list)

    print("Saving summaries to csv...")

    # Output the df_summaries dataframe into a CSV file
    df_summaries_full.to_csv('summaries.csv', index=False)

    print("Creating summaries.docx...")

    converter = CSVtoDOCX()

    converter.convert(summary_dtypes, 'summaries.csv', 'summaries.docx',
                      'NPS Feedback Summary', 'Workspace')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--shard-size', type=int, default=500,
                        help='Shard size for parallel processing')
    parser.add_argument('-m', '--min-length', type=int, default=5,
                        help='Min length of feedback text required to considered for analysis')
    parser.add_argument('-f', '--input-file', type=str, default='feedback.csv',
                        help='Specify the file that contains the feedback')
    parser.add_argument('-r', '--process-raw-feedback', action='store_true', default=False,
                        help='Process the raw data over and above summarizing')
    args = parser.parse_args()

    main(args.input_file, args.shard_size,
         args.min_length, args.process_raw_feedback)
