"""
This module is used to format the prompt for the GPT model
"""


class PromptType:
    """
    This class contains the enums for the various prompt types
    """
    SUMMARY = [
        {"role": "system", "content": """Create a summary for the product team from the customer 
        feedback. Break down the summary by topic and have a section for each key topic.
        The summary must be useful for the team improving the product and written in
        style that is appropriate for the tech sector of corporate america.
        Here is a sample format for the output:
        Summary:
        This is a summary of the feedback.
        Key Topics:
        1. This is a key topic: This is a summary of the key topic.
        2. This is another key topic: This is a summary of the key topic.
        3. This is a third key topic: This is a summary of the key topic.
        """},
        {"role": "user", "content": "{content}"}
    ]
    ACTION_ITEMS = [
        {"role": "system", "content": """Generate a list of suggestions to improve the product 
        based on customer feedback.Provide no more than 10 actionable suggestions 
        that could address the most commonly mentioned issues or complaints in the feedback. 
        The suggestions should be specific, feasible, and focused on improving the overall user experience.
        Here is a sample format for the output:
        1. This is a suggestion
        2. This is another suggestion
        3. This is a third suggestion
          """},
        {"role": "user", "content": "{content}"}
    ]
    TOP_QUOTES = [
        {"role": "system", "content": """Extract key quotes from a large body of customer feedback 
        to represent the overall sentiment and opinions. Generate 5-10 quotes that
         best capture the main themes and emotions expressed in the feedback.
        Here is a sample format for the output:
        1. "This is a quote"
        2. "This is another quote"
        3. "This is a third quote"
        """},
        {"role": "user", "content": "{content}"}
    ]
    CONSOLIDATE_SUMMARY = [
        {"role": "system", "content": """The text contains a list of exec
          summaries and key topics. Create a summary that combines the summaries. And have a section
          for each key topic that merges the corresponding sections in the list of summaries. 
          The style of the summary must
          be appropriate for the tech sector of corporate america.
          Here is a sample format for the output:
          Summary:
          This is a summary of the feedback.
          Key Topics:
          1. This is a key topic: This is a summary of the key topic.
          2. This is another key topic: This is a summary of the key topic.
          3. This is a third key topic: This is a summary of the key topic.
          """},
        {"role": "user", "content": "{content}"}
    ]
    CONSOLIDATE_ACTION_ITEMS = [
        {"role": "system", "content": """The text contains multiple lists of suggestions to improve the product.
          Combine the lists into a single list of no more than 10 suggestions to improve the product.
          The suggestions should be specific, feasible, and focused on improving the overall user experience.
          Here is a sample format for the output:
          1. This is a suggestion
          2. This is another suggestion
          3. This is a third suggestion
          """},
        {"role": "user", "content": "{content}"}
    ]
    CONSOLIDATE_TOP_QUOTES = [
        {"role": "system", "content": """The text contains multiple lists of customer quotes. 
          Generate a consolidated list of up to 10 customer quotes from the feedback that best represent the 
          customer feedback and sentiment. 
          Here is a sample format for the output:
          1. "This is a quote"
          2. "This is another quote"
          3. "This is a third quote"
          """},
        {"role": "user", "content": "{content}"}
    ]
    BATCH_ANALYZE = [
        {"role": "system", "content": """You are a helpful assistant. The user will enter an array 
          of two tuples containing an ID and a string containing customer feedback. Here is the format:
          ["ID", "customer feedback"]
          The feedback might contain profanity. Analyze the thoughtfulness of each message and extract
          the key topics from the message. Classify the message into "Thoughtful" and "Not Thoughtful".
          If there is profanity, classify it as "Not Thoughtful".Respond in the following format for 
          each row. Your response must only have an array of well formatted JSON blobs that can be 
          parsed without requring any stripping of extraneous text. There should be one blob for 
	      each input item in the array and output no  other text outside of the array of JSON blobs.
          Each JSON blob in the array contains the following properties:
          ID (int), classification(string with "Thoughtful" or "Not Thoughtful"), 
          topics (a string that is a comma separate list of key topics)
          example: {{"ID":24, "classification":"Thoughtful", "topics":"topic1, topic2, topic3"}}"""},
        {"role": "user", "content": "{content}"}
    ]


class Prompt:
    """
    This class is used to format the prompt based on the provided arguments
    The format of the prompt is as follows:
    """

    def __init__(self):
        return

    def get_prompt(self, content, prompt_type):
        """
        This function returns the summary prompt
        """
        formatted_prompt = [
            {"role": blob["role"], "content": blob["content"].
             format(content=content)} for blob in prompt_type]
        return formatted_prompt
