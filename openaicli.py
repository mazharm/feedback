"""
This module is used to interact with the OpenAI API
"""
import os
import json
import traceback
import time
import logging
import re
import openai
from openai import Completion
from better_profanity import profanity
from prompt import Prompt, PromptType

p = Prompt()

class OpenAICli:
    """
    This class is used to interact with the OpenAI API
    """
    temperature = 0.7
    top_p = 0.95
    frequency_penalty = 0.0
    presence_penalty = 0.0
    engine = "gpt-35-turbo"
    test_mode = False
    max_tokens = 1024

    def __init__(self):
        openai.api_type = "azure"
        openai.api_version = "2022-12-01"
        openai.api_base = os.getenv("OPENAI_API_BASE")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        return

    def filter_text(self, text):
        """
        scrub text of bad words. It is causing openai to fail
        this code has to become more sophisticated
        """
        text = text.replace('"', ' $$ ')

        # Censor the sentence
        text = profanity.censor(text, '*')

        # Replace the placeholder character with quotes
        text = text.replace(' $$ ', '"')

        text = re.sub(r'http\S+', '', text)

        return text

    def get_prompt(self, messages):
        """
        Create the prompt
        """
        prompt_str = ""
        for item in messages:
            prompt_str += f"<|im_start|>{item['role']}\n{item['content']}\n<|im_end|>\n"

        prompt_str = self.filter_text(prompt_str)

        return prompt_str

    def get_completion(self, text):
        """
        Call the OAI API
        """

        # print(f"Prompt:\n{text}\n\n")
        response = Completion.create(
            engine=self.engine,
            prompt=text,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            max_tokens=self.max_tokens,
            stop=["<|im_end|>"])
        # print (f"Reply:\n{response['choices'][0]['text'].strip()}")
        # s = input("Press enter to continue...")

        return response['choices'][0]['text'].strip()

    def strip_extra_text(self, text):
        """
        remove extra text and only keep stuff inside the enclosing 
        square brackets
        """
        left_bracket = text.find("[")
        right_bracket = text.rfind("]")
        if left_bracket >= 0 and right_bracket >= 0:
            return text[left_bracket:right_bracket+1]
        else:
            return "[]"

    def get_analysis_helper(self, tuples):
        """
        Helper function to enable retries to the model
        The service tends to rate limit us and we need retry logic
        """

        text = json.dumps(tuples)
        messages = p.get_prompt(text, PromptType.BATCH_ANALYZE)

        for i in range(10):
            try:
                prompt = self.get_prompt(messages)
                o_response = self.get_completion(prompt)
                response = self.strip_extra_text(o_response)

                try:
                    json_response = json.loads(response)
                except Exception as _e:  # pylint: disable=broad-except
                    logging.error(
                        "get_analysis -- Error: %s\nprompt:\n%s\nresponse:\n%s\n",
                          _e, prompt, response)
                    logging.error("get_analysis -- %s\n",
                                  traceback.format_exc())
                    break

                return json_response

            except Exception as _e:  # pylint: disable=broad-except
                # most likely being rate limited gotta be patient
                time.sleep(5 + 5 * i)

        return None

    def get_prompt_from_tuples(self, tuples):
        """
        Flatted the list of messages into a string
        then construct the prompt
        """
        text = json.dumps(tuples)
        messages = p.get_prompt(text, PromptType.BATCH_ANALYZE)
        prompt = self.get_prompt(messages)

        return prompt

    def get_analysis(self, tuples):
        """
        This the main function to send the chat context to the GPT model and get a response.
        This function will try to analyze the messages in a batch. If it fails because
        the model is hallucinating, it will fall back to processing one message at a time.
        """
        # try to analyze all the messages as a batch using get_analysis_helper. If that fails
        # process the messages one a time and combine the results
        json_response = self.get_analysis_helper(tuples)
        if (json_response is None or len(json_response) == 0):
            json_response = []
            for _tuple in tuples:
                new_json_response = self.get_analysis_helper([_tuple])
                if new_json_response is None:
                    prompt = self.get_prompt_from_tuples([_tuple])
                    print(f"get_analysis: call failed :\n{prompt}\n")
                else:
                    json_response += new_json_response

        if (json_response is None or len(json_response) == 0):
            prompt = self.get_prompt_from_tuples(tuples)
            print(f"get_analysis: batch failed:\n {prompt}\n")
            return []

        return json_response

    def get_summary(self, tuples, summary_type):
        """
        This the main function to send the chat context to the GPT model and get a response.
        """
        text = json.dumps(tuples)
        messages = p.get_prompt(text, summary_type)
        prompt = ""
        response = ""

        for i in range(5):
            try:
                prompt = self.get_prompt(messages)
                response = self.get_completion(prompt)
                return response
            except Exception as _e:  # pylint: disable=broad-except
                logging.error(
                    "get_summary -- retry#{i}- Error: %s, prompt: %s", _e, prompt)
                logging.error("get_summary -- %s", traceback.format_exc())
                # most likely being rate limited gotta be patient
                time.sleep(5 * i)

        print("get_summary -- Giving up after 5 retries")

        return ""
