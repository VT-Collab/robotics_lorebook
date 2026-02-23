from openai import OpenAI
from ollama import Client, chat
# import warnings
# import os

# warnings.filterwarnings("ignore", category=FutureWarning, module="google.*")
# os.environ["GRPC_PYTHON_LOG_LEVEL"] = "0"
# import google.generativeai as genai
import yaml
import base64
import io
from PIL import Image


class LLM:
    def __init__(self, api_key: str, base_url: str, configfile: str, model: str = None):
        if model is None:
            model = "gpt-oss-120b"
        if "gpt" in model.lower():
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        elif "gemini" in model:
            # genai.configure(api_key=api_key)
            # self.client = genai.GenerativeModel(model_name=model)
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        # elif "qwen" in model:
        #     self.client = Client()
        self.model = model
        with open(configfile, "r") as fh:
            self.prompt_template = yaml.safe_load(fh)

    def generate_system_prompt(self) -> str:
        system_prompt = self.prompt_template["PROMPT_SYSTEM"]
        return system_prompt

    def generate_initial_prompt(self, *args, **kwargs) -> str:
        initial_prompt = self.prompt_template["PROMPT_INITIAL"]
        for k, v in kwargs.items():
            key = f"${{INITIAL.{k}}}"
            initial_prompt = initial_prompt.replace(key, str(v))
        return initial_prompt

    def generate_followup_prompt(self, *args, **kwargs) -> str:
        followup_prompt = self.prompt_template["PROMPT_FOLLOWUP"]
        for k, v in kwargs.items():
            key = f"${{FOLLOWUP.{k}}}"
            followup_prompt = followup_prompt.replace(key, str(v))
        return followup_prompt

    def generate_initial_message(self, *args, **kwargs):
        system_prompt = self.generate_system_prompt()
        initial_prompt = self.generate_initial_prompt(*args, **kwargs)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_prompt},
        ]
        return messages

    def _query_gpt(self, messages, lorebook_content=None):
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=messages,
            extra_body=lorebook_content,
        )
        message = response.choices[0].message
        reasoning = "None"
        if hasattr(message, "reasoning"):
            reasoning = message.reasoning
        content = message.content
        return reasoning, content

    def _query_qwen(self, messages):
        # response = self.client.chat(
        #     model=self.model, messages=messages, think=False, options={"temperature": 0}
        # )
        # content = response["message"]["content"]
        # return "None", content
        response = chat(
            model = self.model,
            messages = messages,
            options = {"temperature": 0}
        )
        return response["message"]["thinking"], response["message"]["content"]

    def _query_gemini(self, messages, image=None):
        # Extract system prompt from messages (Gemini uses a specific param for this)
        # system_msg = next(
            # (m["content"] for m in messages if m["role"] == "system"), None
        # )
        # user_content = []
# 
        # if image:
            # image_bytes = base64.b64decode(image)
            # img = Image.open(io.BytesIO(image_bytes))
            # user_content.append(img)
# 
        # user_input = messages[-1]["content"]
        # user_content.append(user_input)
# 
        # response = self.client.chat.completions.create(
        #     model=self.model,
        #     messages=messages,
        #     temperature=1.0,
        #     extra_body={"reasoning_effort": "medium"},
        # )

        # return "None", response.choices[0].message.content

        response = chat(
            model = self.model,
            messages = messages,
            options = {"temperature": 0}
        )
        return response["message"]["thinking"] if "thinking" in response["message"] else None, response["message"]["content"]

        # model = genai.GenerativeModel(
        #     model_name=self.model,
        #     system_instruction=system_msg,
        # )

        # generation_config = {
        #     "temperature": 1.0,  # CRITICAL: Temp < 1.0 causes latency spikes on Gemini 3
        # }

        # response = model.generate_content(
        #     user_content, generation_config=generation_config
        # )

        # return "None", response.text

    def query(self, messages, lorebook_content=None):
        if hasattr(self, "model") and "qwen" in self.model:
            return self._query_qwen(messages)
        elif hasattr(self, "model") and "gemini" in self.model:
            return self._query_gemini(messages)
        else:
            return self._query_gpt(messages, lorebook_content)
