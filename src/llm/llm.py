from openai import OpenAI
from ollama import Client
import yaml

class LLM:
    def __init__(self, api_key: str, base_url: str, configfile: str, model: str | None):
        if model is None or "gpt" in model:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        elif "qwen" in model:
            self.client = Client()
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
    
    def _query_gpt(self, messages):
        response = self.client.chat.completions.create(
            model="gpt-oss-120b", temperature=0, messages=messages
        )
        reasoning  = response.choices[0].message.reasoning
        content = response.choices[0].message.content
        return reasoning, content

    def _query_qwen(self, messages):
        response = self.client.chat(
            model = self.model,
            messages = messages, 
            think = False,
            options = {"temperature": 0}
        )
        content = response["message"]["content"]
        return None, content

    def query(self, messages):
        if hasattr(self, "model") and "qwen" in self.model:
            return self._query_qwen(messages)
        else:
            return self._query_gpt(messages)
