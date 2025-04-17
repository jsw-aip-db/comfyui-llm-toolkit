import requests
import os

# List of available OpenAI models for the node dropdown
openai_model_ids = [
    "gpt-4.1",
    # Add more model names here as needed
]

class GenerateText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Write a one-sentence bedtime story about a unicorn."}),
                "model": (openai_model_ids,),
            }
        }

    RETURN_TYPES = ("STRING",)

    def run(self, prompt, model):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return ("Error: OPENAI_API_KEY environment variable not set.",)
        url = "https://api.openai.com/v1/responses"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        data = {
            "model": model,
            "input": prompt,
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            # The actual response text may be in a different field depending on OpenAI's API response structure
            # Here, we try to extract a likely field
            if isinstance(result, dict):
                # Try common fields
                for key in ["output", "text", "response", "choices"]:
                    if key in result:
                        if isinstance(result[key], list) and result[key]:
                            return (result[key][0].get("text", str(result[key][0])),)
                        return (str(result[key]),)
                return (str(result),)
            return (str(result),)
        except Exception as e:
            return (f"Error: {e}",)
