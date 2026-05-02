import requests
from .interfaces import BaseLLM
from .config import LLMConfig


class OllamaLLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        self._model = config.model_name
        self._base_url = config.ollama_base_url
        self._temperature = config.temperature
        self._api_url = f"{self._base_url}/api/generate"

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": 1024,
            }
        }
        try:
            response = requests.post(self._api_url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._base_url}.\n"
                "Make sure Ollama is running: ollama serve"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError("Ollama timed out. Try a smaller model.")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def is_available(self) -> bool:
        try:
            return requests.get(f"{self._base_url}/api/tags", timeout=5).status_code == 200
        except Exception:
            return False

    def list_models(self) -> list:
        try:
            data = requests.get(f"{self._base_url}/api/tags", timeout=5).json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []


class GeminiLLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        self._model = config.model_name
        self._temperature = config.temperature
        self._api_key = config.gemini_api_key

    def generate(self, prompt: str) -> str:
        if not self._api_key:
            raise ValueError(
                "Gemini API key not set.\n"
                "Get a free key at: https://aistudio.google.com/app/apikey\n"
                "Then run: export GEMINI_API_KEY=your_key_here"
            )
        try:
            from google import genai
            client = genai.Client(api_key=self._api_key)
            response = client.models.generate_content(
                model=self._model,
                contents=prompt,
            )
            return response.text.strip()
        except ImportError:
            raise ImportError(
                "google-genai package not installed.\n"
                "Run: pip install google-genai"
            )
        except Exception as e:
            raise RuntimeError(f"Gemini error: {e}")

    def is_available(self) -> bool:
        return bool(self._api_key)


def create_llm(config: LLMConfig) -> BaseLLM:
    if config.provider == "gemini":
        return GeminiLLM(config)
    elif config.provider == "ollama":
        return OllamaLLM(config)
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider}. Use 'gemini' or 'ollama'.")
