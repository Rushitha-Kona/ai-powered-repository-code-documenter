import json
import os
import requests


class LLMError(RuntimeError):
    pass


class OllamaLLM:
    def __init__(self, model='qwen2.5-coder:3b', base_url=None, num_ctx=None, num_predict=1500):
        self.model = model
        self.base_url = (base_url or os.getenv('OLLAMA_HOST') or 'http://localhost:11434').rstrip('/')
        if not self.base_url.startswith('http'):
            self.base_url = 'http://' + self.base_url
        # Ollama's default context is small (2048-4096) and silently truncates longer prompts,
        # which is the usual cause of empty / nonsensical documentation. Ask for more.
        self.num_ctx = int(num_ctx or os.getenv('OLLAMA_NUM_CTX', 8192))
        self.num_predict = num_predict

    def generate(self, prompt: str) -> str:
        payload = {
            'model': self.model,
            'prompt': prompt,
            'stream': True,
            'options': {'temperature': 0.2, 'num_ctx': self.num_ctx, 'num_predict': self.num_predict},
        }
        try:
            # (connect timeout, read timeout between streamed chunks). Streaming means a slow
            # CPU model no longer hits a hard 180 s wall for the whole answer.
            with requests.post(f'{self.base_url}/api/generate', json=payload, stream=True,
                               timeout=(10, 300)) as response:
                if response.status_code != 200:
                    detail = response.text[:500]
                    if response.status_code == 404:
                        raise LLMError(f"Model '{self.model}' not found in Ollama. Run: ollama pull {self.model}")
                    raise LLMError(f'Ollama returned HTTP {response.status_code}: {detail}')
                parts = []
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if 'error' in chunk:
                        raise LLMError(f"Ollama error: {chunk['error']}")
                    parts.append(chunk.get('response', ''))
                    if chunk.get('done'):
                        break
        except requests.exceptions.ConnectionError:
            raise LLMError(f'Cannot reach Ollama at {self.base_url}. Start it with `ollama serve`.')
        except requests.exceptions.Timeout:
            raise LLMError('Ollama timed out. The model is too slow for this prompt; try a smaller model.')
        return ''.join(parts).strip()


class MockLLM:
    def generate(self, prompt: str) -> str:
        return ('**Demo mode - no LLM was called.** The prompt that would be sent to the model:\n\n'
                '```text\n' + prompt[:6000] + '\n```')
