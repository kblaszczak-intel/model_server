# How to serve LLM models with Continous Batching via OpenAI API

## Model preparation

## Server configuration

## Start-up

## Client code

Both unary and streaming calls should be available via the same servable:

### Unary:
```bash
curl http://localhost:8000/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What is OpenVINO?"
      },
      {
        "role": "assistant",
        "content": "OpenVINO is an open-source software library for deep learning inference that is designed to optimize and run deep learning models on Intel hardware."
      },
      {
        "role": "user",
        "content": "How to install?"
      }
    ]
  }'
```

Output:
```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "Open a command prompt terminal window. You can use the keyboard shortcut: Ctrl+Alt+T\nCreate the /opt/intel folder for OpenVINO by using the following command. If the folder already exists, skip this step.",
        "role": "assistant"
      },
      "logprobs": null
    }
  ],
  "created": 1677664795,
  "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 17,
    "prompt_tokens": 57,
    "total_tokens": 74
  }
}
```

## Streaming:

Partial outputs:
```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI()

stream = client.chat.completions.create(
    base_url="http://localhost:8000/v3",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

Output:
```
This is a test.
```

Refer to [OpenAI streaming documentation](https://platform.openai.com/docs/api-reference/streaming).
