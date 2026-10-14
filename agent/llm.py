import json
from openai import OpenAI
from agent.config import API_KEY, MODEL


class DeepSeekLLM:
    def __init__(self):
        self.client = OpenAI(
            api_key=API_KEY,
            base_url="https://api.deepseek.com"
        )

    def ask(self, prompt, data):
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(data, ensure_ascii=False)
            }
        ]

        for attempt in range(2):
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=12000,
                extra_body={"thinking": {"type": "disabled"}}
            )

            choice = response.choices[0]
            content = (choice.message.content or "").strip()

            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass

            print(
                f"第 {attempt + 1} 次生成失败，"
                f"结束原因：{choice.finish_reason}"
            )

        raise RuntimeError("DeepSeek 连续两次未返回有效 JSON")
