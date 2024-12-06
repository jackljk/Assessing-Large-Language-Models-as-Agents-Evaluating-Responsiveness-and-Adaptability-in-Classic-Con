import openai
import os
import time

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_action(primer, client):
    print("Getting action from LLM")
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": primer}
    ],
    response_format={
      "type": "text"
    },
    temperature=0.2,
    max_tokens=200,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    action_text = response.choices[0].message.content
    print("Action received from LLM")
    print(action_text)
    print("------")
    return action_text