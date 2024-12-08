import openai
import os
import time

import logging

# A logger for this file
log = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_action(primer, client):
    log.info(f"Getting action from LLM")
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
    log.info(f"Response from LLM: \n {action_text}")
    return action_text