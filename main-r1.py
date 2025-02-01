import groq
import time
import os
import json
from dotenv import load_dotenv
load_dotenv()

# https://console.groq.com/docs/text-chat
from groq import Groq

def askGroq(message="解释快速语言模型的重要性"):
    client = Groq()
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "you are a helpful assistant."
            },
            {
                "role": "user",
                "content": message,
            }
        ],
        model="deepseek-r1-distill-llama-70b",
    )

    print(chat_completion.choices[0].message.content)



def main():
    askGroq("解释快速语言模型的重要性")

if __name__ == "__main__":
    main()
