import groq
import time
import os
import json
from dotenv import load_dotenv
load_dotenv()


class GroqAPIClient:
    def __init__(self, custom_client=None):
        self.client = custom_client or groq.Groq()

    def make_api_call(self, messages, max_tokens, is_final_answer=False):
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.2,
                    response_format={"type": "json_object"} if not is_final_answer else None
                )
                if is_final_answer:
                    return response.choices[0].message.content
                else:
                    return json.loads(response.choices[0].message.content)
            except Exception as e:
                if attempt == 2:
                    error_content = {
                        "title": "Error",
                        "content": f"Failed to generate {'final answer' if is_final_answer else 'step'} after 3 attempts. Error: {str(e)}"
                    }
                    if not is_final_answer:
                        error_content["next_action"] = "final_answer"
                    return error_content
                time.sleep(1)  # Wait for 1 second before retrying


class ReasoningAssistant:
    def __init__(self, api_client):
        self.api_client = api_client

    def generate_response(self, prompt):
        messages = self._initialize_messages(prompt)
        steps = []
        step_count = 1
        total_thinking_time = 0

        while True:
            start_time = time.time()
            step_data = self.api_client.make_api_call(messages, 300)
            end_time = time.time()
            thinking_time = end_time - start_time
            total_thinking_time += thinking_time

            steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time))
            messages.append({"role": "assistant", "content": json.dumps(step_data)})

            if step_data['next_action'] == 'final_answer' or step_count > 25:  # Prevent infinite loop
                break

            step_count += 1

            # Yield after each step for real-time feedback
            yield steps, None

        # Generate final answer
        final_answer = self._generate_final_answer(messages)
        steps.append(("Final Answer", final_answer['content'], final_answer['thinking_time']))
        yield steps, total_thinking_time + final_answer['thinking_time']

        # Save steps and final answer to markdown file
        self._save_to_markdown(prompt, steps)

    def _initialize_messages(self, prompt):
        return [
            {"role": "system", "content": self._system_message()},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
        ]

    def _system_message(self):
        return (
            """
            You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.
            """
        )

    def _generate_final_answer(self, messages):
        messages.append({"role": "user", "content": "Please provide the final answer based solely on your reasoning above. Do not use JSON formatting. Only provide the text response without any titles or preambles. Retain any formatting as instructed by the original prompt, such as exact formatting for free response or multiple choice."})
        start_time = time.time()
        final_data = self.api_client.make_api_call(messages, 1200, is_final_answer=True)
        end_time = time.time()
        thinking_time = end_time - start_time
        return {"content": final_data, "thinking_time": thinking_time}

    def _save_to_markdown(self, prompt, steps):
        if len(prompt) > 20:
            filename = f"{prompt[:20]}.md".replace(" ", "_").replace("/", "-")
        else:
            filename = f"{prompt}.md".replace(" ", "_").replace("/", "-")
        with open(filename, "w", encoding="utf-8") as file:
            file.write(f"# Problem: {prompt}\n\n")
            for step in steps:
                file.write(f"## {step[0]}\n\n{step[1]}\n\n(Thinking time: {step[2]:.2f} seconds)\n\n")


def main():
    custom_client = None  # Replace with a valid client instance if necessary
    api_client = GroqAPIClient(custom_client)
    assistant = ReasoningAssistant(api_client)
    # prompt = "Describe the process for baking a cake."
    # prompt = "When killing Werewolf, I am the real prophet. If someone jumps to the prophet first and kills me, what is my best response strategy?"
    # prompt = "I am playing Werewolf and I am the Hunter. It is the second round and the Seer is dead. Someone who spoke before me claimed to be the Hunter (but is not actually the Hunter). How should I respond when it is my turn to speak?"
    prompt = "Given the parabola \( y^2 = 16x \), the coordinates of the focus are ________."
    for steps, total_time in assistant.generate_response(prompt):
        for step in steps:
            print(f"{step[0]}:\n{step[1]}\n(Thinking time: {step[2]:.2f} seconds)\n")
        if total_time is not None:
            print(f"Total Thinking Time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
