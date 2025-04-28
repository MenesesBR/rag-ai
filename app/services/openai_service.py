from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


def ask_openai(system_prompt, prompt):
    # Faz a chamada correta à API de chat
    response = client.responses.create(
        model="gpt-4.1-nano-2025-04-14",
        instructions=system_prompt,
        input=prompt,
    )

    # Extrai e retorna a resposta do assistente
    return response.output_text
