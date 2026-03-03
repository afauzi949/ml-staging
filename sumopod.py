from openai import OpenAI

from retrieval import OPENAI_API_KEY

# Initialize client with SumoPod AI
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://ai.sumopod.com/v1"
)

# Make a chat completion request
response = client.chat.completions.create(
    model="seed-2-0-mini-free",
    messages=[
        {"role": "user", "content": "Say hello in a creative way"}
    ],
    max_tokens=150,
    temperature=0.7
)

print(response.choices[0].message.content)