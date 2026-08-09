from dotenv import load_dotenv
import os
from openai import OpenAI

# --- Completions API --- 


# API Q1

# Load environment variables from .env
load_dotenv()

# Create OpenAI client
client = OpenAI()

# Make API request
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "What is one thing that makes Python a good language for beginners?"
        }
    ],
)

# Print outputs with labels
print("Response:")
print(response.choices[0].message.content)

print("\nModel:")
print(response.model)

print("\nTotal Tokens:")
print(response.usage.total_tokens)


# API Q2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temp,
    )

    print(f"\nTemperature: {temp}")
    print(response.choices[0].message.content)

# In my experiment, the responses varied across runs even with the same
# temperature, including temperature=0. This shows that the model is not
# perfectly deterministic. In general, however, lower temperatures produce
# more focused and consistent responses, while higher temperatures encourage
# more creative and varied outputs.
#
# If I needed a consistent, reproducible output, I would use temperature=0,
# because it produces the most predictable responses.


# API Q3
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Give me a one-sentence fun fact about pandas (the animal, not the library).",
        }
    ],
    n=3,
    temperature=1.0,
)

for i, choice in enumerate(response.choices, start=1):
    print(f"\nCompletion {i}:")
    print(choice.message.content)


# API Q4

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain how neural networks work."
        }
    ],
    max_tokens=15,
)

print("\nResponse with max_tokens=15:")
print(response.choices[0].message.content)


# --- System Messages and Personas ---


# System Q1

# First personality: Python tutor
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)

print("\nPython Tutor Response:")
print(response.choices[0].message.content)


messages = [
    {"role": "system", "content": "You are a calm Buddhist monk. You speak gently, use simple wisdom and mindfulness in your explanations, and encourage patience and steady learning."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)

print("\nBuddhist Monk Response:")
print(response.choices[0].message.content)

# Both responses explained list comprehensions correctly and used similar
# examples, but the tone changed noticeably. The Python tutor was direct,
# structured, and encouraging, while the Buddhist monk used calmer,
# more reflective language, metaphors, and emphasized patience and mindfulness.


# System Q2

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)

print("\nResponse:")
print(response.choices[0].message.content)

# The model knows the user's name because the entire conversation history
# was included in the messages list.
# Although the API is stateless and does not remember 
# previous requests on its own, it can use any information 
# that is provided in the current request as context.


# --- Prompt Engineering ---


# Prompt Q1 — Zero-Shot

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow.",
]

prompt = f"""
Classify the sentiment of each review as positive, negative, or mixed.

Review 1: {reviews[0]}
Review 2: {reviews[1]}
Review 3: {reviews[2]}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
)

print("\nZero-Shot Sentiment Classifications:")
print(response.choices[0].message.content)


# Prompt Q2 — One-Shot

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow.",
]

prompt = f"""
Classify the sentiment of each review as positive, negative, or mixed.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Review 1: {reviews[0]}
Review 2: {reviews[1]}
Review 3: {reviews[2]}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
)

print("\nOne-Shot Sentiment Classification:")
print(response.choices[0].message.content)

# Yes. Adding one example changed the output format. In Q1, the model
# returned only the sentiment label (Positive, Negative, Mixed). In Q2,
# it followed the example and prefixed each label with "Sentiment:".
# The classifications themselves stayed the same, but the formatting
# became more consistent by matching the provided example.


# Prompt Q3 — Few-Shot

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow.",
]

prompt = f"""
Classify the sentiment of each review as positive, negative, or mixed.

Examples:

Review: "The product exceeded all my expectations."
Sentiment: positive

Review: "The app freezes every time I try to log in."
Sentiment: negative

Review: "The food was delicious, but the service was slow."
Sentiment: mixed

Review 1: {reviews[0]}
Review 2: {reviews[1]}
Review 3: {reviews[2]}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
)

print("\nFew-Shot Sentiment Classification:")
print(response.choices[0].message.content)

# Zero-shot correctly classified all three reviews with a simple output format.
# One-shot and few-shot produced the same classifications but followed the
# example format by adding "Sentiment:" before each label, making the output
# more consistent.
#
# I would use zero-shot for simple tasks with clear instructions. 
# I would use one-shot when I want the model to follow a specific output format. 
# I would use few-shot for more complex or ambiguous tasks 
# where multiple examples can help the model better understand 
# the expected pattern.


# Prompt Q4 — Chain of Thought

prompt = """
Solve the following problem. Show your reasoning step by step before giving
the final answer. Label the final answer clearly.

A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
)

print("\nChain of Thought Response:")
print(response.choices[0].message.content)

# Asking the model to reason step by step can improve accuracy because it
# breaks the problem into smaller calculations and makes it less likely that
# the model will skip a step or combine the operations incorrectly.


# Prompt Q5 — Structured Output

import json

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

prompt = f"""
Analyze the following review and return the result only as valid JSON.

The JSON must contain these keys:
- sentiment
- confidence (a float from 0 to 1)
- reason (one sentence)

Review:
"{review}"
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
)

raw_response = response.choices[0].message.content

print("\nRaw Response:")
print(raw_response)

try:
    result = json.loads(raw_response)

    print("\nParsed JSON:")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Reason: {result['reason']}")

except json.JSONDecodeError:
    print("\nError: Response was not valid JSON.")
    print("Raw Response:")
    print(raw_response)


# Prompt Q6 — Delimiters

# First prompt with step-by-step instructions
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."


prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
)

print("\nInstructions Response:")
print(response.choices[0].message.content)

# Second prompt with regular prose
user_text = (
    "The weather was beautiful, so we spent the afternoon walking through the park "
    "and enjoying the sunshine."
)

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
)

print("\nNon-Instructions Response:")
print(response.choices[0].message.content)

# Delimiters help prevent the model from confusing the user's input with the
# prompt instructions, making it clear which text should be analyzed.


# Ollama Question 1

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain what a large language model is in two sentences.",
        }
    ],
)

print("\nOpenAI Response:")
print(response.choices[0].message.content)

# Ollama Output:
"""
pulling manifest 
pulling 7f4030143c1c: 100% ▕██████████████████████████████ ▏ 522 MB/522 MB   15 MB/s      0s
pulling b0830f4ff6a0: 100% ▕███████████████████████████████▏  490 B                         
verifying sha256 digest 
writing manifest 
success 
Thinking...
Okay, the user is asking for a two-sentence explanation of what a large language model 
is. Let me start by breaking down the key points. 

First, a large language model is a type of AI model that can understand and generate 
text in natural, human-like ways. That's the basic definition. Then, I need to mention 
its capabilities. They might be interested in how it works, like processing a lot of 
text or handling various tasks. Maybe include examples like chatbots or translation 
services to make it concrete. I should also mention that it's trained on vast amounts 
of text for accuracy. Wait, is that all? Let me check the sentences again to ensure 
they're concise and cover both the main features and the example. Alright, that should 
work.
...done thinking.

A large language model is an advanced AI system designed to understand and generate 
human-like text, such as in writing, conversations, or coding. It processes vast 
amounts of text to learn patterns and languages, enabling it to perform tasks like 
translation, text generation, or answering questions with natural responses.
"""

# Both responses correctly explained what a large language model is. The
# Ollama response included a visible "Thinking..." section before the final
# answer, while the OpenAI response returned only the final explanation.
# The OpenAI response was also slightly more detailed and technical.
#
# One advantage of running a model locally is that it provides greater
# privacy because your data stays on your own computer. One disadvantage is
# that local models are often smaller and may produce less detailed or less
# accurate responses than larger cloud-hosted models.