from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

# Task 1: Setup and System Prompt

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

system_prompt = """
You are a job application coach who helps career changers create stronger
application materials for roles in a new field.

Your job is to help the user:
- rewrite resume bullet points so their existing experience highlights
  transferable skills relevant to the job they are targeting
- draft and improve cover letters
- answer follow-up questions about job application materials
- explain suggested wording when helpful

Stay focused on job application materials. If the user asks for something
unrelated, politely redirect the conversation back to resumes, cover letters,
job descriptions, or other application-related materials.

Do not invent experience, qualifications, achievements, employers, education,
skills, or accomplishments that the user has not provided.

When rewriting material, preserve the user's actual experience while making
the language clear, concise, professional, and relevant to the target role.

Always remind the user to review and edit AI-generated application materials
before submitting them to an employer.

You may not know the specific expectations or terminology used in every
industry or organization. Acknowledge this when appropriate and encourage
the user to use their own judgment and knowledge of their target industry.
"""

# I made the assistant specifically a job application coach for career changers
# so its responses stay focused on translating previous experience into language
# that is relevant to the user's target role 
# rather than giving generic career advice.


# Task 2: Bullet Point Rewriter

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]

    response = get_completion(messages)

    response = response.strip()

    if response.startswith("```json"):
        response = response.removeprefix("```json").removesuffix("```").strip()
    elif response.startswith("```"):
        response = response.removeprefix("```").removesuffix("```").strip()

    rewritten = json.loads(response)

    print("\nRewritten Resume Bullet Points:")
    print("-" * 60)

    for item in rewritten:
        print(f"Original: {item['original']}")
        print(f"Improved: {item['improved']}")
        print()

    return rewritten

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time",
]

rewrite_bullets(bullets)

# The original bullet points were weak because they were vague, used weak action
# verbs, and did not clearly communicate the value or impact of the work.
# The model improved them by using stronger action verbs, making the language
# more professional, and emphasizing collaboration, problem-solving, and
# business impact. However, it also added details such as "two weeks ahead of
# schedule" that were not provided in the original bullets, showing why AI-
# generated content should always be reviewed for accuracy.


# Task 3: Cover Letter Generator

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]

    return get_completion(messages)

job_title = "Junior Data Engineer"

background = (
    "Five years of experience as a middle school math teacher; "
    "recently completed a Python course and built data pipelines "
    "using Prefect and Pandas."
)

letter = generate_cover_letter(job_title, background)

print("\nCover Letter Opening:\n")
print(letter)

# I chose examples from two different career changes to show how 
# transferable skills can be connected to new technical abilities. 
# 
# The few-shot examples help guide the model's tone, structure, 
# and level of specificity, producing a more personalized 
# cover letter opening instead of a generic introduction.


# Task 4: Moderation Check

# Moderation function
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )

    flagged = result.results[0].flagged

    if flagged:
        # Message for the user
        print("Your message may contain content that can't be processed. Please rephrase your request.")

        # Flagging causes
        print("Triggered categories:")
        print(result.results[0].categories)
        
        return False

    return True

# Positive test
print(is_safe("Can you help me improve my resume?"))

# Negative test
print(is_safe("I want to hack into my company's database."))


# Task 5: The Chatbot Loop

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            rewrite_bullets(raw_bullets)

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            letter = generate_cover_letter(job_title, background)
            print("\nJob Application Helper:")
            print(letter)

        # 7. Otherwise, handle it as a regular chat turn
        else:
            messages.append({"role": "user", "content": user_input})

            reply = get_completion(messages)

            print(f"\nJob Application Helper: {reply}")

            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_chatbot()


# Task 6: Ethics Reflection

# AI models are trained on large collections of text that may overrepresent
# certain industries, communication styles, or cultural backgrounds. As a
# result, the advice they generate may favor common resume or cover letter
# conventions while overlooking expectations in other industries or regions.
#
# If a job seeker did not review the bot's output carefully, 
# the model may invent details, exaggerate accomplishments, 
# or include generic language that does not accurately reflect
# the person's experience or the specific job they are applying for.
#
# If I were deploying this tool professionally, I would add guardrails such as
# a moderation check to detect potentially unsafe or inappropriate requests
# before sending them to the AI model. I would also remind users to verify all
# AI-generated content before submitting it and prevent the assistant from
# inventing qualifications or work experience. Finally, I would clearly explain
# that the AI is intended to assist with writing, not replace the user's own
# judgment.