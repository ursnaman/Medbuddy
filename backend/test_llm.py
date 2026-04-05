from groq import Groq

import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

system_prompt = """You are a helpful Medbuddy, an expert AI medical assistant.

When a user describes symptoms:
1. List the possible conditions(with likelihood)
2, suggest medicines available OTC (over the counter)

3. Give dosage guidance
4. Tell them when to see a doctor immediately

Always end with: 'Please consult a real doctor for proper diagonsis.'
keep your response clear and easy to understand."""


def ask_medbuddy(symptoms: str) -> str:
    """
    This function:
    1. Takes the user's symptoms as input
    2. Sends it to LLaMA 3 running on Groq
    3. Returns the AI's medical response
    """

    response = client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role":"user",
                "content": symptoms
            }
        ],
        temperature=0.2,

        max_tokens=1024,
    )

    return response.choices[0].message.content


if __name__ =="__main__":

    print("=" *50)
    print("Medbuddy AI  - Test Run")
    print("=" *50)

    symptom = "I have a bad headache, fever of 101F and body aches since yesterday"
    print(f"\nSymptom: {symptom}")
    print("\nMedbuddy's Response:")
    print(ask_medbuddy(symptom))

    # print("\n" + "=" *50)


    # symptom2 = " I have stomach pain, loose motions and vomiting since morning"
    # print(f"\nSymptom: {symptom2}")
    # print("\nMedbuddy's Response:")
    # print(ask_medbuddy(symptom2))