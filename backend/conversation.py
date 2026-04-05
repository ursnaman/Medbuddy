from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

system_prompt = """You are Medbuddy, an expert AI medical Assistant.

Remember  everthing the user tells you in this conversation.
Ask follow-up questions to better understand their condition.

when you have enough information:
1. List the possible conditions with likelihood percentage
2. Suggest OTC medicines with doage guidance
3. Tell them when to see a doctor immediately

Always end with: Please consult a real doctor for proper diagnosis."""

class MedbuddyChat:
    """
    This class manages a full conversation with memory.

    How memory works:
    -We keep a list called conversation_history
    -Every message(user + AI) gets added to this list
    -Every time we call the API, we send the FULL history
    -So the AI always knows everything said before
    -This is exactly how ChatGPT remembers your messages
    """

    def __init__(self):
        self.conversation_history = []
        print("Medbuddy: Hello! I am Medbuddy. Tell me your symptoms.")
        print("-" * 50)

    def chat(self, user_message):

        self.conversation_history.append({
            "role": "user",
            "content":user_message
        })

        response = client.chat.completions.create(
            model = "llama-3.1-8b-instant",
            messages=[
                {"role": "system","content":system_prompt},
                *self.conversation_history
            ],
            temperature=0.2,
            max_tokens=1024,
        )

        ai_response = response.choices[0].message.content

        self.conversation_history.append({
            "role": "assistant",
            "content": ai_response
        })

        return ai_response
    
if __name__ =="__main__":
    print("=" *50)
    print("Medbuddy AI  - Full Conversation Test")
    print("Type 'quit' to exit")
    print("=" *50)
    print()

    chat = MedbuddyChat()


    while True:
        user_input = input("You: ")

        if user_input.lower() == "quit":
            print("Medbuddy: Goodbye! Stay healthy!")
            break

        if not user_input.strip():
            continue

        response = chat.chat(user_input)
        print(f"\nMedbuddy: {response}")
        print("-" * 50)