from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

HISTORY_FOLDER = "data"

SYSTEM_PROMPT = """You are MedBuddy, an expert AI medical assistant.

When a user describes symptoms:
1. List possible conditions with likelihood percentage
2. Suggest OTC medicines with dosage
3. Tell them when to see a doctor immediately

Always end with: Please consult a real doctor for proper diagnosis."""


class MedBuddyChat:

    def __init__(self, user_id="default_user"):
        self.user_id = user_id
        self.history_file = f"{HISTORY_FOLDER}/history_{user_id}.json"
        self.conversation_history = self.load_history()

    def load_history(self):
        os.makedirs(HISTORY_FOLDER, exist_ok=True)
        if os.path.exists(self.history_file):
            with open(self.history_file, "r") as f:
                return json.load(f)
        return []

    def save_history(self):
        with open(self.history_file, "w") as f:
            json.dump(self.conversation_history, f, indent=2)

    def chat(self, user_message):
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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

        self.save_history()
        return ai_response

    def clear_history(self):
        self.conversation_history = []
        if os.path.exists(self.history_file):
            os.remove(self.history_file)