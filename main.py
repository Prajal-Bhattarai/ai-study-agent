import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

#Get the Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

#Check that the API key exists
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")

#Create the Gemini Client
client = genai.Client(api_key=api_key)

#Send a request to Gemini 
while True:
    question = input("You: ")
    if question.lower()=="exit":
        break

    interaction = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input=question,
    )
    print(interaction.output_text)


