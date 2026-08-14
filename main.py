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
study_instruction = '''You are an AI study assistant.
                       Explain concepts clearly and in a way that is easy for students to understand.
                       Identify and highlight the most important concepts for exam preparation.
                       Organize explanations into clear headings, key points, and concise summaries.
                       Generate practice questions that test the student's understanding of the material.
                       When a student provides an answer, evaluate their understanding, point out mistakes, and explain how they can improve.
                       '''
while True:
    question = input("You: ")
    if question.lower()=="exit":
        break

    interaction = client.interactions.create(
        model="gemini-3.5-flash-lite",
        system_instruction=study_instruction,
        input=question,
    )
    print(interaction.output_text)


