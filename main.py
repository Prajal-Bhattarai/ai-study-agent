import os
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

class Topic(BaseModel):
    name:str
    importance:str
class TopicList(BaseModel):
    topics:list[Topic]

load_dotenv()

#Get the Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

#Check that the API key exists
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")

#Create the Gemini Client
client = genai.Client(api_key=api_key)

#Send a request to Gemini
study_instruction = """
You are an AI study assistant.

For this task, identify the most important academic topics from the student's input.

Focus on concepts that are important for understanding the subject and useful for exam preparation.

For each topic, provide a clear topic name and classify its importance as high, medium, or low.

Return only the requested structured data.
"""

while True:
    question = input("You: ")
    if question.lower()=="exit":
        break

    interaction = client.interactions.create(
        model="gemini-3.5-flash-lite",
        system_instruction=study_instruction,
        input=question,
        response_format={
        "type": "text",
        "mime_type": "application/json",
        "schema": TopicList.model_json_schema(),
    },
    )
    
    topics = TopicList.model_validate_json(interaction.output_text)

    print(interaction.output_text)


