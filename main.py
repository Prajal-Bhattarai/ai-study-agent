import os
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
from typing import Literal
from pypdf import PdfReader


class Topic(BaseModel):
    name: str
    importance: Literal["high", "medium", "low"]


class TopicList(BaseModel):
    topics: list[Topic]


class StudyNote(BaseModel):
    topic: str
    explanation: str
    key_points: list[str]


class StudyNotes(BaseModel):
    notes: list[StudyNote]

class Question(BaseModel):
    question: str
    answer: str
    difficulty: Literal["easy","medium","hard"]

class QuestionList(BaseModel):
    questions: list[Question]

class Evaluation(BaseModel):
        result: Literal["correct", "partially_correct", "incorrect"]
        feedback: str
        correct_answer: str


load_dotenv()

# Get the Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

# Check that the API key exists
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")


# Create the Gemini Client
client = genai.Client(api_key=api_key)


# Read the PDF
pdf_path = "sample.pdf"
reader = PdfReader(pdf_path)

print("Number of pages:", len(reader.pages))


# Extract text from all pages
text = ""

for page in reader.pages:
    text += page.extract_text()


# Topic Analyzer instruction
study_instruction = """
You are an AI study assistant.

For this task, identify the most important academic topics from the student's input.

Focus on concepts that are important for understanding the subject and useful for exam preparation.

Classify each topic using these criteria:
- high: fundamental concepts, major topics, or concepts that are likely to be important for exams.
- medium: supporting concepts that help understand the main topics.
- low: minor details or supplementary information.

For each topic, provide a clear topic name and classify its importance as high, medium, or low.

Return only the requested structured data.
"""


# Notes Generator instruction
notes_instruction = """
You are an AI study notes generator.

Create clear and concise study notes from the provided academic material.

Focus especially on the important concepts in the material.

For each topic, provide:
- a clear explanation
- the most important key points

Keep the notes useful for exam preparation and easy to revise.

Return only the requested structured data.
"""


question_instruction = """
You are an AI exam question generator.

Generate useful exam questions from the provided study notes.

Focus on the important concepts and create questions that test the student's understanding.

For each question:
- provide the question
- provide the correct answer
- classify the difficulty as easy, medium, or hard

Make the questions suitable for Computer Science exam preparation.

Return only the requested structured data.
"""


evaluation_instruction = """
You are an AI answer evaluator.

Evaluate the student's answer against the provided question and correct answer.

Classify the student's answer as:
- correct: the answer is accurate and sufficiently complete.
- partially_correct: the answer contains some correct information but misses important parts.
- incorrect: the answer is wrong or does not demonstrate understanding.

Provide clear and concise feedback explaining why the answer received that classification.

Also provide the correct answer.

Return only the requested structured data.
"""



# First Gemini interaction: identify important topics
interaction = client.interactions.create(
    model="gemini-3.5-flash-lite",
    system_instruction=study_instruction,
    input=text,
    response_format={
        "type": "text",
        "mime_type": "application/json",
        "schema": TopicList.model_json_schema(),
    },
)


# Convert Gemini output into a TopicList object
topics = TopicList.model_validate_json(interaction.output_text)


# Convert topics into text
topic_text = ""

for topic in topics.topics:
    topic_text += topic.name + " - " + topic.importance + "\n"


print(topic_text)


# Combine PDF text and important topics
notes_input = text + "\n\nImportant topics:\n" + topic_text


# Second Gemini interaction: generate study notes
notes_interaction = client.interactions.create(
    model="gemini-3.5-flash-lite",
    system_instruction=notes_instruction,
    input=notes_input,
    response_format={
        "type": "text",
        "mime_type": "application/json",
        "schema": StudyNotes.model_json_schema(),
    },
)


# Convert Gemini output into a StudyNotes object
notes = StudyNotes.model_validate_json(notes_interaction.output_text)


question_interaction = client.interactions.create(
    model="gemini-3.5-flash-lite",
    system_instruction=question_instruction,
    input=notes_interaction.output_text,
    response_format={
        "type": "text",
        "mime_type": "application/json",
        "schema": QuestionList.model_json_schema(),
    },
)

questions = QuestionList.model_validate_json(question_interaction.output_text)

for question in questions.questions:
    print(question.question)

    student_answer = input("Your answer: ")

    evaluation_interaction = client.interactions.create(
        model="gemini-3.5-flash-lite",
        system_instruction=evaluation_instruction,
        input=(
            "Question: " + question.question +
            "\nStudent Answer: " + student_answer +
            "\nCorrect Answer: " + question.answer
        ),
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": Evaluation.model_json_schema(),
        },
    )

    evaluation = Evaluation.model_validate_json(
        evaluation_interaction.output_text
    )

    print("Result:", evaluation.result)
    print("Feedback:", evaluation.feedback)
    print("Correct Answer:", evaluation.correct_answer)
    print()


# Display the study notes
for note in notes.notes:
    print(note.topic)
    print(note.explanation)
    print("Key Points:")

    for point in note.key_points:
        print("-", point)