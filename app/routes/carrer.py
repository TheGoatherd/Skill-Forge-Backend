from fastapi import APIRouter
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import os

router = APIRouter()
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class User_Input(BaseModel):
    interest: str

class Topic(BaseModel):
    topic: str

class RoadMap(BaseModel):
    goal: str


@router.post("/career")
async def get_career_suggestion(user_input: User_Input):
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a career advisor AI trained to help individuals discover suitable career paths based on their interests, strengths, and goals.\n\n"
                    f"Input from user:\n{user_input.interest}\n\n"
                    "Based on the input, suggest 3 potential career options. For each suggestion:\n"
                    "- Name the career.\n"
                    "- Explain in 2-3 lines why it is a good fit.\n"
                    "- Mention one potential future opportunity or growth in that field.\n\n"
                    "Ensure your recommendations are realistic, encouraging, and tailored to the user's described interests or background."
                )
            },
            {
                "role": "user",
                "content": f"My interests are {user_input.interest}. Please provide detailed career suggestions."
            }
        ],
        temperature=0.9,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
    )
    suggestion = ""
    for chunk in completion:
        if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
            suggestion += chunk.choices[0].delta.content
    return {"suggestion": suggestion}


@router.post("/roadmap")
async def get_roadmap(user_input: RoadMap):
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional career roadmap advisor.\n\n"
                    f"User goal:\n{user_input.goal}\n\n"
                    "Instructions:\n"
                    "- Break the roadmap into **3 weekly modules**.\n"
                    "- For each week, include:\n"
                    "  - Clear objectives\n"
                    "  - Key skills/topics\n"
                    "  - Recommended resources with URLs\n"
                    "- Be motivational and beginner-friendly.\n"
                    "Only return the roadmap."
                )
            },
            {
                "role": "user",
                "content": f"I want a roadmap for {user_input.goal}."
            }
        ],
        temperature=0.9,
        max_completion_tokens=1024,
        top_p=1,
    )
    return {"roadmap": completion.choices[0].message.content}


@router.post("/resource")
async def resource_find(user_input: Topic):
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a resource recommendation engine trained to help learners find high-quality materials.\n\n"
                    f"User query:\n{user_input.topic}\n\n"
                    "Suggest 3 top resources:\n"
                    "- Prefer YouTube videos\n"
                    "- Include: title, topic, URL, short description, review/testimonial\n"
                    "Ensure relevance and beginner-friendliness."
                )
            },
            {
                "role": "user",
                "content": f"Find resources for {user_input.topic}."
            }
        ],
        temperature=0.9,
        max_completion_tokens=1024,
        top_p=1,
    )
    return {"resources": completion.choices[0].message.content}
