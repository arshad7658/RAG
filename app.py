import uuid
import faiss
import json
from typing import Dict, List
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from google.generativeai import GenerativeModel, ChatSession
from google.generativeai.types import HarmCategory, HarmBlockThreshold

API_KEY= "AIzaSyCJAWHor2PFU3IbDsQ7yNdiMUWvVRwUDUQ"
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("data/traffic_rules.index")
with open("data/chunks.json", "r") as f:
    chunks = json.load(f)



genai.configure(api_key=API_KEY)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

chat_sessions: Dict[str, ChatSession] = {}
model = GenerativeModel('gemini-1.5-flash')

safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}



def get_chunks(query:str, k:int=3):
    vecs=embedder.encode([query], convert_to_numpy=True)
    distances, indcs = index.search(vecs, k)
    return [chunks[i] for i in indcs[0]]



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    session_id = str(uuid.uuid4())
    return templates.TemplateResponse("index.html", {"request": request, "session_id": session_id, "chat_history": []})

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...), session_id: str = Form(...)):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = model.start_chat(history=[])

    chat_session = chat_sessions[session_id]

    retrieved_chunks = get_chunks(user_input, k=3)


    model_prompt = (
        f"You are a local traffic law advisor.\n\n"
        f"User question:\n{user_input}\n\n"
        f"Relevant context from traffic laws and penalties:\n"
        + "\n".join(retrieved_chunks)
        + "\n\nGuidelines:\n"
        "- Base your answer only on the provided context. Be logical and always double check context\n"
        "- If the violation is not in context, say it is not listed.\n"
        "- For multiple violations, if fines exist for each, also provide the total fine.\n"
        "- Extract only the relevant details (do not dump the entire law text). But meantion the regulation and section number if available.\n"
        "- Use clear, simple bullet points starting with '-'.\n"
        "- End with a final section labeled 'Answer:'."
    )




    
    response = await chat_session.send_message_async(model_prompt, safety_settings=safety_settings)

    model_response = response.text
    chat_session.history[-2].parts[0].text = user_input 
    chat_history = chat_session.history
    return templates.TemplateResponse("index.html", 
                                      {"request": request, 
                                       "session_id": session_id, 
                                       "chat_history": chat_history, 
                                       "model_response": model_response})
