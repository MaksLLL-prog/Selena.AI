from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os
import asyncio
import uuid

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Модель ---
base_model_name = "EleutherAI/gpt-neo-125M"  # лёгкая модель для CPU
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name)
device = "cpu"
model.to(device)

# --- LoRA ---
lora_path = "./lora_gptneo"
if os.path.exists(lora_path):
    model = PeftModel.from_pretrained(model, lora_path)
    model.to(device)

# --- История диалога ---
chat_sessions = {}

class Message(BaseModel):
    session_id: str | None = None
    role: str
    content: str

SYSTEM_PROMPT = (
    "Ты — вежливый и умный ассистент. "
    "Отвечай по теме, ясно и коротко. "
    "Не придумывай лишнего.\n"
)

MAX_TOKENS = 512  # максимум токенов для истории

# --- Формирование prompt с историей ---
def format_chat_prompt(session_id: str, new_user_message: str) -> str:
    if session_id not in chat_sessions:
        # создаём историю и вставляем system prompt
        chat_sessions[session_id] = [f"Система: {SYSTEM_PROMPT.strip()}"]

    # добавляем новое сообщение пользователя
    chat_sessions[session_id].append(f"Пользователь: {new_user_message}")

    # собираем историю
    history_text = "\n".join(chat_sessions[session_id]) + "\nАссистент:"

    # токены истории
    tokenized = tokenizer(history_text, return_tensors="pt")
    if tokenized.input_ids.shape[-1] > MAX_TOKENS:
        # если слишком длинно — обрезаем, но system prompt оставляем
        input_ids = tokenized.input_ids[0, -MAX_TOKENS:].unsqueeze(0)
        return tokenizer.decode(input_ids[0], skip_special_tokens=False)

    return history_text

# --- Генерация ответа (синхронная) ---
def generate_response(session_id: str, user_input: str) -> str:
    prompt = format_chat_prompt(session_id, user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    # добавляем ответ ассистента в историю
    chat_sessions[session_id].append(f"Ассистент: {response}")

    return response

# --- Асинхронная обёртка ---
async def generate_response_async(session_id: str, user_input: str) -> str:
    return await asyncio.to_thread(generate_response, session_id, user_input)

# --- HTTP POST endpoint ---
@app.post("/chat")
async def chat_endpoint(message: Message):
    session_id = message.session_id or str(uuid.uuid4())
    reply = await generate_response_async(session_id, message.content)
    return {"session_id": session_id, "content": reply}

# --- Получение истории ---
@app.get("/sessions/{session_id}/history")
async def get_history(session_id: str):
    messages = chat_sessions.get(session_id, [])
    return [
        {"role": "user" if m.startswith("Пользователь:") else
                 "assistant" if m.startswith("Ассистент:") else "system",
         "content": m.split(": ", 1)[1]}
        for m in messages
    ]

# --- Главная страница ---
@app.get("/")
async def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "App.html"))

# --- WebSocket endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = "ws_session"
    if session_id not in chat_sessions:
        chat_sessions[session_id] = [f"Система: {SYSTEM_PROMPT.strip()}"]

    while True:
        try:
            data = await ws.receive_text()
            reply = await generate_response_async(session_id, data)
            await ws.send_text(reply)
        except Exception as e:
            print("WebSocket error:", e)
            await ws.send_text("Произошла ошибка, попробуйте позже.")
