from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch
import os

# --- Настройки ---
MODEL_NAME = "EleutherAI/gpt-neo-125M"
LORA_PATH = "./lora_gptneo"
BATCH_SIZE = 1
MAX_LENGTH = 96
EPOCHS = 2
SAVE_STEPS = 500
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Загружаем токенизатор ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Добавляем pad_token, если его нет
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# --- Загружаем модель ---
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
base_model.resize_token_embeddings(len(tokenizer))
base_model.to(device)

# --- Настройка LoRA ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj"],  # корректные модули GPT-Neo
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)

# --- Загружаем данные ---
dataset = load_dataset("json", data_files={"train": "dialogs_5000.json"}, split="train")

# --- Токенизация ---
def tokenize_function(examples):
    # Объединяем user и assistant в одну строку
    texts = [u + " " + a for u, a in zip(examples["user"], examples["assistant"])]
    return tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"  # важно для батчей одинаковой длины
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["user", "assistant"])

# --- Data Collator ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir="./lora_output",
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_steps=100,
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    fp16=False,  # CPU
    gradient_accumulation_steps=8,
    remove_unused_columns=False,
    report_to=None
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# --- Запуск обучения ---
trainer.train()

# --- Сохраняем LoRA веса ---
model.save_pretrained(LORA_PATH)
print("Обучение завершено, LoRA веса сохранены в", LORA_PATH)
