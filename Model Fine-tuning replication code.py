# STEP 1: Install required libraries
!pip install transformers datasets peft accelerate bitsandbytes trl

# STEP 2: Import libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer
import torch
import pandas as pd

# STEP 3: Upload file to Colab
from google.colab import files
uploaded = files.upload()  # wvs_prompt_response_indonesia.csv

# STEP 4: Load data with pandas and refine prompt format
df = pd.read_csv("wvs_prompt_response_indonesia.csv")

# Format conversion function
import re

def convert_prompt(row):
    prompt_text = row["prompt"]

    try:
        age = re.search(r"A (\d+)-year-old", prompt_text).group(1)
        gender = re.search(r"-year-old (\w+)", prompt_text).group(1)
        income = re.search(r"with (\w+) income", prompt_text).group(1)
        religion = re.search(r"religion code (\d+)", prompt_text).group(1)
        year = re.search(r"in (\d{4}) was", prompt_text).group(1)
        question = re.search(r"was asked: (.+)", prompt_text).group(1)
    except AttributeError:
        return None  # Return None if no match

    country = "Indonesia"

    return (
        f"Respondent Profile:\n"
        f" - Age: {age}\n"
        f" - Gender: {gender}\n"
        f" - Income: {income}\n"
        f" - Religion Code: {religion}\n"
        f" - Country: {country}\n"
        f" - Year: {year}\n"
        f"Survey Question:\n"
        f"{question}\n"
        f"Answer: {row['answer']}"
    )

# Apply and convert to Hugging Face Dataset
df["text"] = df.apply(convert_prompt, axis=1)
df = df.dropna(subset=["text"])  # Remove rows where conversion failed
dataset = Dataset.from_pandas(df[["text"]])

# STEP 5: Load tokenizer and model
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

# STEP 6: LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value", "dense"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# STEP 7: Training configuration
training_args = TrainingArguments(
    output_dir="./qlora-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    num_train_epochs=6,  # ✅ Increased number of epochs
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    report_to="none"
)

# STEP 8: Start training
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args
)

trainer.train()

# STEP 9: Save model
model.save_pretrained("./qlora-finetuned")
tokenizer.save_pretrained("./qlora-finetuned")

# STEP 10: Compress and download model from Colab
import shutil
shutil.make_archive("qlora-finetuned", 'zip', "./qlora-finetuned")
files.download("qlora-finetuned.zip")