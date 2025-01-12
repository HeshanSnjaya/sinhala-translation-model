import pandas as pd
from transformers import MBart50Tokenizer, MBartForConditionalGeneration
from datasets import Dataset
from transformers import Trainer, TrainingArguments

# Load and preprocess dataset
df = pd.read_excel("D:/Heshan/Developments/test/Translate/sinhala_translation.xlsx")

# Normalize transliteration column to remove unwanted characters
def normalize_sentence(sentence):
    return sentence.replace('-', ' ').replace('[', '').replace(']', '')

# Update column names based on the DataFrame
df['Transliteration (Sinhala)'] = df['Transliteration (Sinhala)'].apply(normalize_sentence)
df['Translation (Sinhala)'] = df['Translation (Sinhala)'].apply(normalize_sentence)

# Rename columns for clarity
df.columns = ['source', 'target']  # Now renaming to 'source' and 'target'

# Tokenizer and Model for mBART-large-50
model_name = 'facebook/mbart-large-50'
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Tokenize the dataset
def preprocess_function(examples):
    model_inputs = tokenizer(examples['source'], max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(examples['target'], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)
processed_dataset = dataset.map(preprocess_function, batched=True)

# Split into train and validation
split_datasets = processed_dataset.train_test_split(test_size=0.2)
train_dataset = split_datasets['train']
val_dataset = split_datasets['test']

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Start training
print("Starting training...")
trainer.train()
print("Training complete.")

# Save the trained model
model.save_pretrained('./sinhala_final_translation_model')
tokenizer.save_pretrained('./sinhala_final_translation_model')

# Translate new sentences
def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Example translation with multiple new sentences
new_sentences = [
    "අණිකට ශොණ පිතහ බරිය උපශික තිය ලෙණෙ",
    "දමරකිත තෙරශ",
    "පරුමක-පලිකද-පුත පරුමක-පලිකද-පුත උපසක-හරුමස ලෙණෙ චතු-දිස ශගස"
]

for sentence in new_sentences:
    print(f"Original: {sentence}")
    print(f"Translated: {translate(sentence)}\n")
