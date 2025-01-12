import pandas as pd
from transformers import MBart50Tokenizer, MBartForConditionalGeneration
from datasets import Dataset
from transformers import Trainer, TrainingArguments

# Load the new corrected dataset
corrected_data_path = "D:/Heshan/Developments/test/Translate/corrected_sinhala_translation.xlsx"
df = pd.read_excel(corrected_data_path)

# Normalize the source and target sentences
def normalize_sentence(sentence):
    return sentence.replace('-', ' ').replace('[', '').replace(']', '')

df['Transliteration (Sinhala)'] = df['Transliteration (Sinhala)'].apply(normalize_sentence)
df['Translation (Sinhala)'] = df['Translation (Sinhala)'].apply(normalize_sentence)

# Rename columns to 'source' and 'target'
df.columns = ['source', 'target']

# Load the saved model and tokenizer
saved_model_path = "./sinhala_final_translation_model"
tokenizer = MBart50Tokenizer.from_pretrained(saved_model_path)
model = MBartForConditionalGeneration.from_pretrained(saved_model_path)

# Tokenize the corrected dataset
def preprocess_function(examples):
    model_inputs = tokenizer(examples['source'], max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(examples['target'], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Convert to Hugging Face Dataset
corrected_dataset = Dataset.from_pandas(df)
processed_corrected_dataset = corrected_dataset.map(preprocess_function, batched=True)

# Split into train and validation sets
split_datasets = processed_corrected_dataset.train_test_split(test_size=0.2)
train_dataset = split_datasets['train']
val_dataset = split_datasets['test']

# Define fine-tuning arguments
training_args = TrainingArguments(
    output_dir='./corrected_model_results',
    evaluation_strategy="epoch",
    learning_rate=1e-5,  # Use a lower learning rate for fine-tuning
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir='./corrected_model_logs',
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Start fine-tuning
print("Starting fine-tuning on corrected data...")
trainer.train()
print("Fine-tuning complete.")

# Save the fine-tuned model with a new name
fine_tuned_model_path = './sinhala_corrected_translation_model'
model.save_pretrained(fine_tuned_model_path)
tokenizer.save_pretrained(fine_tuned_model_path)

print(f"Fine-tuned model saved to {fine_tuned_model_path}")
