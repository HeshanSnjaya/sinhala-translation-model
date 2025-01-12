from transformers import MBart50Tokenizer, MBartForConditionalGeneration
from nltk.translate.bleu_score import corpus_bleu

# Load the trained model and tokenizer
model_path = './sinhala_final_translation_model'  # Path where your model is saved
tokenizer = MBart50Tokenizer.from_pretrained(model_path)
model = MBartForConditionalGeneration.from_pretrained(model_path)

# Function to translate text
def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Example sentences for translation and corresponding reference translations
example_sentences = [
    "අණිකට-ශොණ-පිතහ බරිය [උ]පශික-ති[ශ]ය ලෙණෙ අගත-අනගත-චතු-දිශ-ශගශ",
    "පරුමක-පලිකද-පුත පරුමක-පලිකද-පුත උපසක-හරුමස ලෙණෙ චතු-දිස ශගස",
    "පරුමක-පලිකදස බරිය පරුමක-ශුරකිත-ක්ධි ත උපශික-චිතය ලෙණෙ ශගශ චතු-දිශ"
]

# Corresponding reference translations (the ground truth)
references = [
    ["අශ්වාරෝහක ශොණගේ පියාගේ බිරිද වූ තිස්සා උපාසිකාවගේ ලෙණ පැමිණි නොපැමිණි සිව්දිග සංඝයාටයි"],
    ["පරුමක පලිකදගේ පුත් පරුමක පලිකදගේ පුත් උපාසක හරුමගේ ලෙණ සිව්දිග සංඝයාටයි"],
    ["ප්‍රමුඛ පලිකදගේ භාර්යාව වූ ද පරශි මුඛ සුරක්ඛිතගේ දියණිය වූ උපාසිකා චිත්තාගේ ලෙණ සිව්දිග සංඝයාටයි"]
]

# Translate the sentences and store predictions
predictions = []
for sentence in example_sentences:
    translated = translate(sentence)
    predictions.append(translated)

# Calculate BLEU score using corpus_bleu
bleu_score = corpus_bleu(references, predictions)
print(f"BLEU score: {bleu_score:.4f}")

# Print the translations
print("\nTranslated sentences:")
for sentence, translated in zip(example_sentences, predictions):
    print(f"Original: {sentence}")
    print(f"Translated: {translated}")
    print("---")
