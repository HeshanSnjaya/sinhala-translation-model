import pandas as pd
from transformers import MBart50Tokenizer, MBartForConditionalGeneration

# Load the trained model and tokenizer
model_path = './sinhala_final_translation_model'  # Path where your model is saved
tokenizer = MBart50Tokenizer.from_pretrained(model_path)
model = MBartForConditionalGeneration.from_pretrained(model_path)

# Load dictionary for rule-based segmentation
def load_dictionary(dictionary_path):
    word_dict_df = pd.read_csv(dictionary_path, sep='\t')
    return set(word_dict_df['Transliteration'])  

# Rule-based segmentation function
def rule_based_segmentation(text, dictionary):
    words = []
    i = 0
    while i < len(text):
        for j in range(len(text), i, -1):
            segment = text[i:j]
            if segment in dictionary:
                words.append(segment)
                i = j - 1
                break
        else:
            words.append(text[i])
            dictionary.add(text[i])  # Add new word to dictionary
        i += 1
    return words

def segment_text(input_text, dictionary_path):
    # Load the dictionary
    word_dict = load_dictionary(dictionary_path)

    # Perform rule-based segmentation and update dictionary
    segmented_words = rule_based_segmentation(input_text, word_dict)

    # Join segmented words into a sentence
    segmented_sentence = " ".join(segmented_words)
    return segmented_sentence

# Function to translate text
def translate(text, dictionary_path):
    # Segment the input text first
    segmented_text = segment_text(text, dictionary_path)
    
    # Tokenize the segmented text
    inputs = tokenizer(segmented_text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    
    # Get the translated sentence
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    # Return original, segmented, and translated sentences
    return text, segmented_text, translated_text

# Example usage
example_sentences = [
    "අණිකටශොණපිතහබරියඋපශිකතියලෙණෙ",
    "අණිකටශොණපිතහබරිය[උ]පශිකති[ශ]යලෙණෙඅගතඅනගතචතුදිශශගශ",
    "පරුමකපලිකදපුතපරුමකපලිකදපුතඋපසකහරුමසලෙණෙචතුදිසශගස"
]

dictionary_path = "D:/Heshan/Developments/test/Translate/word_translation_dictionary.tsv"  # Update your dictionary path here

print("Sentence translation results:")
for sentence in example_sentences:
    original, segmented, translated = translate(sentence, dictionary_path)
    print(f"Original: {original}")
    print(f"Segmented: {segmented}")
    print(f"Translated: {translated}")
    print("---")
