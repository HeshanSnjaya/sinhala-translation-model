from googletrans import Translator

# Initialize the Google Translator
translator = Translator()

def translate_text(text, src_lang='si', tgt_lang='en'):
    """
    Translate text from source language to target language using Google Translate.
    
    :param text: Text to be translated
    :param src_lang: Source language code (default is 'si' for Sinhala)
    :param tgt_lang: Target language code (default is 'en' for English)
    :return: Translated text
    """
    try:
        # Translate the text
        translation = translator.translate(text, src=src_lang, dest=tgt_lang)
        return translation.text
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    # Example Sinhala text
    sinhala_text = "අණිකට කුමරුගේ භාර්යාව වන උපාසිකා නාගාගේ ලෙණ පැමිණියාවූත් නොපැමිණියාවූත් සතරදෙස සංඝයාට දෙන ලදී."
    
    # Translate to English
    translated_text = translate_text(sinhala_text)
    
    # Print the translated text
    print(f"Original (Sinhala): {sinhala_text}")
    print(f"Translated (English): {translated_text}")
