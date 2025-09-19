import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()

# --- FIX IS HERE ---
# Use the relative path to include the 'Include' folder
try:
    with open('Include/intents.json', encoding='utf-8') as f:
        intents = json.load(f)
except FileNotFoundError:
    print("Error: 'Include/intents.json' not found. Check the file path.")
    # You might want to exit or raise an exception here
# --- END FIX ---

# The paths for these files remain the same (in the same directory as the script)
try:
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbot_model.h5')
except FileNotFoundError as e:
    print(f"Error loading required files: {e}. Ensure 'words.pkl', 'classes.pkl', and 'chatbot_model.h5' are present.")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words] 
    return sentence_words

def bag_of_words (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class (sentence):
    bow = bag_of_words (sentence)
    if not np.any(bow):
        return []
    
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that."
        
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice (i['responses'])
            break
    else:
        result = "Hmm, I know that intent but can't find a response for it."
    return result

print("GO! Bot is running!")

while True:
    try:
        message = input("You: ")
        if message.lower() in ['quit', 'exit', 'bye']:
            print("Bot: Goodbye!")
            break
            
        ints = predict_class (message)
        res = get_response (ints, intents)
        print (f"Bot: {res}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")