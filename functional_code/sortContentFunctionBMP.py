import nltk
from nltk import pos_tag, word_tokenize
import os
import re
import shutil

# Download necessary data for POS tagging
nltk.download('averaged_perceptron_tagger_eng')
nltk.download("punkt")

# Function words POS tags (closed class words)
FUNCTION_TAGS = {
    "DT",   # Determiner (e.g., the, a, an)
    "CC",   # Coordinating conjunction (e.g., and, but, or)
    "IN",   # Preposition or subordinating conjunction (e.g., in, on, of, because)
    "PRP",  # Personal pronoun (e.g., he, she, it, they)
    "PRP$", # Possessive pronoun (e.g., his, her, their)
    "TO",   # "to" as a preposition or infinitive marker
    "MD",   # Modal verb (e.g., can, will, must)
    "WP",   # Wh-pronoun (e.g., who, what, which)
    "WRB",  # Wh-adverb (e.g., when, where, why)
    "EX"    # Existential "there"
}
def classify_word_nltk(word):
    """Classifies a word as 'Content word' or 'Function word' using NLTK POS tagging."""
    word = word.lower()  # Convert word to lowercase
    
    # Tokenize and POS tag the word
    tag = pos_tag([word])[0][1]  # Get the POS tag
    
    if tag in FUNCTION_TAGS:
        return "function"
    else:
        return "content"
    
def extract_word(dir):
    bmps = os.listdir(dir)
    print ("\n",bmps[0])
    filename = bmps[0]

    if not filename.endswith(".bmp"):  # Skip non-BMP files
        print ("error1:", filename)
        return None
        
    # Remove 'content' or 'function' from the start
    filename = re.sub(r'^(content|function)', '', filename)

    # Extract words that are NOT part of the numbers
    matches = re.findall(r'[a-zA-Z]+', filename)  # Find all words

    if matches:
        return matches[0]
    else:
        print ("error2:", filename)  # If no word is found

    return None

def getUniqueFolder(base_path, word):
    """Finds a unique folder name by appending a number if needed."""
    folder_path = os.path.join(base_path, word)
    counter = 1

    while os.path.exists(folder_path):  # Check if folder already exists
        folder_path = os.path.join(base_path, f"{word}{counter}")
        counter += 1

    return folder_path

def exportToContent(source_folder, word):
    destination_folder = getUniqueFolder("/home/greg/Documents/GregCode/spectrogramDataHighGranFull2/content", word)
    shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)

def exportToFunction(source_folder, word):
    destination_folder = getUniqueFolder("/home/greg/Documents/GregCode/spectrogramDataHighGranFull2/function", word)
    shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)

def main():
    folders = ["content", "function"]
    pariticpants = 1

    for folder in folders:
        print ("\n\n\n",folder)
        for i in range (1,pariticpants+1):
            directory = "/home/greg/Documents/GregCode/dataSets/spectrogramDataHighGranFull/"+folder+"/"+str(i)+"/"
            word_folders = os.listdir(directory)  # List all files and folders
            for word_folder in word_folders:
                sourceFolder = os.path.join(directory, word_folder)
                word = extract_word(sourceFolder)
                wordType = classify_word_nltk(word)

                if (wordType=="content"):
                    exportToContent(sourceFolder, word)

                elif (wordType=="function"):
                    exportToFunction(sourceFolder, word)

                else:
                    print ("error!", classify_word_nltk(word))

                # print (os.path.join(directory, word_folder)) #source folder

                



main()


#exmaple files
# content126pick3.csv
# content113by3.csv
# function17at3.csv
# function17at33.csv
# function141at14.csv