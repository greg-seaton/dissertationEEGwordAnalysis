import nltk
from nltk import pos_tag
import os
import re
import shutil

# Download necessary data for POS tagging
nltk.download('averaged_perceptron_tagger_eng')
nltk.download("punkt")

# Function words POS tags (closed class words)
FUNCTION_TAGS = {
    "DT", "CC", "IN", "PRP", "PRP$", "TO", "MD", "WP", "WRB", "EX"
}


def classify_word_nltk(word):
    word = word.lower()  #convert word to lowercase
    
    tag = pos_tag([word])[0][1]  #get the POS tag
    
    if tag in FUNCTION_TAGS:
        return "function"
    else:
        return "content"
    
def extract_word(dir):
    csvs = os.listdir(dir)
    print ("\n",csvs[0])
    filename = csvs[0]

    if not filename.endswith(".csv"):  # Skip non-CSV files
        print ("error1:", filename)
        return None
        
    #extract only the word from each file name
    filename = re.sub(r'^(content|function)', '', filename)
    matches = re.findall(r'[a-zA-Z]+', filename)

    if matches:
        return matches[0]
    else:
        print ("error2:", filename)  # If no word is found

    return None

def main():
    folders = ["content", "function"]
    pariticpants = 1
    contentErr=0
    functionErr=0

    for folder in folders:
        print ("\n\n\n",folder)
        for i in range (1,pariticpants+1):
            directory = "/home/greg/Documents/GregCode/dataSets/spectrogramDataBig1channel/"+folder+"/"+str(i)+"/"
            word_folders = os.listdir(directory)  # List all files and folders
            print (len(word_folders))
            for word_folder in word_folders:
                sourceFolder = os.path.join(directory, word_folder)
                word = extract_word(sourceFolder)
                wordType = classify_word_nltk(word)
                if wordType!=folder:
                    if folder=="content":
                        contentErr=contentErr+1
                    elif folder=="function":
                        functionErr=functionErr+1
                    else:
                        print ("cant count error!")

    print ("Content misclassification:", contentErr/1077)
    print ("Function misclassification:", functionErr/793)

main()