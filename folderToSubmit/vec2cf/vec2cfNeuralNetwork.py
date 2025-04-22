import os
import re
import nltk
from nltk import pos_tag
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")

#loading the data

folder = "data/"     ##change this to look at different runs
train_path = os.path.join(folder, "train_data.npz")
test_path = os.path.join(folder, "test_data.npz")
words_path = os.path.join(folder, "words_labels.npz")

if not (os.path.exists(train_path) or os.path.exists(test_path)):
    print("Model path does not exist!")
    exit()

train_data = np.load(train_path, allow_pickle=True)  
X_train = train_data["predictions_train"] #vectors

test_data = np.load(test_path, allow_pickle=True)  
X_test = test_data["predictions_test"] #vectors

words = np.load(words_path, allow_pickle=True)  
train_words = words["train_words"] #words
valid_words = words["valid_words"] #words
test_words = words["test_words"] #words

train_words = np.concatenate([train_words, valid_words], axis=0)

FUNCTION_TAGS = {
    "DT", "CC", "IN", "PRP", "PRP$", "TO", "MD", "WP", "WRB", "EX"
}

#should now have
#   X_train     train_words
#   X_test      valid_words

train_words = [re.sub(r'\d+$', '', word) for word in train_words] #remove the trailing numbers
test_words = [re.sub(r'\d+$', '', word) for word in test_words] #remove the trailing numbers


def classifyCF(word):
    word = word.lower()  #convert word to lowercase
    
    tag = pos_tag([word])[0][1]  #get the POS tag
    
    if tag in FUNCTION_TAGS:
        return 0 #function
    else:
        return 1 #content

y_train=np.array([classifyCF(word) for word in train_words])
y_test=np.array([classifyCF(word) for word in test_words])

#ML Method: neural Network
#overwriting x_train and x_test to get baseline results, this should normally be commented out
testTrain_path = os.path.join(folder, "testTrain_baseline.npz")
X_testTrain = np.load(testTrain_path, allow_pickle=True)
X_train = X_testTrain["train_baseline"]
X_test = X_testTrain["test_baseline"]

param_grid = {
    'hidden_layer_sizes': [(64, 64), (92), (128, 64, 32), (256,128), (512,256,128)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'max_iter': [300, 500]
}

# Set up the MLP and GridSearchCV
mlp = MLPClassifier(random_state=42)
grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)

# Run the search
grid_search.fit(X_train, y_train)

# Get best model and evaluate
best_mlp = grid_search.best_estimator_
y_pred_best = best_mlp.predict(X_test)
print("Best Params:", grid_search.best_params_)
print("Best Neural Net Accuracy:", accuracy_score(y_test, y_pred_best))


