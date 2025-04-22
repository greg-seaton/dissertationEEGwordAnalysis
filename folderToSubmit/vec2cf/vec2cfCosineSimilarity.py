import re
import nltk
from nltk import pos_tag
from sklearn.metrics import accuracy_score
import numpy as np
import os
nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")

FUNCTION_TAGS = {
    "DT", "CC", "IN", "PRP", "PRP$", "TO", "MD", "WP", "WRB", "EX"
}

words_path = os.path.join("data", "words_labels.npz")
words = np.load(words_path, allow_pickle=True)  
x_test_words = words["test_words"] #words
x_test_words = [re.sub(r'\d+$', '', word) for word in x_test_words] #remove the trailing numbers

test_data = np.load("data/test_data.npz", allow_pickle=True)  
predictions = test_data["predictions_test"] #predicted vectors to test vs avgContent and avgFunction vectors

#this overwrites predictions to test the base on true labels (vectors)
# baseline_path = os.path.join("data", "testTrain_baseline.npz")
# words = np.load(baseline_path, allow_pickle=True)  
# predictions = words["test_baseline"] #words

avgContentVec = np.array([
     1.6113760e-02, -2.7301470e-02, -4.2580515e-02,  3.3281676e-02,
    -7.0191650e-03, -3.2150541e-03,  8.4313788e-03, -1.6408065e-02,
     4.4093490e-02, -9.3304804e-03, -2.5781933e-03,  2.3656979e-02,
    -2.2654904e-02, -2.5810283e-02, -6.7046769e-03,  2.0750238e-02,
    -4.8571322e-02,  1.7146017e-02,  4.2861439e-02, -1.3733760e-02,
    -4.7471970e-02, -2.9108115e-03, -4.0479001e-02, -1.5264681e-02,
    -3.0475313e-02,  7.1197296e-03,  6.8012993e-03,  5.4742835e-02,
    -7.0938841e-03,  3.6199996e-03,  6.1321440e-03, -4.3329254e-02,
     1.0385245e-02, -9.9380370e-03,  5.3730230e-03, -1.9063262e-02,
     3.8970059e-03, -4.5011450e-02,  1.5936576e-02,  6.9743195e-03,
     4.4736732e-02,  5.0976381e-02, -1.6193917e-02,  3.7708446e-02,
    -2.9769463e-02,  6.3082562e-03,  7.8464290e-03,  4.5936339e-02,
     4.5730867e-03,  5.4335508e-02, -2.1407390e-05,  9.4636548e-03,
    -2.4255974e-02, -7.2123870e-02,  2.6899420e-02,  1.3408467e-01,
     1.6282141e-02,  1.4706583e-02, -1.0815738e-01, -5.9308060e-02,
     1.1803578e-02, -5.1881347e-02,  2.5552213e-02, -9.7079910e-03,
    -6.3689329e-02,  1.0080665e-02, -4.0080469e-02, -4.1629918e-02,
    -3.2871503e-02, -9.3659759e-03,  1.7968511e-02,  2.9740646e-02,
     2.3033427e-02,  5.8206391e-02, -1.9986197e-02,  3.7314128e-03,
     9.4836010e-03, -6.8239188e-03,  8.1913367e-02,  8.4258299e-03,
    -6.6716276e-02, -1.3339374e-02,  3.0618263e-02, -1.3972881e-02,
     8.3867215e-02,  1.5953898e-02,  3.6695725e-03,  2.1234388e-02,
    -2.0940816e-02,  4.3375842e-02,  4.9595661e-03,  2.0830126e-02,
     2.2962689e-02, -3.1962495e-02,  5.8184251e-02, -2.0664297e-02,
     2.3749661e-02,  3.8182609e-02, -6.7233182e-02, -2.7060395e-03
])

avgFunctionVec = np.array([
    0.01242405, -0.04524031, -0.04850681,  0.02916224, -0.00683834, -0.02153979,
    0.00984589, -0.01555265,  0.05197541, -0.02102798, -0.00473367,  0.02173241,
   -0.03011763, -0.02083798, -0.01698121,  0.01995791, -0.0449665,   0.02318156,
    0.04003574, -0.01163875, -0.05914337, -0.0065301,  -0.04403298, -0.01504537,
   -0.04286955,  0.02076024,  0.0052991,   0.05010216, -0.00779848,  0.01359448,
    0.00556209, -0.04135015,  0.0173913,  -0.00076846,  0.01088278, -0.02440821,
    0.0162266,  -0.04041993,  0.02270043, -0.00286986,  0.03876036,  0.05813753,
   -0.0119468,   0.03857458, -0.03245796,  0.00969075,  0.00739152,  0.04819891,
    0.01324301,  0.07134043,  0.00234784,  0.01545194, -0.02700345, -0.08603726,
    0.02380971,  0.13184959,  0.01084673,  0.02095599, -0.11973975, -0.05707669,
    0.02227229, -0.06325232,  0.03157363, -0.01617456, -0.06236492,  0.0143396,
   -0.03877049, -0.03986187, -0.04335722,  0.0012139,   0.02495528,  0.03211166,
    0.02265753,  0.05503268, -0.00700555,  0.00298321,  0.01733664, -0.00658866,
    0.08637135,  0.01022197, -0.06659757, -0.01777261,  0.02557676, -0.02779806,
    0.09666562,  0.01089516, -0.01587794,  0.01467959, -0.01208429,  0.04943546,
    0.01725956,  0.00970297,  0.03211832, -0.02289558,  0.06545261, -0.01555047,
    0.02981459,  0.0399132,  -0.0788419,  -0.00250498
])


def cosine_similarity(y_true, y_pred):
    dot_product = np.sum(y_true * y_pred, axis=-1)
    norm_true = np.linalg.norm(y_true, axis=-1)
    norm_pred = np.linalg.norm(y_pred, axis=-1)
    return dot_product / (norm_true * norm_pred)


def classifyCF(word):
    word = word.lower()  #convert word to lowercase
    
    tag = pos_tag([word])[0][1]  #get the POS tag
    
    if tag in FUNCTION_TAGS:
        return "F"
    else:
        return "C"

cf_labels_pred = []

for prediction in predictions:
    if (cosine_similarity(prediction,avgContentVec) > cosine_similarity(prediction,avgFunctionVec)):
        cf_labels_pred.append("C")
    else:
        cf_labels_pred.append("F")

cf_labels_true = []

for word in x_test_words:
    cf_labels_true.append(classifyCF(word))

print (cf_labels_true)

print (accuracy_score(cf_labels_true,cf_labels_pred))





 