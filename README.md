# NER-on-Telugu-language.
This algorithm implements word embedding–based named entity recognition (NER). Words are assigned POS tags and we try to figure out the type of named entity based on the pattern in which the POS tags occur. 
1. Run extract.py: Derive the word embeddings using a document into
a new file “process.txt”. Middle element of the 5 tags correspond to
the current NER tag, first 2 elements denote already seen tags. Next 2
elements denote upcoming(next) POS tags.
2. Run training.py:
 i) Create 2 dictionaries containing NER tags and POS tags.
 ii) Create training and test datasets from "process.txt". The values in
the dataset will be weights assigned according to tags(POS & NER).
 iii) Save the datasets as numpy files.
3. Run fitting.py:
 i) Load training and test files.
 ii) Create an ANN model with 1 input layer and 1 output layer.
 iii) Compile the model with Stochastic Gradient Descent optimizer.
 iv) Fit the model to training set and evaluate on test set. Gives an
accuracy of 94% and loss of 26%.
 v) Save the model as ANN and use it on test sets.(any pickle files)
 vi) For the given test datasets, 92.25% accuracy has been achieved.
