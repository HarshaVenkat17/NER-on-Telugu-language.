import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt

#load datasets
y_train = np.load("./TEST-1//y_train.npy")
x_train = np.load("./TEST-1//x_train.npy")
x_test = np.load("./TEST-1//x_test_n.npy")
y_test = np.load("./TEST-1//y_test_n.npy")

#Create an ANN
model = Sequential()
#create an input layer
model.add(Dense(60,input_dim=120,activation="relu"))#input_dim: 24(tags)*5(n-gram)=120
#create an output layer
model.add(Dense(19,activation="sigmoid"))
# create an instance of SGD optimizer and compile the model.
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
#fit the model to trainin g set
model.fit(x_train,y_train,epochs=700,batch_size=1000,verbose=1,validation_split=0.2)

#Evaluate on test set and save the model.
score = model.evaluate(x_test,y_test,batch_size=120)
model.save("ANN")
print(score)

#load the model
model = load_model('ANN')
#load the test datasets
x1=np.load("./TEST-1//x_test.npy")
y1=np.load("./TEST-1//y_test.npy")
#Predict the result and find the loss and accuracy.
score = model.evaluate(x1, y1, verbose=0)

y2=np.argmax(model.predict(x1),axis=1)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

NER_dict = {1:"B-PERSON",2:"I-PERSON",3:"B-ORG",4:"I-ORG",5:"B-LOC",6:"I-LOC",7:"B-NUM",8:"I-NUM",9:"B-TIME",10:"I-TIME",
            11:"B-DAY",12:"B-MONEY",13:"I-MONEY",14:"B-DATE",15:"I-DATE",16:"B-PERIOD",17:"I-PERIOD",18:"B-YEAR",19:"O"}
tags=[]
for i in range(len(y2)):
    tags.append(NER_dict[y2[i]+1])
#print(tags)    