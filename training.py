import pandas as pd
import numpy as np
df1 = pd.read_csv("./Data//process.txt",sep=" ",header=None,dtype=np.dtype("str"))
#divide the dataset into dependent and independent variables
y_train = df1.iloc[:,0].copy().values
x_train = df1.iloc[:,1:].copy().values

#create arrays of numpy 0s to store the data obtained from the tags 
y_train_ann = np.zeros((y_train.shape[0],19))#19 NER tags
x_train_ann = np.zeros((x_train.shape[0],120))#24(POS-tags)*5(n-gram)=120
NER_dict = {"B-PERSON":1,"I-PERSON":2,"B-ORG":3,"I-ORG":4,"B-LOC":5,"I-LOC":6,"B-NUM":7,"I-NUM":8,"B-TIME":9,"I-TIME":10,"B-DAY":11,"B-MONEY":12,"I-MONEY":13,"B-DATE":14,"I-DATE":15,
            "B-PERIOD":16,"I-PERIOD":17,"B-YEAR":18,"O":19}
POS_dict = {"NN":1,"NST":2,"NNP":3,"PRP":4,"DEM":5,"VM":6,"JJ":7,"RB":8,"PSP":9,"RP":10,"CC":11,"WQ":12,"QF":13,"QC":14,"QO":15,
                "CL":16,"INTF":17,"INJ":18,"UT":19,"SYM":20,"RDP":21,"0":22,"SYMP":23,"NNO":24}

# set element corresponding to the NER-tag as 1
for i,j in enumerate(y_train):
    y_train_ann[i,NER_dict.get(str(j))-1]=1

#set weights for tags in every line as follows: 0th and 5th elements are assigned 0.25, 1st and 4th element are assigned 0.125 
#middle element is assigned weight 1, its adjacent elements are assigned 0.125 and the extreme tags are assigned 0.25.
for i,j in enumerate(x_train):
    for a,k in enumerate(j):
        if np.abs(a-2)==2:
            weight = 0.25
        elif np.abs(a-2)==1:
            weight= 0.125
        else:
            weight =1;
        x_train_ann[i,24*a+(POS_dict.get(str(k))-1)]=weight
        
# save the arrays as numpy files
np.save("./TEST-1//y_train.npy",y_train_ann[:50000,:])
np.save("./TEST-1//x_train.npy",x_train_ann[:50000,:])
np.save("./TEST-1//y_test_n.npy",y_train_ann[50000:,:],allow_pickle=True)
np.save("./TEST-1//x_test_n.npy",x_train_ann[50000:,:],allow_pickle=True)
