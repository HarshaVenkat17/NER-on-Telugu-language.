"""Create dataset "process.txt" that contains 1 NER tag and 5 POS-tags in each line.
Middle element of the 5 tags correspond to the current NER tag, other elements are used for context."""
import pandas as pd
df1 = pd.read_csv("./Data//NER_data.txt",sep="\s+",header=None,dtype=str)

with open("./Data//process.txt","w+") as file:
    for i in range(df1.shape[0]):
        arr = ""
        if i-2>=0:
            arr = arr + " " + str(df1.iloc[i-2,2])
        else:
            arr = arr + " " + "0"
        if i-1>=0:
            arr = arr + " " + str(df1.iloc[i-1,2])
        else:
            arr = arr + " " + "0"
        arr = arr + " " + str(df1.iloc[i,2])
        if i+1<df1.shape[0]:
            arr = arr + " " + str(df1.iloc[i+1,2])
        else:
            arr = arr + " " + "0"
        if i+2<df1.shape[0]:
            arr = arr + " " + str(df1.iloc[i+2,2])
        else:
            arr = arr + " " + "0"
        file.write(str(df1.iloc[i,1])+" " + arr.strip()+"\n")
