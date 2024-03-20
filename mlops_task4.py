from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords=stopwords.words('english')

data = pd.read_csv("labeled_data.csv")
print(data.head())

data["labels"] = data["class"].map({0:"Hate Speech", 1:"Offensive Speech", 2:"No Hate and Offensive Speech"})

data = data[["tweet", "labels"]]

data.head()

def clean(text):
    text = str(text).lower()
    text = re.sub('[,?_]', '', text)
    text = re.sub('https?://\S+|www.\S+', '', text)
    text = re.sub('<,?>+', '',text)
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub('\n','',text)
    text = re.sub('\w\d\w', '', text)
    text = [word for word in text.split(' ') if word not in stopwords]
    text = " ".join(text)
    return text

data["tweet"] = data["tweet"].apply(clean)
data['tweet'].head()

x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
x = cv.fit_transform(x)
X_train, X_text, y_train, y_test = train_test_split(x,y,test_size=0.30,random_state=42)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_text)
print(accuracy_score(y_test, y_pred))

i = "you are bad person"
i = cv.transform([i]).toarray()
print(model.predict((i)))