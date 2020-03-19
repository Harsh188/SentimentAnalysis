"""
Authors: Harshith MohanKumar & Swarupa
Date: 16/3/2020
analyzer.py
Description: This program aims to build a sentiment analyzer which can predict the mood
of a given text. It will classify the text into neutral, positive or negative.
"""

import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#Lets first geth the file paths for the train and test data so we can open it
test_file_path = os.path.join(os.getcwd(),'data/movie_data/full_test.txt')
train_file_path = os.path.join(os.getcwd(),'data/movie_data/full_train.txt')

#testing purposes
# print(test_file_path)
# print(train_file_path)

# STEP 1. Open the Test and Train files and read them into an array
reviews_train = []
for line in open(test_file_path, 'r'):
    reviews_train.append(line.strip())
    
reviews_test = []
for line in open(train_file_path, 'r'):
    reviews_test.append(line.strip())

#Testing purposes
# print(reviews_train)

# STEP 2. Clean the data by removing all unnecessary symbols and punctuation. 
#	Also reomove all stop words (characters below len 3 are filtered out)
stop_words = set(stopwords.words('english'))
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
# print(len(stop_words))
def preprocess_reviews(reviews):
	reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
	reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
	reviews = [re.sub(r'\b\w{1,3}\b', '', line) for line in reviews]
	return reviews

reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)

#testing purposes
# print(reviews_train_clean)
# print(reviews_test_clean)

# STEP 4. neutralize the data by feeding it through a count vectorizer function
cv = CountVectorizer(binary=True)
cv.fit(reviews_train_clean)
X = cv.transform(reviews_train_clean)
X_test = cv.transform(reviews_test_clean)

# STEP 5. Now we will split the data and train it based on negative and positve words
#	In order to do this we have set the data in a way such that the first 12,500 reviews
#		are positive and the next 12,500 are negative.
#	Therefore the first 12,500 are given a 1 for being positive and the rest a 0 for negative
target = [1 if i < 12500 else 0 for i in range(25000)]

X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = 0.75
)

# STEP 6. Traning the model using Logistic regression
# Fist we will figure out which C value results in the most amount of accuracy
# This will get rid of all the outliers.
for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    # print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))

# Using the most accurate C value we will train the final model
final_model = LogisticRegression(C=0.05)
final_model.fit(X, target)
print ("Final Accuracy: %s" 
       % accuracy_score(target, final_model.predict(X_test)))

# STEP 7. Now we will store the data in a dictionary with the word being the key 
# and the value being it's accuracy coefficient.
feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), final_model.coef_[0]
    )
}

# For testing purposes
for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:5]:
    print (best_positive)
    
for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:5]:
    print (best_negative)

# print (feature_to_coef.items())

# STEP 9. Accepting user input and determining a tone scale
while(True):
	print("\n\n\n")
	s = input("Enter paragraph to check tone: Enter 'quit' to quit:\n")
	if s =='quit':
		break
	words = s.lower().split()
	polarity = 0
	ctr=0
	for w in words:
		if len(w)>3:
			if w in feature_to_coef:
				polarity+=feature_to_coef[w]
				ctr+=1
	polarity=(polarity/ctr)*100
	print("\n\nThe tone of your speech is: ",polarity)
