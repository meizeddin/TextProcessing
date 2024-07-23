#!/usr/bin/env python
import re, random, math, collections, itertools

PRINT_ERRORS=0

#------------- Function Definitions ---------------------


def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia):

    #reading pre-labeled input and splitting into lines
    posSentences = open('rt-polarity.pos', 'r', encoding="ISO-8859-1")
    posSentences = re.split(r'\n', posSentences.read())

    negSentences = open('rt-polarity.neg', 'r', encoding="ISO-8859-1")
    negSentences = re.split(r'\n', negSentences.read())

    posSentencesNokia = open('nokia-pos.txt', 'r')
    posSentencesNokia = re.split(r'\n', posSentencesNokia.read())

    negSentencesNokia = open('nokia-neg.txt', 'r', encoding="ISO-8859-1")
    negSentencesNokia = re.split(r'\n', negSentencesNokia.read())

    with open('positive-words.txt', 'r', encoding="ISO-8859-1") as posDictionary:
        posWordList = []
        for line in posDictionary:
            if not line.startswith(';'):
                posWordList.extend(re.findall(r"[a-z\-]+", line))
    posWordList.remove('a')

    with open('negative-words.txt', 'r', encoding="ISO-8859-1") as negDictionary:
        negWordList = []
        for line in negDictionary:
            if not line.startswith(';'):
                negWordList.extend(re.findall(r"[a-z\-]+", line))

    for i in posWordList:
        sentimentDictionary[i] = 1
    for i in negWordList:
        sentimentDictionary[i] = -1

    #create Training and Test Datsets:
    #We want to test on sentences we haven't trained on, to see how well the model generalses to previously unseen sentences

  #create 90-10 split of training and test data from movie reviews, with sentiment labels    
    
    # adding a random seed just to fix the calculation in order to produce the cleanest code for classification calculations.
    random.seed(42)
    for i in posSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="positive"
        else:
            sentencesTrain[i]="positive"

    for i in negSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="negative"
        else:
            sentencesTrain[i]="negative"

    #create Nokia Datset:
    for i in posSentencesNokia:
            sentencesNokia[i]="positive"
    for i in negSentencesNokia:
            sentencesNokia[i]="negative"

#----------------------------End of data initialisation ----------------#

#calculates p(W|Positive), p(W|Negative) and p(W) for all words in training data
def trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord):
    posFeatures = [] # [] initialises a list [array]
    negFeatures = [] 
    freqPositive = {} # {} initialises a dictionary [hash function]
    freqNegative = {}
    dictionary = {}
    posWordsTot = 0
    negWordsTot = 0
    allWordsTot = 0

    #iterate through each sentence/sentiment pair in the training data
    for sentence, sentiment in sentencesTrain.items():
        wordList = re.findall(r"[\w']+", sentence)
        
        for word in wordList: #calculate over unigrams
            allWordsTot += 1 # keeps count of total words in dataset
            if not (word in dictionary):
                dictionary[word] = 1
            if sentiment=="positive" :
                posWordsTot += 1 # keeps count of total words in positive class

                #keep count of each word in positive context
                if not (word in freqPositive):
                    freqPositive[word] = 1
                else:
                    freqPositive[word] += 1    
            else:
                negWordsTot+=1# keeps count of total words in negative class
                
                #keep count of each word in positive context
                if not (word in freqNegative):
                    freqNegative[word] = 1
                else:
                    freqNegative[word] += 1

    for word in dictionary:
        #do some smoothing so that minimum count of a word is 1
        if not (word in freqNegative):
            freqNegative[word] = 1
        if not (word in freqPositive):
            freqPositive[word] = 1

        # Calculate p(word|positive)
        pWordPos[word] = freqPositive[word] / float(posWordsTot)

        # Calculate p(word|negative) 
        pWordNeg[word] = freqNegative[word] / float(negWordsTot)

        # Calculate p(word)
        pWord[word] = (freqPositive[word] + freqNegative[word]) / float(allWordsTot) 

#---------------------------End Training ----------------------------------

#implement naive bayes algorithm
#INPUTS:
#  sentencesTest is a dictonary with sentences associated with sentiment 
#  dataName is a string (used only for printing output)
#  pWordPos is dictionary storing p(word|positive) for each word
#     i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
#  pWordNeg is dictionary storing p(word|negative) for each word
#  pWord is dictionary storing p(word)
#  pPos is a real number containing the fraction of positive reviews in the dataset
def testBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord, pPos):
    pNeg=1-pPos

    #These variables will store results
    total=0
    correct=0
    totalpos=0
    totalpospred=0
    totalneg=0
    totalnegpred=0
    correctpos=0
    correctneg=0

    #for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentencesTest.items():
        wordList = re.findall(r"[\w']+", sentence)#collect all words

        pPosW=pPos
        pNegW=pNeg

        for word in wordList: #calculate over unigrams
            if word in pWord:
                if pWord[word]>0.00000001:
                    pPosW *=pWordPos[word]
                    pNegW *=pWordNeg[word]

        prob=0;            
        if pPosW+pNegW >0:
            prob=pPosW/float(pPosW+pNegW)


        total+=1
        if sentiment=="positive":
            totalpos+=1
            if prob>0.5:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("ERROR (pos classed as neg %0.2f):" %prob + sentence)
        else:
            totalneg+=1
            if prob<=0.5:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("ERROR (neg classed as pos %0.2f):" %prob + sentence)
    print(dataName)
    performanceMetrics(total, correct, totalpos, totalneg, correctpos, correctneg, totalpospred, totalnegpred)
# TODO for Step 2: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
# (3) precision and recall for the negative class; (4) F1 score;
 
def performanceMetrics(total, correct, totalpos, totalneg, correctpos, correctneg, totalpospred, totalnegpred):
    # accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = (correct) / (total)

    # Positive class calculations

    # precision = (TP) / (TP + FP)
    precision_pos = (correctpos) / (totalpospred)
    # recall = (TP) / (TP + FN)
    recall_pos = (correctpos) / (totalpos)
    # F1 score = 2 * (precision * recall) / (precision + recall)
    f1_score_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos)

    #Negative class calculations
    
    # precision = (TN) / (TN + FN)
    precision_neg = (correctneg) / (totalnegpred)
    # recall = (TN) / (TN + FP)
    recall_neg = (correctneg) / (totalneg)
    # F1 score = 2 * (precision * recall) / (precision + recall)
    f1_score_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg)

    # Rounding the results
    accuracy = round(accuracy, 4)

    precision_pos = round(precision_pos, 4)
    recall_pos = round(recall_pos, 4)
    f1_score_pos = round(f1_score_pos, 4)

    precision_neg = round(precision_neg, 4)
    recall_neg = round(recall_neg, 4)
    f1_score_neg = round(f1_score_neg, 4)

    print("The accuracy of the model = ", accuracy, " ≈ ", "{:.4g}".format((accuracy * 100)), "%\n")
    print("Positive class Caculations:")
    print("The precision of [positive] prediction = ", precision_pos, " ≈ ", "{:.4g}".format((precision_pos * 100)), "%")
    print("The recall of [positive] prediction = ", recall_pos, " ≈ ", "{:.4g}".format((recall_pos * 100)), "%")
    print("The F1_score of [positive] prediction = ", f1_score_pos, " ≈ ", "{:.4g}".format((f1_score_pos * 100)), "%\n")
    print("Negative class Caculations:")
    print("The precision of [negative] prediction = ", precision_neg, " ≈ ", "{:.4g}".format((precision_neg * 100)), "%")
    print("The recall of [negative] prediction = ", recall_neg, " ≈ ", "{:.4g}".format((recall_neg * 100)), "%")
    print("The F1_score of [negative] prediction = ", f1_score_neg, " ≈ ", "{:.4g}".format((f1_score_neg * 100)), "%\n")
    print("_________________________________________________________")



# This is a simple classifier that uses a sentiment dictionary to classify 
# a sentence. For each word in the sentence, if the word is in the positive 
# dictionary, it adds 1, if it is in the negative dictionary, it subtracts 1. 
# If the final score is above a threshold, it classifies as "Positive", 
# otherwise as "Negative"
def testDictionary(sentencesTest, dataName, sentimentDictionary, threshold):
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        for word in Words:
            if word in sentimentDictionary:
               score+=sentimentDictionary[word]
 
        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("ERROR (pos classed as neg %0.2f):" %score + sentence)
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("ERROR (neg classed as pos %0.2f):" %score + sentence)

    print(dataName)
    performanceMetrics(total, correct, totalpos, totalneg, correctpos, correctneg, totalpospred, totalnegpred)

    


#Print out n most useful predictors
def mostUseful(pWordPos, pWordNeg, pWord, n):
    predictPower={}
    for word in pWord:
        if pWordNeg[word]<0.0000001:
            predictPower[word]=1000000000
        else:
            predictPower[word]=pWordPos[word] / (pWordPos[word] + pWordNeg[word])
            
    sortedPower = sorted(predictPower, key=predictPower.get)
    head, tail = sortedPower[:n], sortedPower[len(predictPower)-n:]
    print ("NEGATIVE:")
    print (head)
    print ("\nPOSITIVE:")
    print (tail)
    return head + tail

# suggested fix for the mostUseful function, that better prints the data with more information:
def improved_mostUseful(pWordPos, pWordNeg, pWord, n):
    predictPower = {}
    for word in pWord:
        if pWordNeg[word] < 0.0000001:
            predictPower[word] = 1000000000
        else:
            predictPower[word] = pWordPos[word] / (pWordPos[word] + pWordNeg[word])

    sortedPower = sorted(predictPower, key=predictPower.get)
    head, tail = sortedPower[:n], sortedPower[len(predictPower) - n:][::-1]

    print("NEGATIVE:")
    for idx, (neg_word_1, neg_word_2) in enumerate(zip(head[:n // 2], head[n // 2:]), start=1):
        print(f"{idx}. {neg_word_1.ljust(20)} {predictPower[neg_word_1]:.4g} "
              f" \t{idx + n // 2}. {neg_word_2.ljust(20)} {predictPower[neg_word_2]:.4g}")

    print("\nPOSITIVE:")
    for idx, (pos_word_1, pos_word_2) in enumerate(zip(tail[:n // 2], tail[n // 2:]), start=1):
        print(f"{idx}. {pos_word_1.ljust(20)} {predictPower[pos_word_1]:.4g} "
              f" \t{idx + n // 2}. {pos_word_2.ljust(20)} {predictPower[pos_word_2]:.4g}")

    return head + tail

def most_useful_in_dic(sentimentDictionary, mostUsefulWords):
    count = 0
    mostUsefulWordsCount = len(mostUsefulWords)
    for word in mostUsefulWords:
        if word in sentimentDictionary:
            count += 1
    print("\nCount of the most useful words present in the sentiment dictionary = ", count)
    print("Percentage present in sentimentDictionary = ", "{:.4g}".format(((count / mostUsefulWordsCount) * 100)),"%\n")
    return count 

def most_useful_not_in_dic(sentimentDictionary, mostUsefulWords):
    count = 0
    list = []
    mostUsefulWordsCount = len(mostUsefulWords)
    for word in mostUsefulWords:
        if word not in sentimentDictionary:
            count += 1
            list.append(word)
    print("\nCount of the most useful words not present in the sentiment dictionary = ", count)
    print("Percentage not present in sentimentDictionary = ", "{:.4g}".format(((count / mostUsefulWordsCount) * 100)),"%\n")
    print("Words not present in the sentiment dictionary:")
    print(list, "\n")

    return list 


def testDictionaryImproved(sentencesTest, dataName, sentimentDictionary, threshold):

    negation_words = ["not", "no", "never", "none", "nobody", "nothing", "neither", "nowhere",
                      "hardly", "barely", "scarcely", "without", "cannot", "can't", "isn't",
                      "aren't", "won't", "don't", "didn't", "couldn't"]
    diminisher_words = ["little", "few", "slightly", "somewhat", "kind of", "sort of", "bit",
                    "marginally", "scarcely", "moderately", "hardly", "scarcely", "partially",
                    "inadequately", "insufficiently", "incompletely", "sporadically", "scantly",
                    "meagerly", "minimally", "negligibly", "sparsely", "barely", "middling"]
    emphasis_words = ["very", "extremely", "highly", "intensely", "remarkably", "especially",
                      "excessively", "astonishingly", "overwhelmingly", "wildly", "unbelievably",
                      "amazingly", "incredibly", "absolutely", "decidedly"]
                      
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        negation_flag = False
        for i, word in enumerate(Words):
            if word.lower() in negation_words:
                negation_flag = True
            if negation_flag and i + 1 < len(Words):
                next_word = Words[i + 1]
                if next_word in sentimentDictionary:
                    score += sentimentDictionary[next_word] * -1
                    negation_flag = False
            if word.lower() == "but":
                score = 0
            if word.lower() in diminisher_words:
                    score -= 1
            if word.lower() in emphasis_words:
                    next_word = Words[i + 1]
                    if next_word in sentimentDictionary:
                        score = sentimentDictionary[next_word] * 1.5
                    else:
                        score += 1
            if word in sentimentDictionary:
                    score += sentimentDictionary[word]
 
        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("ERROR (pos classed as neg %0.2f):\n" %score + sentence)
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("ERROR (neg classed as pos %0.2f):\n" %score + sentence)

    print(dataName)
    
    performanceMetrics(total, correct, totalpos, totalneg, correctpos, correctneg, totalpospred, totalnegpred)



sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}
sentencesNokia={}
#initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia)

pWordPos={} # p(W|Positive)
pWordNeg={} # p(W|Negative)
pWord={}    # p(W) 

#build conditional probabilities using training data
trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)

print ("Naive Bayes\n")

#testBayes(sentencesTrain,  "Films (Train Data, Naive Bayes):\t\n", pWordPos, pWordNeg, pWord,0.5)
#testBayes(sentencesTest,  "Films  (Test Data, Naive Bayes):\t\n", pWordPos, pWordNeg, pWord, 0.5)
#testBayes(sentencesNokia, "Nokia   (All Data,  Naive Bayes):\t\n", pWordPos, pWordNeg, pWord,0.7)

#print("The most useful word for classifing positive and negative sentiments: \n")
# print most useful words
mostUsefulWords = mostUseful(pWordPos, pWordNeg, pWord, 50)
mostUsefulWords = improved_mostUseful(pWordPos, pWordNeg, pWord, 50)
most_useful_in_dic(sentimentDictionary, mostUsefulWords)
most_useful_not_in_dic(sentimentDictionary, mostUsefulWords)

print ("Test Dictionary\n")

#run sentiment dictionary based classifier on datasets
#testDictionary(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, 1)
#testDictionary(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, 1)
#testDictionary(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, 1)

print ("Improved Test Dictionary\n")

#run sentiment dictionary based classifier on datasets
testDictionaryImproved(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, 0.5)
testDictionaryImproved(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, 0.5)
testDictionaryImproved(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, 0)



 