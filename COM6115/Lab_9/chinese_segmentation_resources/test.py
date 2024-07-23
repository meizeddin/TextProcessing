wordList = []
with open("chinesetrad_wordlist.utf8", "r", encoding= "utf-8") as f:
    for line in f:
        word = line.strip()
        wordList.append(word)

def segment(sent, wordSet):   # dummy definition provided, so can 
    sentLength = len(sent)
    sentence = sent
    sentList = []
    pos = 0

    for i in range(sentLength):
        maxWord = 5
        for j in range(5):
            word = sentence[pos:pos + maxWord]
            if word in wordSet:
                sentList.append(word)
                pos += len(word)
            else:
                maxWord -= 1 
    return sentList        # write code for main loop first.


sentence = "醫護人員很快的將她的媽媽抬上救護車"

print(segment(sentence, wordList))

with open("chinesetext.utf8", "r", encoding= "utf-8") as f:
    with open("MYRESULTS.utf8", "w", encoding= "utf-8") as output_file:
        for line in f:
            sent = line.strip()
            sentList = segment(sent, wordList)
            output_file.write(" ".join(sentList) + "\n")

