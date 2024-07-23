"""\
------------------------------------------------------------
USE: python <PROGNAME> (options) file1...fileN
OPTIONS:
    -h : print this help message
    -s FILE : use stoplist file FILE
    -p : use Porter stemming (default: no stemming)
    -b : use BINARY weights (default: count weighting)
------------------------------------------------------------\
"""

import sys, re, getopt
import glob
from nltk.stem import PorterStemmer

opts, args = getopt.getopt(sys.argv[1:], 'hs:pbI:')
opts = dict(opts)

##############################
# HELP option

if '-h' in opts:
    progname = sys.argv[0]
    progname = progname.split('/')[-1] # strip out extended path
    help = __doc__.replace('<PROGNAME>', progname, 1)
    print(help, file=sys.stderr)
    sys.exit()

##############################
# Identify input files, when "-I" option used

if '-I' in opts:
    filenames = glob.glob(opts['-I'])
else:
    filenames = args

# Check if filenames are being found 
# (comment out after checking)
print('INPUT-FILES:', filenames, file=sys.stderr)

##############################
# STOPLIST option

stops = set()
if '-s' in opts:
    with open(opts['-s'], 'r') as stop_fs:
        for line in stop_fs :
            stops.add(line.strip())
            
##############################
# Stemming function

stemmer = PorterStemmer().stem

def stem_word(word):
    return stemmer(word)

##############################
# COUNT-WORDS function. 
# Takes 2 inputs: 1= FILE-NAME, 2= stoplist
# Returns a dictionary of word counts

#wordsF = filter(
        #lambda ThisWord: not re.match('^(?:(?:[0-9]{2}[:\/,]){2}[0-9]{2,4}|am|pm)$', ThisWord), words)

def count_words(filename, stops):
    counts = {}
    readFile = open(filename, 'r')
    for line in readFile:
        #\b is for the word boundry(space before)
        #(?![0-9]+\b) If a word consists solely of numbers that word is removed, this allows numbers between alphabatic char
        #\w+ is for removing any special char
        #\b is for the second boundary at the end of the word(space after)
        words = re.findall(r'\b(?![0-9]+\b)\w+\b', line.lower())
        #words = line.lower().split()
        for word in words:
            if word not in stops:
                #this line checks if a command for stemming is detected and then stems all the words in the.
                if '-p' in opts:
                    word = stem_word(word)
                if word in counts:
                    counts[word] += 1
                else:
                    counts[word] = 1
    return counts

##############################
# Compute counts for individual documents

docs = [ ]

for infile in filenames:
    docs.append(count_words(infile, stops))

##############################
# Compute similarity score for document pair
# Inputs are dictionaries of counts for each doc
# Returns similarity score

#overlap = 0
            #for w in doc1:
                #if w in doc2:
                    #overlap += 1
            #return overlap / (size1 + size2 - overlap)

def jaccard(doc1, doc2):
    size1 = len(doc1)
    size2 = len(doc2)
    intersec = set(doc1) & set(doc2)
    union = set(doc1) | set(doc2)
    if size1 + size2 > 0: # avoid a divide-by-zero error
        if '-b' in opts:
            return len(intersec)/len(union)
        else:
            minSum= 0
            maxSum= 0
            for w in union:
                if w not in doc1:
                    doc1[w] = 0
                if w not in doc2:
                    doc2[w] = 0
                minSum += min(doc1[w], doc2[w])
                maxSum += max(doc1[w], doc2[w])
            return minSum / maxSum
    else:
        return 0.0

##############################
# Compute scores for all document pairs

results = {}
for i in range(len(docs)-1):
    for j in range(i+1, len(docs)):        
        pair_name = '%s <> %s' % (filenames[i], filenames[j])
        results[pair_name] = jaccard(docs[i], docs[j])

##############################
# Sort, and print top N results

top_N = 20

pairs = sorted(results, key=results.get, reverse = True)[:top_N] # DUMMY CODE LINE 
# Replace with code to sort results based on scores.
# Have only results for highest "top_N" scores printed.





# Printing
c = 0
for pair in pairs:
    c += 1
    print('[%d] %s = %.3f' % (c, pair, results[pair]), file=sys.stdout)

##############################

#print(count_words("News/news01.txt", "stop_list.txt"))

