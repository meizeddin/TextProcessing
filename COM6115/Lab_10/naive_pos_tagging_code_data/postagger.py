"""
USE: python <PROGNAME> (options) 
OPTIONS:
    -h : print this help message and exit
    -d FILE : use FILE as data to create a new lexicon file
    -t FILE : apply lexicon to test data in FILE
"""
################################################################

import sys, re, getopt

################################################################
# Command line options handling, and help

opts, args = getopt.getopt(sys.argv[1:], 'hd:t:')
opts = dict(opts)

def printHelp():
    progname = sys.argv[0]
    progname = progname.split('/')[-1] # strip out extended path
    help = __doc__.replace('<PROGNAME>', progname, 1)
    print('-' * 60, help, '-' * 60, file=sys.stderr)
    sys.exit()
    
if '-h' in opts:
    printHelp()

if '-d' not in opts:
    print("\n** ERROR: must specify training data file (opt: -d FILE) **", file=sys.stderr)
    printHelp()

if len(args) > 0:
    print("\n** ERROR: no arg files - only options! **", file=sys.stderr)
    printHelp()

################################################################

dictionary = {}
with open("training_data.txt", "r") as f:
    for line in f:
        words = line.split()
        for word in words:
            dictionary2 = {}
            splitWord = word.split("/")
            key = splitWord[0]
            value = splitWord[1]
            count = 1
            if key not in dictionary:
                dictionary2[value] = count
                dictionary[key] = dictionary2
            else:
                check = dictionary[key]
                if value in check:
                    check[value] += 1
                    dictionary[key] = check
                else:
                    check[value] = count
print(dictionary["add"])
            
poslist = {}
def pos(dictionary):
    for key1 in dictionary:
        dic2 = dictionary.get(key1)
        for key in dic2:  # Loop through keys in dic2
            value = dic2.get(key)
            if key not in poslist:
                poslist[key] = value
            else:
                poslist[key] += value
    sorted_dict = {k: v for k, v in sorted(poslist.items(), key=lambda item: item[1], reverse=True)}
    return sorted_dict


lis = pos(dictionary)
print(lis)

