
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
print(dictionary["electronics"])
            
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

def accuracy(key, posList):
    countOfKey = posList.get(key)
    sumVar = sum(posList.values())
    ac = (countOfKey/sumVar)*100
    return ac 

poslist = pos(dictionary)
print(accuracy("NN", poslist))



def word_count(filename, target_word):
    # Initialize a count variable
    count = 0

    try:
        # Open the file in read mode
        with open(filename, 'r') as file:
            # Read the contents of the file
            words1 = []
            for line in file:
                words = line.split(" ")
                for word in words:
                    splited = word.split("/")
                    wordNeeded = splited[1]
                    words1.append(wordNeeded)
                # Count occurrences of the target word
            count = words1.count(target_word)
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred:", e)

    return count

# Example usage
file_path = 'training_data.txt'  # Replace with your file path
search_word = 'NNP'

result_count = word_count(file_path, search_word)
print(f"The word '{search_word}' occurs {result_count} times in the document.")
