import nltk
import json
import math
import sys
from mpi4py import MPI
from nltk.tokenize import sent_tokenize, word_tokenize

def exit_with_error(message):
    print(message)
    MPI.Finalize()
    sys.exit(1)

def determine_sentiment(positive, negative):
    if positive <= 0.1 and negative >= 0.1:
        return "negative"
    elif negative <= 0.1 and positive >= 0.1:
        return "positive"
    elif positive > negative and (positive - negative) >= 0.1:
        return "positive"
    elif negative > positive and (negative - positive) >= 0.1:
        return "negative"
    elif positive == 0 and negative < 0.1:
        return "neutral"
    elif negative == 0 and positive < 0.1:
        return "neutral"
    else:
        return "neutral"

def main(file_paths):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if len(file_paths) != size:
        exit_with_error("The number of files must be equal to the number of processes.")
    
    if rank == 0:
        nltk.download('punkt')
        
    MPI.COMM_WORLD.Barrier()
    
    with open(file_paths[rank], 'r', encoding='utf-8') as file:
        text = file.read()
        
    words = word_tokenize(text)
    words = [word.capitalize() for word in words]
    
    try:
        with open('data.json', 'r') as file:
            conotation_words_dict = json.load(file)
        print("Data successfully read from data.json by process", rank)
    except FileNotFoundError:
        exit_with_error("The file was not found.")
    except json.JSONDecodeError:
        exit_with_error("Error decoding JSON.")
    
    conotation_words = [0] * 2
    
    for word in words:
        if word in conotation_words_dict:
            if conotation_words_dict[word]:
                conotation_words[0] += 1
            else:
                conotation_words[1] += 1
    
    tf_p = conotation_words[0] / len(words)
    tf_n = conotation_words[1] / len(words)
    
    data_local = [conotation_words[0], conotation_words[1], len(words), tf_p, tf_n]
    
    data = []
    data = comm.gather(data_local, root=0)
    
    if rank == 0:
        word_count = sum([d[2] for d in data])
        
        positive = 0
        negative = 0
        for data_list in data:
            data_tuple = tuple(data_list)
            positive_words, negative_words, words_count, tf_p, tf_n = data_tuple
            idf_p = 0
            idf_n = 0
            
            if positive_words > 0:
                idf_p = math.log(word_count / positive_words)
                
            if negative_words > 0:
                idf_n = math.log(word_count / negative_words)
                
            positive += tf_p * idf_p
            negative += tf_n * idf_n
            
        sentiment = determine_sentiment(positive, negative)
        
        print("Positive: ", positive)
        print("Negative: ", negative)
        print("Sentiment: ", sentiment)
        
    MPI.Finalize()

if __name__ == "__main__":
    main(sys.argv[1:])