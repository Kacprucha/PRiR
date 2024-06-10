import nltk
import json
import math
import sys
from mpi4py import MPI
from nltk.tokenize import sent_tokenize, word_tokenize

def read_files(file_paths):
    sentences_with_file_numbers = []
    word_counts = []
    
    file_number = 0
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            
            sentences = sent_tokenize(text)  
            sentences_with_file_numbers.extend([(sentence.strip(), file_number) for sentence in sentences])
            
            words = word_tokenize(text)  
            word_counts.insert(len(word_counts), len(words))
        file_number += 1
    
    return sentences_with_file_numbers, word_counts

def divide_sentences_among_threads(sentences, num_threads, comm):
    chunk_size = len(sentences) // num_threads
    remainder = len(sentences) % num_threads
    
    divided_sentences = []
    result = []
    
    start = 0
    for i in range(num_threads):
        end = start + chunk_size + (1 if i < remainder else 0)
        
        if i == 0:
            result = sentences[start:end]
        else:
            divided_sentences = sentences[start:end]
            comm.send(divided_sentences, dest=i)
            divided_sentences = []
            
        start = end
        
    return result

def count_if_positives_are_in_file(number_of_conotation_words):
    result = 0
    for i in range(len(number_of_conotation_words)):
        if i % 2 == 0 and number_of_conotation_words[i] > 0:
            result += number_of_conotation_words[i]
        
    return result

def count_if_negatives_are_in_file(number_of_conotation_words):
    result = 0
    for i in range(len(number_of_conotation_words)):
        if i % 2 != 0 and number_of_conotation_words[i] > 0:
            result += number_of_conotation_words[i]
        
    return result

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
    
    word_counts = []
    number_of_conotation_words = [0] * (len(sys.argv) - 1) * 2
    data_to_process = []
    
    if len(sys.argv) < 2:
        exit_with_error("Usage: python main.py <file> ...")
    
    try:
        with open('data\data.json', 'r') as file:
            conotation_words_dict = json.load(file)
        print("Data successfully read from data.json by process", rank)
    except FileNotFoundError:
        exit_with_error("The file was not found.")
    except json.JSONDecodeError:
        exit_with_error("Error decoding JSON.")
        
    if rank == 0:
        merged_number_of_conotation_words = [0] * (len(sys.argv) - 1)*2
        nltk.download('punkt')
        sentences_with_file_numbers, word_counts = read_files(file_paths)
        
        if sum(word_counts) < 1000 and sum(word_counts) < 1000000:
            if rank == 0:
                print("Number of words in documents: ", sum(word_counts))
                print("The total number of words in all files must be at least 1000 and less than 1000000.")
                MPI.COMM_WORLD.Abort(1)
            
        data_to_process = divide_sentences_among_threads(sentences_with_file_numbers, size, comm)
    else:
        data_to_process = comm.recv(source=0)
    
    for sentences in data_to_process:           
        sentence, file_number = sentences
        words = word_tokenize(sentence)
        words = [word.capitalize() for word in words]
        for word in words:
            if word in conotation_words_dict:
                if conotation_words_dict[word]:
                    number_of_conotation_words[file_number * 2] += 1
                else:
                    number_of_conotation_words[(file_number * 2) + 1] += 1
                        
    merged_number_of_conotation_words = comm.gather(number_of_conotation_words, root=0)
    
    if rank == 0:
        merged_number_of_conotation_words = [sum(x) for x in zip(*merged_number_of_conotation_words)]
        positive = 0
        negative = 0
        file_number = 0
        for i in range((len(sys.argv) - 1)*2):
            if i % 2 == 0:
                tf_p = merged_number_of_conotation_words[i] / word_counts[file_number]
                tf_n = merged_number_of_conotation_words[i+1] / word_counts[file_number]
            
                if count_if_positives_are_in_file(merged_number_of_conotation_words) > 0:
                    idf_p = math.log(sum(word_counts) / count_if_positives_are_in_file(merged_number_of_conotation_words))
                else:
                    idf_p = 0
                    
                if count_if_negatives_are_in_file(merged_number_of_conotation_words) > 0:
                    idf_n = math.log(sum(word_counts) / count_if_negatives_are_in_file(merged_number_of_conotation_words))
                else:
                    idf_n = 0    
            
                positive += tf_p * idf_p
                negative += tf_n * idf_n
                
                file_number += 1
                
            
        sentiment = determine_sentiment(positive, negative)
        
        print("Positive: ", positive)
        print("Negative: ", negative)
        print("Sentiment: ", sentiment)
        
    MPI.Finalize()

if __name__ == "__main__":
    main(sys.argv[1:])