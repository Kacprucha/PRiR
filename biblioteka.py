import sys
from sklearn.feature_extraction.text import TfidfVectorizer
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
            word_counts.append(len(words))
        file_number += 1
    
    return sentences_with_file_numbers, word_counts

def main(file_paths):
    sentences_with_file_numbers, word_counts = read_files(file_paths)
    sentences = [sentence for sentence, _ in sentences_with_file_numbers]
    
    tfidf = TfidfVectorizer()

    result = tfidf.fit_transform(sentences)
    
    print('\nidf values:')
    for ele1, ele2 in zip(tfidf.get_feature_names_out(), tfidf.idf_):
        print(ele1, ':', ele2)

    print('\nWord indexes:')
    print(tfidf.vocabulary_)
    
    print('\ntf-idf value:')
    print(result)

    print('\ntf-idf values in matrix form:')
    print(result.toarray())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <file> ...")
        sys.exit(1)
    
    file_paths = sys.argv[1:]
    main(file_paths)
