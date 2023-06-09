from pickle import TRUE
import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    dict_map = {}

    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        with open(path, encoding="utf-8") as f:
            dict_map[file] = f.read()
    
    return dict_map

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tk = nltk.tokenize.word_tokenize(document.lower())
    
    ls_word = []

    for i in tk:
        if i not in string.punctuation and i not in nltk.corpus.stopwords.words("english"):
            ls_word.append(i)
    
    return sorted(ls_word)

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Get all words in corpus
    words = set()
    for filename in documents:
        words.update(documents[filename])

    # Calculate IDFs
    idfs = {}
    for word in words:
        f = sum(word in documents[document] for document in documents)
        idf = math.log(len(documents) / f)
        idfs[word] = idf

    return idfs

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    top_files = {}

    for file, words in files.items():
        rank = 0
        for word in query:
            if word in words:
                # Sum of the tf-idf values for any word in the query and appear in the file
                rank += words.count(word) * idfs[word]
        # Word in the query don't appear in the file not contribute to rank
        if rank != 0:
            top_files[file] = rank
    
    # Order with the best match first, return a list n top file
    sort = sorted(list(top_files.keys()), key = lambda i: top_files[i], reverse=True)[:n]

    return sort

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    top_sens =  {}

    for sen, words in sentences.items():
        rank = 0
        # Sum of idf values for any word int he query and in the sentence
        for word in query:
            if word in words:
                rank += idfs[word]
        # Word in the query don't appear in the sentence not contribute to rank
        if rank != 0:
            density = sum(word in query for word in words) / len(words)
            top_sens[sen] = (rank, density)
    
    # Order n top sentences that match the query, rank occording to IDF
    sort = sorted(list(top_sens.keys()), key = lambda i: top_sens[i], reverse=True)[:n]

    return sort

if __name__ == "__main__":
    main()
