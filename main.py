# Imports
import os
import glob
import math
import nltk
from nltk.corpus import state_union, stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.text import TextCollection


# Initialize necessary components
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# PHASE 1: **INDEXATION**
# Preprocesses text: tokenization, stop word removal, and stemming
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered = [w for w in tokens if w not in stop_words]
    stemmed_tokens = [ps.stem(w) for w in filtered]
    return stemmed_tokens

# Create a corpus by reading text files from the 'corpus' directory
def create_corpus():
    corpus = {}
    
    # we can load text from state_union of nltk.corpus
    # you can remove those text files from the corpus by just commenting this section
    #start of comment section
    corpus = {
        "2003-GWBush": state_union.raw("2003-GWBush.txt"),
        "2004-GWBush": state_union.raw("2004-GWBush.txt"),
        "2005-GWBush": state_union.raw("2005-GWBush.txt"),
        "2006-GWBush": state_union.raw("2006-GWBush.txt")
    }

    #end of comment section

    corpus_path = os.path.join(os.path.dirname(__file__), 'corpus')

    # Read all .txt files from 'corpus' directory and add to corpus
    for file_path in glob.glob(os.path.join(corpus_path, '*.txt')):
        filename = os.path.splitext(os.path.basename(file_path))[0]
        with open(file_path, 'r', encoding='utf-8') as file:
            corpus[filename] = file.read()
    
    return corpus

# Process the corpus by applying the preprocessing to all documents
def process_corpus(corpus):
    for filename, text in corpus.items():
        corpus[filename] = preprocess_text(text)

# PHASE 2: **QUERY PROCESSING**
# Get the query from the user
def get_request():
    return input('Enter a word: ').lower()

# Preprocess the query word (apply stemming)
def process_request(request):
    return ps.stem(request)

# PHASE 3: **APPARIEMENT (MATCHING)**
# Perform the search and calculate scores (TF-IDF, weights, etc.)
def fitting(corpus, stemmed_word):
    # Create a TextCollection for TF-IDF calculation
    mytexts = TextCollection(list(corpus.values()))

    # Find and analyze documents containing the stemmed word
    docs = find_documents(stemmed_word)
    if not docs:
        print("The word is not present in any document.")
    else:
        print(f"The word is present in {len(docs)} document(s):")
        for i in range(len(docs)):
            print(f"{i+1} - ", docs[i])
        print()

        occ = nb_occ_each_doc(stemmed_word, docs)
        print("The occurrence count of the word in each document:")
        for i in range(len(docs)):
            print(f"{i+1} - {docs[i]} : ", occ[docs[i]])
        print()

        w = weight_each_doc(stemmed_word, mytexts, occ)
        print("The weight of the word in each document:")
        for i in range(len(docs)):
            print(f"{i+1} - {docs[i]} : ", w[docs[i]])
        print()

        tf_idf = tf_idf_each_doc(stemmed_word, mytexts, docs)
        print("The TF-IDF of the word in each document:")
        for i in range(len(docs)):
            print(f"{i+1} - {docs[i]} : ", tf_idf[docs[i]])
        print()

        result = most_relevant(w)
        print("The most relevant document is: (", docs.index(result)+1, ' - ', result, ")")
        print()

# Search functions for word matching and document analysis
def find_documents(word):
    """Return documents containing the specified word."""
    return [doc for doc, tokens in corpus.items() if word in tokens]

def nb_occ_each_doc(word, docs):
    """Return the occurrence count of the word in each document."""
    return {doc: corpus[doc].count(word) for doc in docs}

def weight_each_doc(word, mytexts, occ):
    """Calculate the weight of the word in each document using IDF."""
    idf = mytexts.idf(word)
    return {doc: idf * (1 + math.log(nb)) for doc, nb in occ.items()}

def tf_idf_each_doc(word, mytexts, docs):
    """Calculate the TF-IDF of the word in each document."""
    return {doc: mytexts.tf_idf(word, corpus[doc]) for doc in docs}

def most_relevant(weights):
    """Return the most relevant document based on word weight."""
    return max(weights, key=weights.get)

# PHASE 4: **EXTRA INFORMATION**
# Function to display more information about the query word
def more_infos(word):
    """Identify and print grammatical and semantic details of the word."""
    
    # Grammatical function (POS tagging)
    try:
        tagged = nltk.pos_tag([word])[0]
        print(f"The grammatical function of {tagged[0]} is:", tagged[1])
    except Exception as e:
        print(str(e))
    
    # Synonyms, antonyms, and definition
    syns = wordnet.synsets(word)
    if syns:
        print("The meaning of the word is:", syns[0].definition())
        synonyms = {l.name() for syn in syns for l in syn.lemmas()}
        antonyms = {l.antonyms()[0].name() for syn in syns for l in syn.lemmas() if l.antonyms()}

        if synonyms:
            print(f"Synonyms of '{word}':", synonyms)
        else:
            print(f"No synonyms found for '{word}'")
        
        if antonyms:
            print(f"Antonyms of '{word}':", antonyms)
        else:
            print(f"No antonyms found for '{word}'")
    else:
        print(f"No information found for '{word}'.")

# MAIN FUNCTION: Execution starts here
if __name__ == "__main__":
    # Step 1: Create and process the corpus
    corpus = create_corpus()
    process_corpus(corpus)

    # Step 2: Get the query from the user
    request = get_request()
    stemmed_word = process_request(request)

    # Step 3: Search for matching documents
    fitting(corpus, stemmed_word)
    
    # Step 4: Show additional information about the query word
    more_infos(request)
