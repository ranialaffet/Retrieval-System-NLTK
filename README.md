# Retrieval-System-NLTK

**Introduction**

This project provides a Python script for text retrieval from a corpus of documents. It utilizes NLTK for natural language processing tasks and allows users to search for information within the corpus.

**Features:**

* **Indexing:** Preprocesses text files (tokenization, stop word removal, stemming) and creates a searchable document collection.
* **Query Processing:** Accepts user input to search the corpus.
* **Matching:** Finds documents containing the query term and calculates scores (TF-IDF, weights).
* **Extra Information:** Provides details about the query term, including:
    * **Grammatical Function (POS tagging):** Identifies the part-of-speech of the query word (if possible).
    * **Synonyms & Antonyms (WordNet):** Lists synonyms and antonyms for the query word (if available).
    * **Definition (WordNet):** Provides the definition of the query word (if available).
**Note:**

* WordNet may not provide information for all words.

**Requirements:**

* Python 3.x
* NLTK library

**Instructions:**

1. **Clone or download the repository.**
2. **Install NLTK:**
    * Visit the official NLTK installation guide: [https://www.nltk.org/install.html](https://www.nltk.org/install.html)
    * Follow the instructions provided to install NLTK for your operating system (typically using `pip install nltk`).
3. **Download necessary NLTK resources:**  Use `python NecessaryDownloads.py` in your terminal.
4. **Corpus:** The script includes sample data (`state_union` corpus) for demonstration purposes. You can:
    * **Use the sample data:** No additional corpus preparation is needed.
    * **Create your own corpus:** Place your text files in a directory named `corpus` within the project directory. Ensure each file has a unique name `unique_name.txt` and the content you want to search.
5. **Run the script:** Use `python main.py` in your terminal.

