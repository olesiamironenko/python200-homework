from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# --- RAG Concepts ---


# Concepts Q1

# Scenario A: RAG
# RAG is the best approach because the assistant needs to answer questions using
# a large internal policy library that is updated regularly. The documents can
# be updated in the knowledge base without retraining the model.
#
# Scenario B: Fine-tuning
# Fine-tuning is the best approach because the startup has many examples of the
# specific brand voice it wants the model to learn. The 3,000 examples can help
# the model consistently reproduce that writing style.
#
# Scenario C: Prompt engineering
# Prompt engineering is the best approach because the analyst only needs to ask
# questions about one short report. The report can simply be included in the
# prompt as context, so RAG or fine-tuning would be unnecessary.


# Concepts Q2

# A confidently wrong answer is more harmful because people are more likely to
# trust and act on it, even when the information is incorrect. A response that
# says "I am not sure" encourages the user to verify the information before
# making a decision.
#
# For example, if an AI confidently gives the wrong dosage for a medication, a
# patient could take the wrong amount and suffer serious health consequences.
# The confident tone makes the answer sound reliable, increasing the chance
# that someone will follow the incorrect advice without questioning it.


# Concepts Q3

# 1. Extract text from source documents
# Read and extract the text content from the documents that will be used 
# as the knowledge source.
#
# 2. Split text into chunks
# Break the extracted text into smaller pieces so they can be searched 
# and retrieved effectively.
#
# 3. Convert text chunks into embeddings
# Turn each text chunk into a numerical vector that represents its meaning.
#
# 4. Receive the user's query
# Accept the question or request that the user wants the system to answer.
#
# 5. Embed the user's query
# Convert the user's query into an embedding so it can be compared 
# with the document chunks.
#
# 6. Retrieve the most relevant chunks
# Find the chunks whose embeddings are most similar to the user's query.
#
# 7. Inject retrieved chunks into the prompt
# Add the relevant retrieved text to the prompt as context for the LLM.
#
# 8. Generate a response from the LLM
# Ask the LLM to produce an answer using the user's query and 
# the retrieved context.


# --- Keyword RAG ---

import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    # Translator to remove punctuation (so "Solar?" -> "Solar")
    translator = str.maketrans("", "", string.punctuation)

    # Tokenize query: lowercase, remove punctuation and stopwords
    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        # Tokenize document: lowercase, remove punctuation and stopwords
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        # Compute simple overlap score
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    # Sort by overlap score (descending)
    scores.sort(reverse=True)

    # Pick the single best match (if score > 0)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

# Keyword Q1

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

result = simple_keyword_retrieval(query, documents, verbose=True)

print(f"\nSelected document: {result[0][0]}")

# The function selected "loyalty.txt" because it had the same keyword overlap
# score as "hours.txt" and "hiring.txt." Since all three documents scored 1,
# the function broke the tie by sorting the document names in reverse order,
# which caused "loyalty.txt" to be chosen even though "hours.txt" is actually
# the most relevant document.


# Keyword Q2

query = "Do you have anything without caffeine?"

result = simple_keyword_retrieval(query, documents, verbose=True)

print(f"\nSelected document: {result[0][0]}")

# No document was selected because none of the query words exactly match 
# words in the documents. Keyword RAG did not get this right because 
# "menu.txt" is the most relevant document, 
# but it does not contain the word "caffeine."
# Semantic retrieval using embeddings would work better because 
# it can compare meaning rather than relying only on exact keyword matches.


# Keyword Q3

# Prediction: 
# I think "loyalty.txt" will be selected because the query asks about 
# signing up for rewards, which is related to joining a loyalty program.

query = "How do I sign up for rewards?"

result = simple_keyword_retrieval(query, documents, verbose=True)

print(f"\nSelected document: {result[0][0]}")

# My prediction was not correct. The function returned "None found" because
# none of the query words exactly matched words in the documents. 
# Although "sign up for rewards" has a similar meaning 
# to "join our loyalty program," keyword retrieval cannot recognize 
# that semantic relationship.


# --- Semantic RAG Concepts ---


# Semantic Q1

# A vector embedding is a text represented as a list of numbers. 
# Texts with similar meanings have embeddings that are closer together
# in vector space.
#
# The chunk with a cosine similarity score of 0.85 is more relevant because a
# higher score means its meaning is more similar to the query. A score of 0.30
# indicates a much weaker semantic relationship.
#
# Semantic search can find relevant information even when the exact words do
# not match because it compares the meaning of the text 
# rather than identifying identical keywords.


# Semantic Q2

# | Feature                | Keyword RAG                   | Semantic RAG             |
# |------------------------|-------------------------------|--------------------------|
# | What is compared?      | Exact word overlap            | Meaning                  |
# | What is retrieved?     | Full document                 | Most relevant text chunks|
# | Can it handle synonyms?| No                            | Yes                      |
# | Storage format         | Plain text dictionary         | Embedding vectors in a   |
# |                        |                               | vector DB.               |
# | Relevance score        | Number of overlapping keywords| Cosine similarity score  |