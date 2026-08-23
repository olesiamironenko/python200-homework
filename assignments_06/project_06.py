from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


# Step 1: Setup
load_dotenv()
print("API key loaded.")

docs_dir = Path("resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"


# Step 2: Load the Documents
documents = SimpleDirectoryReader(docs_dir).load_data()

print(f"\nDocuments loaded: {len(documents)}")

for doc in documents:
    print(doc.metadata["file_name"])


# Step 3: Build the Index and Query Engine
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)
print("\nIndex built successfully. Ready to answer questions.")


# Step 4: Query the Assistant

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:
    response = query_engine.query(question)

    print(f"\nQuestion: {question}")
    print(f"Answer: {response}")

    print("\nRetrieved Sources:")

    top_node = response.source_nodes[0]

    file_name = top_node.node.metadata.get("file_name", "Unknown")
    score = top_node.score
    chunk_text = top_node.node.get_content()

    print (f"\nTop Retrieved Source:")
    print(f"Document: {file_name}")
    print(f"Similarity Score: {score:.4f}")
    print(f"Chunk Preview: {chunk_text[:200]}...")  # Print first 200 characters of the chunk

# The assistant sounded confident and accurate because all five answers matched
# the information in the Groundwork documents. The retrieved source nodes were
# relevant to each question and helped show where the answers came from. One
# interesting observation was that semantic search does not always retrieve the
# document I expected first. For example, the dairy-free milk question retrieved
# the seasonal specials document even though the menu also contains the general
# milk options, showing that semantic search finds the most relevant text chunks
# rather than relying on document names.


# Step 5: Find a Failure

failure_question = "Do you roast your own coffee beans?"
response = query_engine.query(failure_question)

print(f"\nFailure Question: {failure_question}")
print(f"Answer: {response}")

print("\nRetrieved Sources:")

for i, node in enumerate(response.source_nodes, start=1):
    file_name = node.node.metadata.get("file_name", "Unknown")
    score = node.score
    chunk_text = node.node.get_content()
    
    print(f"\nSource {i}:")
    print(f"Document: {file_name}")
    print(f"Similarity Score: {score:.4f}")
    print(f"Chunk Preview: {chunk_text[:200]}...") 

# I asked whether Groundwork roasts its own coffee beans because I expected the
# documents would not answer that question directly. Although the retrieved
# documents discussed the company's story and wholesale coffee, none of them
# stated whether Groundwork actually roasts its own beans.
#
# Instead of saying the information was unavailable, the assistant answered a
# different question by mentioning the available roast profiles (light, medium,
# and dark). This shows that the model tried to provide a helpful response even
# though the retrieved documents did not contain the requested information.
#
# The assistant still sounded confident even though it did not answer my actual
# question. This suggests that AI-generated responses should always be verified
# against the retrieved sources rather than trusted solely because they sound
# confident.
#
# To improve the system, I would modify the prompt so the assistant only answers
# questions that are directly supported by the retrieved documents. If the
# information is missing, it should clearly say that it does not know instead of
# making an unsupported inference or answering a different question.


# Step 6: Reflection

# In this project, the LlamaIndex implementation only took about two main lines
# of code to build the index and query engine, compared with the many lines
# needed to manually chunk, embed, and index documents. This shows the value of
# using a framework because it handles a lot of the repetitive setup for you and
# lets you focus more on the behavior of the system instead of the low-level
# implementation details.
#
# A different use case for RAG could be to help data analysts 
# search internal documentation, database schemas, data dictionaries, 
# and reporting guides. Instead of spending time looking through documentation, 
# analysts could quickly find the information they need 
# to understand data sources and business rules.
#
# One failure mode that RAG cannot fully prevent is the model misinterpreting
# correctly retrieved information. Even when the right source is found, the
# model can still misunderstand the question, make an unsupported inference, or
# give an answer that sounds confident but does not fully match what the source
# says. The failure in this project about whether Groundwork roasts its own beans
# is an example of this problem.