#THIS FIRST CHUNK OF CODE ANSWERS QUESTIONS BASED ON A PROVIDED DATASET USING THE VECTOR DATABASE 
#AND SIMILARITY SEARCH



#run pip install -r requirements.txt to install necessary packages
from flask import Flask, render_template, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.csv_loader import CSVLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import HuggingFaceDatasetLoader


#Load the data from hugging face dataset
dataset_name = "brucewlee1/mmlu-astronomy"
page_content_column = "correct_options_literal"  

#Create a loader instance
loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)

# Load the data
data = loader.load()



#CHUNKING

#uses RecursiveCharacterTextSplitter (chunking method)
#parameters: 300 chunk size, 30 overlap between chunks (appropriate for our dataset)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

#text from file is split into documents
docs = text_splitter.split_documents(data)



#EMBEDDING

modelPath = "sentence-transformers/all-MiniLM-l6-v2"

#dictionary to configure the model (specifies CPU use)
model_kwargs = {'device':'cpu'}

#dictionary with encoding options
#:False means that vector embeddings will NOT be altered to have a unit magnitude (||v|| = 1) 
encode_kwargs = {'normalize_embeddings': False}

#initialize instance of HuggingFaceEmbeddings with parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     #give pre-trained model's path
    model_kwargs=model_kwargs, #pass model configuration options
    encode_kwargs=encode_kwargs #pass encoding options
)

#FAISS vector store for embeddings and efficient searching of them (basically does vector search)
db = FAISS.from_documents(docs, embeddings)

#access question
with open("question.txt", "r", encoding="utf-8") as file:
    # Read the entire file as a string
    question = file.read().strip()

#conduct similarity search
searchDocs = db.similarity_search(question)
context = searchDocs[0].page_content




#THIS CHUNK OF CODE USES OPEN AI'S API AND GPT-4o-MINI, JUST IN CASE THE DATABASE DOESN'T HAVE A SATISFACTORY ANSWER

import openai

#set up API key
openai.api_key = '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'

#make a request to the GPT-4o Mini model using the chat endpoint
response = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "You are a kind astronomy teacher."},
        {"role": "user", "content": question}
    ]
)

#response
openAIResponse = response.choices[0].message['content'].strip()




# BOTH VERSIONS OF THE ANSWER FOR THE USER
output = context[2:-2] + "\n\nIn the case this doesn't answer your question completely, here is a more thorough description!\n" + openAIResponse

print(output)

#OUTPUT ON WEBSITE

app = Flask(__name__)


@app.route("/")
def about():
    return render_template("about.html")

@app.route("/ask_question", methods=["POST"])
def ask_question():
    question = request.json.get('msg')

    # Conduct similarity search
    searchDocs = db.similarity_search(question)
    context = searchDocs[0].page_content if searchDocs else "No relevant context found."

    # Use OpenAI to generate additional response
    openai.api_key = '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'
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "You are a kind astronomy teacher."},
            {"role": "user", "content": question}
        ]
    )
    openAIResponse = response.choices[0].message['content'].strip()

    # Combine context and OpenAI response
    combined_response = context[2:-2] + "\n\nIn the case this doesn't answer your question completely, here is a more thorough description!\n" + openAIResponse
    
    return jsonify({"response": combined_response})

if __name__ == "__main__":
    app.run(debug=True)


print(output)

