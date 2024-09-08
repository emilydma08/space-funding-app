#run pip install -r requirements.txt to install necessary stuff



#Install and import necessary libraries

#!pip install -q langchain
#!pip install -q torch
#!pip install -q transformers
#!pip install -q sentence-transformers
#!pip install -q datasets
#!pip install -q faiss-cpu

#following line is a debugging attempt to ensure the compatible version of each library
#!pip install -q langchain==0.1.10 torch transformers sentence-transformers datasets dill==0.3.1.1 pyarrow<10.0.0 jupyterlab~=3.6.0
#!pip install datasets

#following lines were suggested by compiler 
#!pip install -U langchain-community
#!pip install -U langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline

from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import FAISS
#from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.csv_loader import CSVLoader
#from langchain.document_loaders import HuggingFaceDatasetLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import HuggingFaceDatasetLoader



#Load the data from csv files using CSVLoader from Langchain
#reference article used a different loader for HuggingFace dataset but CSVLoader works for my file type

dataset_name = "brucewlee1/mmlu-astronomy"
page_content_column = "correct_options_literal"  # or any other column you're interested in

#Create a loader instance
loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)

# Load the data
data = loader.load()

#file_path = (
#    "/kaggle/input/yt-llm-database/YT-LLM-Database.csv"
#)

#loader = CSVLoader(file_path=file_path)
#data = loader.load()

#for record in data[:2]:
#    print(record)


#Chunking

#uses RecursiveCharacterTextSplitter (chunking method)
#parameters: 300 chunk size, 30 overlap between chunks (appropriate for our file)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

#text from file is split into documents
docs = text_splitter.split_documents(data)


#Embedding


#path to model (model suggested by reference article)
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
#modelPath = "abhitopia/question-answer-generation"
#modelPath="twigs/bart-text2text-simplifier"

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


#Connecting with LLM: create LangChain pipeline with more arguments

#specify model name
model_name = "Intel/dynamic_tinybert"
#model_name = "adsabs/astroBERT"
#model_name ="impira/layoutlm-document-qa"
#model_name = "abhitopia/question-answer-generation"
#model_name = "twigs/bart-text2text-simplifier"
#model_name = "edwinmoradian90/email_parser_mistral_t5_small"
#model_name="google/pegasus-xsum"
#model_name = "microsoft/DialoGPT-large"

#####pipe = pipeline("text2text-generation", model="abhitopia/question-answer-generation")


#load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)

#define pipeline using above variables
question_answerer = pipeline(
    "question-answering", 
    model=model_name, 
    tokenizer=tokenizer,
    #return_tensors='pt',
    max_length=512
)

#create an instance of HuggingFacePipeline, "wraps" previous question-answering pipeline
#but with additional arguments
llm = HuggingFacePipeline(
    pipeline=question_answerer,
    model_kwargs={"temperature": 0.7, "max_new_tokens": 50},
)

#Connecting with LLM: retrieve the answer 

#retrieve up to 4 relevant workflows
retriever = db.as_retriever(search_kwargs={"k": 4})


#create and configure q&a instance
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)


#User query 
question = "What is the source of the material that causes meteor showers?"

#debugging attempt--qa.run needs context?
searchDocs = db.similarity_search(question)
context = searchDocs[0].page_content

#result = qa.invoke({"query": question, "context":context})

#result = qa.invoke(question)
#final = result['result'].split(':')

print(context)