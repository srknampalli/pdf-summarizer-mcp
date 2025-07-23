import os
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI

# Load environment variables from .env file
load_dotenv()
#load_dotenv("C:/Tredence/pdf-extraction-mcp/.env")
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"), 
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-preview"),
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
# Invoke the model
response = llm.invoke("Explain me anout SVM in machine learning")
print(response)

####------------------Embedding Testing code --------------------------##
# import os
# from dotenv import load_dotenv
# from langchain_openai import AzureOpenAIEmbeddings

# # Step 1: Load environment variables
# #load_dotenv()
# import os
# from dotenv import load_dotenv
# #from langchain.embeddings import AzureOpenAIEmbeddings

# embed = AzureOpenAIEmbeddings(
#     model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),  # Example: "text-embedding-3-large"
#     deployment = "embedding-model",
#     azure_endpoint="https://llm-mlops-openai.openai.azure.com/",
#     api_key="90858d0b603c4323a2df07d8064dbcf6",
#     openai_api_version= "2024-02-01",
# )


# embed = AzureOpenAIEmbeddings(
#     model="text-embedding-3-small" ,  # Example: "text-embedding-3-large"
#     deployment = "embedding-model",
#     azure_endpoint="https://llm-mlops-openai.openai.azure.com/",
#     api_key="90858d0b603c4323a2df07d8064dbcf6",
#     openai_api_version= "2024-02-01",
# )

# # Embed a single query
# input_text = "The meaning of life is 42"
# vector = embed.embed_query(input_text)
# print("Query embedding (first 3 dims):", vector[:3])

# # Embed multiple documents
# input_texts = ["Document 1: This is an introduction.", "Document 2: This is a summary."]
# vectors = embed.embed_documents(input_texts)
# print("Total documents embedded:", len(vectors))
# print("First doc embedding (first 3 dims):", vectors[0][:3])



# text = "Azure OpenAI is a powerful tool for enterprise-grade LLM applications."

# # Get embedding
# embedding_vector = embed.embed_query(text)

# # Show embedding
# print(f"Embedding length: {len(embedding_vector)}")
# print(embedding_vector[:10])  # Display first 10 values