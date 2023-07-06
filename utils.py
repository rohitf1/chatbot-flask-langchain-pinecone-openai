from sentence_transformers import SentenceTransformer
import pinecone
import openai

openai.api_key = "ADD-YOUR-OPENAI-API-KEY"
model = SentenceTransformer('all-MiniLM-L6-v2')
pinecone.init(api_key='ADD-PINECONE-API-KEY', environment='asia-southeast1-gcp')
index = pinecone.Index('langchain-chatbot')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):
    response = openai.Completion.create(
    model="ada:ft-personal:refined-query-2023-07-05-21-38-35",
    prompt=f"Given the following user query and conversation log, formulate a long detailed question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\n?Refined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["?"]
    )
    return response['choices'][0]['text']

def get_conversation_string(requests, responses):
    conversation_string = ""
    for i in range(len(responses)-1):
        conversation_string += "Human: "+requests[i] + "\n"
        conversation_string += "Bot: "+ responses[i+1] + "\n"
    return conversation_string
