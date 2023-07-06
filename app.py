from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import os
from dotenv.main import load_dotenv
from utils import *

app = Flask(__name__)
CORS(app)
buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="ADD-YOUR-OPENAI-API-KEY")
system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
conversation = ConversationChain(memory=buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

requests = []
responses = ["How can I assist you?"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data", methods=["POST"])
def get_data():
    data = request.get_json()
    query = data.get("data")
    print(query)

    try:
        if len(requests) > 0: # if it's not the first request
            conversation_string = get_conversation_string(requests,responses)
            refined_query = query_refiner(conversation_string, query)
            print("Refined Query:", refined_query)
            
        else: # if it's the first request
            refined_query = query
        context = find_match(refined_query)
        response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        requests.append(query)
        responses.append(response)
        print(response)
        return jsonify({"response": True, "message": response})
    except Exception as e:
        print(e)
        error_message = f"Error: {str(e)}"
        return jsonify({"message": error_message, "response": False})

if __name__ == "__main__":
    app.run()
