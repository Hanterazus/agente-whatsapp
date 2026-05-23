import os
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
tools = [TavilySearchResults(max_results=3)]
memory = MemorySaver()
agent = create_react_agent(llm, tools, checkpointer=memory)

SYSTEM_PROMPT = """Você é um atendente virtual simpático e prestativo.
Sempre responda em português brasileiro.
Seja cordial, objetivo e útil."""

app = Flask(__name__)

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    numero = request.form.get("From")
    mensagem = request.form.get("Body")
    config = {"configurable": {"thread_id": numero}}
    result = agent.invoke({
        "messages": [
            ("system", SYSTEM_PROMPT),
            ("user", mensagem)
        ]
    }, config)
    resposta = result["messages"][-1].content
    resp = MessagingResponse()
    resp.message(resposta)
    return str(resp)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)