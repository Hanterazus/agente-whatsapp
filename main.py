from flask import Flask, Response

app = Flask(__name__)


@app.get("/")
def inicio():
    return {"status": "online"}


@app.post("/whatsapp")
def whatsapp():
    resposta = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>Olá, integração funcionando!</Message>
</Response>"""
    return Response(resposta, mimetype="application/xml")
