from flask import Flask, request, Response
import requests

app = Flask("Server")

OLLAMA_API_URL = "http://localhost:11434/api/chat"

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    try:
        def generate():
            resp = requests.post(OLLAMA_API_URL, json=data, stream=True)
            resp.raise_for_status() 
            for line in resp.iter_lines():
                if line:
                    yield line + b'\n'

        return Response(generate(), content_type='application/json')
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)