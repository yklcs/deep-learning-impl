import argparse
import requests
import sys
import json

available_models = ["llama3.2", "gemma3:1b", "gemma3:4b"]

def request(server, stream, history, prompt, models, **kwargs):
    history.append({"role": "user", "content": prompt})
    payload = {
        "model": models,
        "messages": history,
        "stream": stream,
        **kwargs
    }
    if stream:
        with requests.post(server, json=payload,stream=True) as response:
            response.raise_for_status()
            full_content = ""
            for raw_line in response.iter_lines():
                if raw_line:
                    line = json.loads(raw_line)['message']['content']
                    print(line, end='', flush=True)
                    full_content += line
            print()
            history.append({"role": "assistant", "content": full_content})
            return full_content
    else:
        with requests.post(server, json=payload) as response:
            response.raise_for_status()
            response = response.json()['message']
            print(response['content'])
            history.append(response)
            return response

def main():
    parser = argparse.ArgumentParser(description="Ollama Client")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=11434, help="Server port")
    parser.add_argument("--model", type=str, default="llama3.2", help=f"Model name: {available_models}")
    parser.add_argument("--stream", action="store_true", help="Enable streaming")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Disable streaming")
    parser.set_defaults(stream=False)
    args = parser.parse_args()

    if args.model not in available_models:
        print(f"Model {args.model} is not available. Choose from {available_models}.")
        sys.exit(1)
    
    
    server = f"http://{args.host}:{args.port}/api"
    history = []
    currnent_model = args.model
    while True:
        prompt = input(">>> ")

        if prompt[0] == '/':
            if prompt.split()[0] == '/load':
                payload = {
                    "model": prompt.split()[1]
                }
                with requests.post(f"{server}/generate", json=payload) as response:
                    response.raise_for_status()
                    print(response.json())
                    currnent_model = response.json()['model']
            elif prompt.split()[0] == '/unload':
                payload = {
                    "model": currnent_model,
                    "keep_alive" :0
                }
                with requests.post(f"{server}/generate", json=payload) as response:
                    response.raise_for_status()
                    print(response.json())
            elif prompt.split()[0] == '/exit':
                print("Exiting...")
                break

        else:
            print()
            response = request(f"{server}/chat", args.stream, history, prompt, currnent_model)
            print()

if __name__ == "__main__":
    main()