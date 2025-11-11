from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import uvicorn

app = FastAPI()
OLLAMA_API_URL = "http://localhost:11434/api"

@app.post("/api/chat")
async def chat(req: Request):
    try:
        data = await req.json()
        async def _generate():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{OLLAMA_API_URL}/chat", json=data, timeout=None) as resp:
                    if resp.status_code != 200:
                        error_msg = await resp.aread()
                        yield error_msg
                        return

                    async for line in resp.aiter_lines():
                        if line:
                            yield line + "\n"

        return StreamingResponse(_generate(), media_type="application/json")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/generate")
async def generate(req: Request):
    try:
        data = await req.json()
        async def _generate():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{OLLAMA_API_URL}/generate", json=data, timeout=None) as resp:
                    if resp.status_code != 200:
                        error_msg = await resp.aread()
                        yield error_msg
                        return

                    async for line in resp.aiter_lines():
                        if line:
                            yield line + "\n"

        return StreamingResponse(_generate(), media_type="application/json")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)