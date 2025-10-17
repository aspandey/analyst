from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from rag.chat_functions import app_stocks_info
import uvicorn

from debug.logger_config import dbg
from fastapi.responses import StreamingResponse

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


# For development allow the UI origin (or use ["*"] temporarily)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] for quick dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/stocks_info")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_message = data.get("message")
    dbg.info(f"Received message: {user_message}")

    if not user_message:
        return JSONResponse(status_code=400, content={"error": "Missing 'message' in request body."})

    async def stream_response():
        # Assuming chat_with_user yields chunks of text
        async for chunk in app_stocks_info(user_message):
            yield chunk

    return StreamingResponse(stream_response(), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run("endpoints.chat:app", host="0.0.0.0", port=8000, reload=True)

# Example curl command to test the /stocks_info endpoint:
# curl -X POST "http://localhost:8000/stocks_info" -H "Content-Type: application/json" -d '{"message": "Hello"}'