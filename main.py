# # main.py
# # docx file download for the table
# # timeline, investment, location, company, date, contact details, updated date, added date....
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator
import httpx
import os
import json
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
# ─────────────────────────────────────────────
# Load ENV, 
# ─────────────────────────────────────────────
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
AGENT_ID = os.getenv("AGENT_ID")
ENVIRONMENT_ID = os.getenv("ENVIRONMENT_ID")
client = MongoClient(os.getenv("MONGO_URI"))

db = client["chat_app"]
messages_collection = db["messages"]
# 🚨 Fail fast if missing
if not ANTHROPIC_API_KEY:
    raise Exception("❌ ANTHROPIC_API_KEY is missing")
if not AGENT_ID:
    raise Exception("❌ AGENT_ID is missing")
if not ENVIRONMENT_ID:
    raise Exception("❌ ENVIRONMENT_ID is missing")

# Debug (first 10 chars only)
# print("✅ API KEY:", ANTHROPIC_API_KEY[:10])
# print("✅ AGENT ID:", AGENT_ID)
# print("✅ ENV ID:", ENVIRONMENT_ID)

BASE_URL = "https://api.anthropic.com/v1"

HEADERS = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01",
    "anthropic-beta": "managed-agents-2026-04-01",  # ✅ correct
    "content-type": "application/json",
}

STREAM_HEADERS = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01",
    "anthropic-beta": "agent-api-2026-03-01",  # ✅ REQUIRED
}

# ─────────────────────────────────────────────
# App Init
# ─────────────────────────────────────────────
app = FastAPI(title="USA Market Expansion Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class MessageRequest(BaseModel):
    message: str
    session_id: str | None = None


class SessionResponse(BaseModel):
    session_id: str


# ─────────────────────────────────────────────
# Create Session
# ─────────────────────────────────────────────
@app.post("/api/sessions", response_model=SessionResponse)
async def create_session():
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{BASE_URL}/sessions",
            headers=HEADERS,
            json={
                "agent": AGENT_ID,
                "environment_id": ENVIRONMENT_ID,
                "title": "POC Session",
            },
        )

        print("🔵 SESSION STATUS:", resp.status_code)
        print("🔵 SESSION RESPONSE:", resp)

        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail=resp.text,
            )

        data = resp.json()
        return {"session_id": data["id"]}

@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),  # 2s, 4s, 8s, 16s
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
    reraise=True
)
async def send_event_with_retry(session_id: str, message: str):
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{BASE_URL}/sessions/{session_id}/events",
            headers=HEADERS,
            json={
                "events": [{
                    "type": "user.message",
                    "content": [{"type": "text", "text": message}]
                }]
            },
        )
        if resp.status_code == 429:
            raise httpx.HTTPStatusError("Rate limited", request=resp.request, response=resp)
        return resp

# ─────────────────────────────────────────────
# Chat + Streaming
# ─────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: MessageRequest):
    if not req.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    try:
        # 1. Send message with retry
        await send_event_with_retry(req.session_id, req.message)

        # Save to MongoDB
        messages_collection.insert_one({
            "session_id": req.session_id,
            "role": "user",
            "content": req.message,
            "created_at": datetime.utcnow()
        })

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise HTTPException(status_code=429, detail="Rate limited by Claude. Please wait a moment.")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)

    # 2. Stream response
    async def stream_events() -> AsyncGenerator[str, None]:
        full_response = ""
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "GET",
                f"{BASE_URL}/sessions/{req.session_id}/stream",
                headers=STREAM_HEADERS,
            ) as response:

                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    while "\n\n" in buffer:
                        event_chunk, buffer = buffer.split("\n\n", 1)
                        for line in event_chunk.split("\n"):
                            if not line.startswith("data:"):
                                continue
                            raw = line[5:].strip()
                            if not raw:
                                continue
                            try:
                                event = json.loads(raw)
                                etype = event.get("type", "")

                                if etype == "agent":
                                    for block in event.get("content", []):
                                        if block.get("type") in ["text", "output_text"]:
                                            text = block.get("text", "")
                                            if text:
                                                full_response += text
                                                yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"

                                elif etype == "agent_tool_use":
                                    yield f"data: {json.dumps({'type': 'tool', 'content': event.get('name', 'searching')})}\n\n"

                                elif etype == "status_idle":
                                    if full_response.strip():
                                        messages_collection.insert_one({
                                            "session_id": req.session_id,
                                            "role": "agent",
                                            "content": full_response,
                                            "created_at": datetime.utcnow()
                                        })
                                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                                        return

                                elif etype == "error":
                                    yield f"data: {json.dumps({'type': 'error', 'content': str(event)})}\n\n"
                                    return
                            except Exception as e:
                                print("Parse error:", e)

    return StreamingResponse(stream_events(), media_type="text/event-stream")

@app.get("/api/messages/{session_id}")
async def get_messages(session_id: str):

    docs = list(
        messages_collection
        .find({"session_id": session_id})
        .sort("created_at", 1)
    )

    return [
        {
            "role": d["role"],
            "content": d["content"]
        }
        for d in docs
    ]

# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "agent_id": AGENT_ID,
        "environment_id": ENVIRONMENT_ID,
    }
# import os
# import json
# import httpx
# import sqlite3
# from datetime import datetime
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from motor.motor_asyncio import AsyncIOMotorClient
# from pydantic import BaseModel
# from dotenv import load_dotenv

# load_dotenv()

# # ───────────────────────────────
# # CONFIG
# # ───────────────────────────────
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# AGENT_ID = os.getenv("AGENT_ID")
# ENVIRONMENT_ID = os.getenv("ENVIRONMENT_ID")

# if not ANTHROPIC_API_KEY:
#     raise Exception("Missing ANTHROPIC_API_KEY")

# BASE_URL = "https://api.anthropic.com/v1"

# HEADERS = {
#     "x-api-key": ANTHROPIC_API_KEY,
#     "anthropic-version": "2023-06-01",
#     "anthropic-beta": "agent-api-2026-03-01",
#     "content-type": "application/json",
# }

# # ───────────────────────────────
# # DB SETUP
# # ───────────────────────────────
# mongo = AsyncIOMotorClient("mongodb://localhost:27017")
# db = mongo.agent_db

# sql_conn = sqlite3.connect("data.db", check_same_thread=False)

# # ───────────────────────────────
# # APP
# # ───────────────────────────────
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ───────────────────────────────
# # MODELS
# # ───────────────────────────────
# class SearchPayload(BaseModel):
#     start_date: str
#     end_date: str
#     industry_type: str | None = ""

# # ───────────────────────────────
# # SAFE PARSER
# # ───────────────────────────────
# def safe_parse(text):
#     try:
#         return json.loads(text)
#     except:
#         start = text.find("[")
#         end = text.rfind("]")
#         if start != -1 and end != -1:
#             return json.loads(text[start:end+1])
#         raise Exception("Invalid JSON from agent")

# # ───────────────────────────────
# # AGENT SEARCH
# # ───────────────────────────────
# @app.post("/api/agent/search")
# async def agent_search(payload: SearchPayload):

#     query = json.dumps({
#         "start_date": payload.start_date,
#         "end_date": payload.end_date,
#         "industry_type": payload.industry_type or ""
#     })

#     async with httpx.AsyncClient(timeout=None) as client:

#         # Create session
#         payload = {
#     "agent": {
#         "type": "agent_reference",
#         "id": AGENT_ID
#     },
#     "environment": ENVIRONMENT_ID,
#     "title": "structured extraction"
# }

#         print("🚀 FINAL PAYLOAD:", payload)

#         session = await client.post(
#     f"{BASE_URL}/sessions",
#     headers=HEADERS,
#     json=payload
# )
#         # print(session)
#         data = session.json()
#         print("🔴 SESSION RESPONSE:", data)

#         if "id" not in data:
#             raise HTTPException(status_code=500, detail=data)

#         session_id = data["id"]

#         # Send input
#         await client.post(
#             f"{BASE_URL}/sessions/{session_id}/events",
#             headers=HEADERS,
#             json={
#                 "events": [{
#                     "type": "user.message",
#                     "content": [{"type": "text", "text": query}]
#                 }]
#             }
#         )

#         # Read stream
#         full_text = ""

#         async with client.stream(
#             "GET",
#             f"{BASE_URL}/sessions/{session_id}/stream",
#             headers=HEADERS
#         ) as response:

#             async for line in response.aiter_lines():
#                 if not line.startswith("data:"):
#                     continue

#                 data = json.loads(line[5:].strip())
#                 etype = data.get("type")

#                 if etype == "agent.message":
#                     for c in data.get("content", []):
#                         if c.get("type") == "text":
#                             full_text += c["text"]

#                 elif etype == "session.status_idle":
#                     break

#     parsed = safe_parse(full_text)

#     return parsed


# # ───────────────────────────────
# # SAVE TO MONGO
# # ───────────────────────────────
# @app.post("/api/save/mongo")
# async def save_mongo(rows: list):

#     if not rows:
#         return {"msg": "no data"}

#     for r in rows:
#         r["created_at"] = datetime.utcnow()

#     await db.company_data.insert_many(rows)

#     return {"msg": "saved to mongo"}


# # ───────────────────────────────
# # SAVE TO SQL
# # ───────────────────────────────
# @app.post("/api/save/sql")
# async def save_sql(rows: list):

#     cur = sql_conn.cursor()

#     cur.execute("""
#     CREATE TABLE IF NOT EXISTS companies (
#         company TEXT,
#         country TEXT,
#         industry TEXT,
#         expansion_type TEXT,
#         location TEXT,
#         investment TEXT,
#         date TEXT,
#         source TEXT,
#         url TEXT,
#         summary TEXT
#     )
#     """)

#     for r in rows:
#         cur.execute("""
#         INSERT INTO companies VALUES (?,?,?,?,?,?,?,?,?,?)
#         """, (
#             r.get("company"),
#             r.get("country"),
#             r.get("industry"),
#             r.get("expansion_type"),
#             r.get("location"),
#             r.get("investment"),
#             r.get("date"),
#             r.get("source"),
#             r.get("url"),
#             r.get("summary"),
#         ))

#     sql_conn.commit()

#     return {"msg": "saved to sql"}

# main.py
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse, JSONResponse
# from pydantic import BaseModel
# from typing import AsyncGenerator, List, Dict, Any
# import httpx
# import os
# import json
# from dotenv import load_dotenv
# from pymongo import MongoClient
# from datetime import datetime

# load_dotenv()

# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# AGENT_ID = os.getenv("AGENT_ID")
# ENVIRONMENT_ID = os.getenv("ENVIRONMENT_ID")
# MONGO_URI = os.getenv("MONGO_URI")

# if not all([ANTHROPIC_API_KEY, AGENT_ID, ENVIRONMENT_ID, MONGO_URI]):
#     raise Exception("❌ Missing required environment variables")

# client = MongoClient(MONGO_URI)
# db = client["chat_app"]
# messages_collection = db["messages"]
# expansions_collection = db["usa_expansions"]

# BASE_URL = "https://api.anthropic.com/v1"

# HEADERS = {
#     "x-api-key": ANTHROPIC_API_KEY,
#     "anthropic-version": "2023-06-01",
#     "anthropic-beta": "managed-agents-2026-04-01",
#     "content-type": "application/json",
# }

# STREAM_HEADERS = {
#     "x-api-key": ANTHROPIC_API_KEY,
#     "anthropic-version": "2023-06-01",
#     "anthropic-beta": "agent-api-2026-03-01",
# }

# app = FastAPI(title="USA Market Expansion Agent API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ====================== SCHEMAS ======================
# class CrawlRequest(BaseModel):
#     start_date: str
#     end_date: str
#     industry_type: str = ""

# class UploadRequest(BaseModel):
#     records: List[Dict[str, Any]]

# # ====================== HELPER ======================
# async def stream_agent_response(session_id: str):
#     """Robust streaming with better error handling and timeout"""
#     full_response = ""
#     timeout = httpx.Timeout(180.0, connect=30.0)   # 3 minutes total

#     try:
#         async with httpx.AsyncClient(timeout=timeout) as http_client:
#             async with http_client.stream(
#                 "GET",
#                 f"{BASE_URL}/sessions/{session_id}/stream",
#                 headers=STREAM_HEADERS,
#             ) as response:
#                 response.raise_for_status()
#                 buffer = ""

#                 async for chunk in response.aiter_text():
#                     buffer += chunk

#                     while "\n\n" in buffer:
#                         event_chunk, buffer = buffer.split("\n\n", 1)

#                         for line in event_chunk.splitlines():
#                             if line.startswith("data:"):
#                                 raw = line[5:].strip()
#                                 if not raw or raw == "[DONE]":
#                                     continue
#                                 try:
#                                     event = json.loads(raw)
#                                     etype = event.get("type", "")

#                                     if etype == "agent":
#                                         for block in event.get("content", []):
#                                             if block.get("type") in ["text", "output_text"]:
#                                                 text = block.get("text", "")
#                                                 if text:
#                                                     full_response += text
#                                                     # Yield for frontend (optional - you can remove if not needed)
#                                                     yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"

#                                     elif etype == "status_idle" or etype == "done":
#                                         # Agent finished
#                                         yield f"data: {json.dumps({'type': 'done'})}\n\n"
#                                         return

#                                 except json.JSONDecodeError:
#                                     continue  # ignore malformed lines

#     except httpx.RemoteProtocolError as e:
#         print(f"⚠️ RemoteProtocolError (common with long agent runs): {e}")
#         # Return whatever we accumulated so far
#         if full_response.strip():
#             yield f"data: {json.dumps({'type': 'partial', 'content': full_response})}\n\n"
#         yield f"data: {json.dumps({'type': 'error', 'content': 'Connection interrupted. Try again or increase timeout.'})}\n\n"
#     except Exception as e:
#         print(f"❌ Stream error: {type(e).__name__}: {e}")
#         yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


# # ====================== NEW CRAWL ENDPOINT (Improved) ======================
# @app.post("/api/crawl-expansion")
# async def crawl_expansion(req: CrawlRequest):
#     # 1. Create a fresh session
#     async with httpx.AsyncClient(timeout=30) as client:
#         session_resp = await client.post(
#             f"{BASE_URL}/sessions",
#             headers=HEADERS,
#             json={
#                 "agent": AGENT_ID,
#                 "environment_id": ENVIRONMENT_ID,
#                 "title": f"Expansion Crawl {req.start_date}–{req.end_date}",
#             },
#         )
#         if session_resp.status_code != 200:
#             raise HTTPException(status_code=500, detail="Failed to create session")
#         session_id = session_resp.json()["id"]

#     # 2. Send the input JSON
#     input_data = {
#         "start_date": req.start_date,
#         "end_date": req.end_date,
#         "industry_type": req.industry_type.strip()
#     }

#     async with httpx.AsyncClient(timeout=60) as client:
#         event_resp = await client.post(
#             f"{BASE_URL}/sessions/{session_id}/events",
#             headers=HEADERS,
#             json={
#                 "events": [
#                     {
#                         "type": "user.message",
#                         "content": [{"type": "text", "text": json.dumps(input_data)}]
#                     }
#                 ]
#             },
#         )
#         if event_resp.status_code != 200:
#             raise HTTPException(status_code=event_resp.status_code, detail=event_resp.text)

#     # 3. Stream and collect ALL possible text output
#     full_response = ""
#     timeout = httpx.Timeout(300.0, connect=30.0)  # 5 minutes max

#     try:
#         async with httpx.AsyncClient(timeout=timeout) as http_client:
#             async with http_client.stream(
#                 "GET",
#                 f"{BASE_URL}/sessions/{session_id}/stream",
#                 headers=STREAM_HEADERS,
#             ) as response:
#                 response.raise_for_status()
#                 buffer = ""

#                 async for chunk in response.aiter_text():
#                     buffer += chunk
#                     while "\n\n" in buffer:
#                         event_chunk, buffer = buffer.split("\n\n", 1)
#                         for line in event_chunk.splitlines():
#                             if not line.startswith("data:"):
#                                 continue
#                             raw = line[5:].strip()
#                             if not raw:
#                                 continue
#                             try:
#                                 event = json.loads(raw)
#                                 etype = event.get("type", "")

#                                 print(f"Event received: {etype}")   # ← Debug: see all event types

#                                 # Collect text from more event types
#                                 if etype in ["agent", "agent.message", "output"]:
#                                     for block in event.get("content", []):
#                                         if isinstance(block, dict):
#                                             if block.get("type") in ["text", "output_text"]:
#                                                 text = block.get("text", "")
#                                                 if text:
#                                                     full_response += text + "\n"

#                             except json.JSONDecodeError:
#                                 continue
#     except Exception as e:
#         print(f"Streaming error: {type(e).__name__}: {e}")

#     # === SAVE RAW OUTPUT FOR DEBUGGING ===
#     print("=== FULL RAW RESPONSE ===")
#     print(repr(full_response[:2000]))
#     with open("agent_raw_output.txt", "w", encoding="utf-8") as f:
#         f.write(full_response)
#     print("✅ Raw agent output saved to: agent_raw_output.txt")

#     # Clean up common wrappers
#     cleaned = full_response.strip()
#     if cleaned.startswith("```json"):
#         cleaned = cleaned.split("```json", 1)[1].rsplit("```", 1)[0].strip()
#     elif cleaned.startswith("```"):
#         cleaned = cleaned.split("```", 1)[1].rsplit("```", 1)[0].strip()

#     # Aggressive JSON extraction (in case extra text is present)
#     import re
#     json_array_match = re.search(r'(\[\s*\{[\s\S]*?\}\s*\])', cleaned, re.DOTALL)
#     if json_array_match:
#         cleaned = json_array_match.group(1)

#     try:
#         data = json.loads(cleaned)
#         if not isinstance(data, list):
#             data = []
#         print(f"✅ Successfully parsed {len(data)} records")
#         return data
#     except Exception as e:
#         print(f"JSON parse failed: {e}")
#         print("First 300 chars of cleaned:", repr(cleaned[:300]))
#         raise HTTPException(
#             status_code=500,
#             detail="Agent did not return valid JSON array. Check agent_raw_output.txt for details."
#         )


# # ====================== UPLOAD ======================
# @app.post("/api/upload-expansions")
# async def upload_expansions(req: UploadRequest):
#     if not req.records:
#         raise HTTPException(status_code=400, detail="No records provided")

#     for record in req.records:
#         record["uploaded_at"] = datetime.utcnow()
#         record["crawled_at"] = datetime.utcnow()  # optional

#     result = expansions_collection.insert_many(req.records)
#     return {"inserted_count": len(result.inserted_ids), "status": "success"}


# # ====================== HEALTH ======================
# @app.get("/api/health")
# async def health():
#     return {"status": "ok", "agent_id": AGENT_ID}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)