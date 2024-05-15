"""
Streaming LLM response over websockets for FE
Stream LLM response (Open Spurce Hugging Face models)
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
from fastapi.responses import StreamingResponse
import requests
import httpx
from urllib.parse import quote

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from langchain.llms import HuggingFacePipeline 
from transformers import pipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")



from streamer import CustomStreamer
from threading import Thread

from queue import Queue


app = FastAPI()

model, tokenizer = llm, tokenizer
streamer_queue = Queue()
streamer = CustomStreamer(streamer_queue, tokenizer, True)

def start_generation(query):
    prompt = """
            # You are assistant that behaves very professionally. 
            # You will only provide the answer if you know the answer. If you do not know the answer, you will say I dont know. 

            # ###Human: {instruction},
            # ###Assistant: """.format(instruction=query)
  
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda:0")
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=200, temperature=0.1)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()


async def response_generator(query):
    start_generation(query)
    while True:
        value = streamer_queue.get()
        if value == None:
            break
        yield value
        streamer_queue.task_done()
        await asyncio.sleep(0.1)

@app.get('/query-stream/')
async def stream(query: str):
    print(f'Query receieved: {query}')
    return StreamingResponse(response_generator(query), media_type='text/event-stream')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        while True:
            query = await websocket.receive_text()
            print(query)
            encoded_query = quote(query, safe='')
            url = f"http://127.0.0.1:8007/query-stream/?query={encoded_query}"
            async for chunk in fetch_stream(url):
                print("chunk")
                await websocket.send_text(chunk.decode('utf-8'))
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()

import aiohttp

async def fetch_stream(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=None) as response:
            async for chunk in response.content.iter_any():
                yield chunk
