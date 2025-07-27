import requests
import google.generativeai as genai
from typing import List, Dict
import json

# Configuration
GOOGLE_API_KEY = ""  # Replace with your Google Custom Search API key
SEARCH_ENGINE_ID = ""  # Replace with your Google Custom Search Engine ID
GEMINI_API_KEY = ""  # Replace with your Gemini API key

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def web_search(query: str) -> List[Dict[str, str]]:
        
    try:
        # Append India and policy-related terms to focus the search
        #enhanced_query = f"{query} site:*.in | site:*.gov.in AI policies India chatbots"
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": SEARCH_ENGINE_ID,
            "q": query,
            "num": 3
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("items", [])
        return [
            {
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                "url": result.get("link", "")
            } for result in results
        ]
    except requests.RequestException as e:
        print(f"Web search error: {str(e)}")
        return []

def call_gemini(prompt: str, context: str, conversation_history: str) -> str:
    
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        full_prompt = (
            f"You are an expert on Indian government and private policies realted to social good and development "
            f"Conversation history:\n{conversation_history}\n\n"
            f"Current query: {prompt}\n\n"
            f"Web search context:\n{context}\n\n"
            f"Provide a concise, accurate answer focusing on recent Indian policies or developments "
            f"related to intelligent chat-based systems. If the context lacks policy details, "
            f"state that and provide a general answer based on your knowledge. Cite URLs from the context if relevant."
        )
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini API error: {str(e)}"

def chat_agent():
   
    conversation_history = []
    
    print("Welcome to the Policy Chat Agent! Ask about Indian policies. Type 'exit' to quit.")
    
    while True:
        user_prompt = input("\nYour prompt: ").strip()
        if user_prompt.lower() == "exit":
            print("Goodbye!")
            break
        
        # Perform web search
        search_results = web_search(user_prompt)
        context = ""
        if not search_results:
            context = "No relevant web search results found."
        else:
            for i, result in enumerate(search_results, 1):
                context += f"Result {i}:\nTitle: {result['title']}\nSnippet: {result['snippet']}\nURL: {result['url']}\n\n"
        
        # Format conversation history
        history_text = "\n".join([f"User: {h['prompt']}\nAI: {h['response']}" for h in conversation_history[-3:]])  # Keep last 3 exchanges
        
        # Generate answer
        answer = call_gemini(user_prompt, context, history_text)
        
        # Store in history
        conversation_history.append({"prompt": user_prompt, "response": answer})
        
        # Print response
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    chat_agent()
    
"""

# import requests
# import google.generativeai as genai
# from typing import List, Dict
# import json
# from fastapi import FastAPI, Request, HTTPException, Header
# from pydantic import BaseModel
# import uvicorn

# # Configuration
# GOOGLE_API_KEY = "AIzaSyDYPRZRkn5Dgmez17Krqket5a8IOY1OLkE"  # Replace with your Google Custom Search API key
# SEARCH_ENGINE_ID = "a56d28eb5509e451c"  # Replace with your Google Custom Search Engine ID
# GEMINI_API_KEY = "AIzaSyDPCjJaJQUwjtFbTOCM_Yxali_hl5a4IR8"  # Replace with your Gemini API key

# # Configure Gemini API
# genai.configure(api_key=GEMINI_API_KEY)

# # FastAPI app
# app = FastAPI()

# # In-memory conversation history per session
# conversation_histories = {}

# class JsonRpcRequest(BaseModel):
#     jsonrpc: str
#     id: int
#     method: str
#     params: Dict

# def web_search(query: str) -> List[Dict[str, str]]:
#     """
#     Search the web using Google Custom Search JSON API for Indian AI policies.
#     Returns up to 3 results with title, snippet, and URL.
#     """
#     try:
#         url = "https://www.googleapis.com/customsearch/v1"
#         params = {
#             "key": GOOGLE_API_KEY,
#             "cx": SEARCH_ENGINE_ID,
#             "q": query,
#             "num": 3
#         }
#         response = requests.get(url, params=params)
#         response.raise_for_status()
#         results = response.json().get("items", [])
#         return [
#             {
#                 "title": result.get("title", ""),
#                 "snippet": result.get("snippet", ""),
#                 "url": result.get("link", "")
#             } for result in results
#         ]
#     except requests.RequestException as e:
#         print(f"Web search error: {str(e)}")
#         return []

# def call_gemini(prompt: str, context: str, conversation_history: str) -> str:
#     """
#     Call Gemini API to generate a policy-focused answer based on prompt, context, and history.
#     """
#     try:
#         model = genai.GenerativeModel("gemini-2.5-pro")
#         full_prompt = (
#             f"You are an expert on Indian government and private policies related to social good and development "
#             f"Conversation history:\n{conversation_history}\n\n"
#             f"Current query: {prompt}\n\n"
#             f"Web search context:\n{context}\n\n"
#             f"Provide a concise, accurate answer focusing on recent Indian policies or developments "
#             f"related to intelligent chat-based systems. If the context lacks policy details, "
#             f"state that and provide a general answer based on your knowledge. Cite URLs from the context if relevant."
#         )
#         response = model.generate_content(full_prompt)
#         return response.text.strip()
#     except Exception as e:
#         return f"Gemini API error: {str(e)}"

# @app.post("/mcp/stream")
# async def mcp_stream(request: JsonRpcRequest, mcp_session_id: str = Header(None)):
#     """
#     MCP endpoint to handle JSON-RPC requests for the chat agent.
#     Requires Mcp-Session-Id header for authentication.
#     """
#     if not mcp_session_id:
#         raise HTTPException(status_code=401, detail="Session ID required")

#     # Initialize conversation history for the session if not exists
#     if mcp_session_id not in conversation_histories:
#         conversation_histories[mcp_session_id] = []

#     if request.jsonrpc != "2.0":
#         raise HTTPException(status_code=400, detail="Invalid JSON-RPC version")

#     if request.method == "chat_agent":
#         user_prompt = request.params.get("prompt", "")
#         if not user_prompt:
#             raise HTTPException(status_code=400, detail="Prompt is required")

#         # Perform web search
#         search_results = web_search(user_prompt)
#         context = "No relevant web search results found." if not search_results else "\n".join(
#             f"Result {i}:\nTitle: {result['title']}\nSnippet: {result['snippet']}\nURL: {result['url']}\n"
#             for i, result in enumerate(search_results, 1)
#         )

#         # Format conversation history
#         history_text = "\n".join(
#             f"User: {h['prompt']}\nAI: {h['response']}"
#             for h in conversation_histories[mcp_session_id][-3:]
#         )

#         # Generate answer
#         answer = call_gemini(user_prompt, context, history_text)

#         # Store in history
#         conversation_histories[mcp_session_id].append({"prompt": user_prompt, "response": answer})

#         # Return JSON-RPC response
#         return {
#             "jsonrpc": "2.0",
#             "id": request.id,
#             "result": {
#                 "content": [{"text": answer}]
#             }
#         }
#     else:
#         raise HTTPException(status_code=400, detail="Unsupported method")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8080)
