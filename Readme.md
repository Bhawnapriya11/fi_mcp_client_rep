Intelligent MCP Routing System
This project implements a smart routing mechanism between two Modular Command Processor (MCP) servers:

Policy MCP Server — Uses Google Custom Search and Gemini API to answer policy-related queries.

FI MCP Server — Returns financial data such as bank transactions using the Go-based fi-mcp-dev server.

Based on the user prompt, the system intelligently decides which server to call.

.
├── final_mcpserverficlient.py       # Direct client to call the FI MCP server (localhost:8085)
├── integrated_mcp_client.py         # Standalone client for the Policy MCP server using Gemini (localhost:8090)
├── mcp_server_searchagent.py        # FastAPI-based Policy MCP server
├── final_mcp_router.py              # Router client that directs prompt to the correct MCP server
└── README.md                        # Project documentation
Features
Routes user prompts to the appropriate MCP server based on keyword matching

Leverages Google Search API and Gemini for intelligent policy-based responses

Fully compatible with Fi's MCP server for financial queries

Requirements
Install Python dependencies:

pip install fastapi uvicorn requests google-generativeai
Setup Instructions
1. Clone or set up your project directory with all scripts
Ensure all .py files mentioned above are present in the same directory.

2. Start the Policy MCP Server
Run the following command from the terminal:

uvicorn mcp_server_searchagent:app --reload --port 8090
This launches the policy-aware FastAPI server which calls Gemini and Google Search.

3. Start the FI MCP Server
Run the Go-based MCP server (Fi's backend) on port 8085:

go run main.go
Or follow the startup instructions provided in the fi-mcp-dev repository.

4. Run the Routing Client
This client accepts prompts and dynamically routes to either server:

python final_mcp_router.py
Example prompt:


Enter your prompt (or type 'exit'): What are the latest AI regulations in India?
This will be routed to the Policy MCP server.

Testing Each Server Directly
To test the Policy MCP server standalone:

python integrated_mcp_client.py
To test the FI MCP server standalone:

python final_mcpserverficlient.py
Prompt Routing Logic
The router checks if the prompt contains any of the following keywords:

"policy", "government", "regulation", "scheme", "initiative",
"chatbot rules", "AI policy", "digital india", "public sector",
"education policy", "indian government"
If so, it routes the request to the Policy MCP server; otherwise, to the FI MCP server.

Example Prompts
Prompt	Routed To
Show me my last transactions	FI MCP Server
What are India's AI policies in education?	Policy MCP Server
List my savings account balance	FI MCP Server
How is India regulating AI chatbots?	Policy MCP Server

Configuration
Update your API keys in integrated_mcp_client.py:

GOOGLE_API_KEY

SEARCH_ENGINE_ID

GEMINI_API_KEY

These are required for web search and Gemini-based answer generation.

Future Enhancements
Replace keyword matching with Gemini-based intent classification

Add session logging and metrics

Expand to support additional MCP methods like fetch_net_worth

Containerize and deploy both servers with orchestration

Maintainers
Developed to integrate Fi's backend financial services with generative AI capabilities via modular, intelligent routing.