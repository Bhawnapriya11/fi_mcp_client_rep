import requests
import json

# Ports
FI_MCP_URL = "http://localhost:8085/mcp/stream"
POLICY_MCP_URL = "http://localhost:8090/mcp/stream"

# Simulated session ID
session_id = "mcp-session-b5901770-45a2-4421-892c-ff873e4dcd06"

# Keywords that indicate a policy-related query
policy_keywords = [
    "policy", "government", "regulation", "scheme", "initiative", "chatbot rules",
    "AI policy", "digital india", "public sector", "education policy", "indian government"
]

def is_policy_related(prompt: str) -> bool:
    return any(keyword.lower() in prompt.lower() for keyword in policy_keywords)

def route_prompt(prompt: str):
    target_url = POLICY_MCP_URL if is_policy_related(prompt) else FI_MCP_URL

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "fetch_bank_transactions",
            "arguments": {
                "query": prompt
            }
        }
    }

    headers = {
        "Content-Type": "application/json",
        "Mcp-Session-Id": session_id
    }

    print(f"\n🔁 Routing to: {'Policy MCP Server' if target_url == POLICY_MCP_URL else 'Fi MCP Server'}")
    response = requests.post(target_url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        response_data = response.json()
        content = response_data.get('result', {}).get('content', [])

        print("\n📋 Response:")
        for item in content:
            if 'text' in item:
                data = json.loads(item['text'])
                for bank in data.get("bankTransactions", []):
                    print(f"\n🏦 Bank: {bank['bank']}")
                    for txn in bank['txns']:
                        transaction = {
                            "Transaction Amount": txn[0],
                            "Narration": txn[1],
                            "Date": txn[2],
                            "Type": txn[3],
                            "Mode": txn[4],
                            "Balance": txn[5]
                        }
                        for k, v in transaction.items():
                            print(f"{k}: {v}")
                        print("-" * 40)
    else:
        print(f"❌ Failed with status {response.status_code}")
        print(response.text)


# ========== MAIN ENTRY ==========
if __name__ == "__main__":
    while True:
        user_prompt = input("\n🧠 Enter your prompt (or type 'exit'): ").strip()
        if user_prompt.lower() == "exit":
            break
        route_prompt(user_prompt)
