import requests
import json

# MCP server URL
mcp_url = "http://localhost:8080/mcp/stream"

# Session ID from login
session_id = "mcp-session-b5901770-45a2-4421-892c-ff873e4dcd06"  # Use the session ID you got after login

# Payload to fetch bank transactions (this can be adjusted to fetch other data, e.g., net worth)
payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "fetch_bank_transactions",  # You can change this to any other method (e.g., fetch_net_worth)
        "arguments": {}
    }
}

# Headers with the session ID
headers = {
    "Content-Type": "application/json",
    "Mcp-Session-Id": session_id  # Pass the session ID in the headers
}

# Send the POST request to the MCP server
response = requests.post(mcp_url, headers=headers, data=json.dumps(payload))
if response.status_code == 200:
    # Parse the response JSON
    response_data = response.json()
    
    # Extracting the content part which contains the bank transaction data
    content = response_data.get('result', {}).get('content', [])
    
    # Format the output in a readable way
    for item in content:
        if 'text' in item:
            # The actual JSON text for the bank transactions
            bank_transactions_data = json.loads(item['text'])
            print("\nBank Transactions:")
            
            # Loop over the banks and transactions
            for bank in bank_transactions_data.get('bankTransactions', []):
                print(f"\nBank: {bank['bank']}")
                print("Transactions:")
                for txn in bank['txns']:
                    transaction = {
                        "Transaction Amount": txn[0],
                        "Transaction Narration": txn[1],
                        "Transaction Date": txn[2],
                        "Transaction Type": txn[3],
                        "Transaction Mode": txn[4],
                        "Current Balance": txn[5]
                    }
                    # Print each transaction in a readable format
                    for key, value in transaction.items():
                        print(f"{key}: {value}")
                    print("-" * 50)
else:
    print("Failed to get data. Status code:", response.status_code)
    print(f"Response: {response.text}")