import requests

# MCP server URL
mcp_url = "http://localhost:8080/mcp/stream"

login_url="https://fi.money/wealth-mcp-login?token=mcp-session-b5901770-45a2-4421-892c-ff873e4dcd06%7C1753531659.vBcvLK9GsNaMu3ZjZ6smVzKyjt64DVcar4eZ68Ol5KQ%3D"
# Phone number from the allowed list (replace this with one from the test data)
phone_number = "2222222222"  # Example phone number from the README

login_params = {
    "phone_number": phone_number  # Include the phone number as a query parameter
}
# Initial payload to get the login URL
login_payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "fetch_bank_transactions",
        "arguments": {}
    }
}
login_response = requests.get(login_url, params=login_params)
if login_response.status_code == 200:
    print("Login successful!")
    # Extract the session ID from the URL (assuming it's included in the response body or URL)
    session_id = login_url.split('token=')[1].split('%7C')[0]  # Extract the session ID from the token
    print(f"Session ID: {session_id}")
else:
    print(f"Login failed. Status code: {login_response.status_code}")
    print(f"Response: {login_response.text}")

# # Send a request to get the login URL
# response = requests.post(mcp_url, json=login_payload)

# # Check if the response is successful
# if response.status_code == 200:
#     data = response.json()
#     # Extract the login URL from the response
#     if "result" in data and "login_url" in data["result"]:
#         login_url = data["result"]["login_url"]
#         print(f"Login URL: {login_url}")
        
#         # Now, simulate logging in using the phone number
#         # Send a GET request to the login URL with the phone number as a parameter
#         login_params = {
#             "phone_number": phone_number  # Include the phone number parameter
#         }
        
#         # Send the GET request to simulate login
#         login_response = requests.get(login_url, params=login_params)
        
#         if login_response.status_code == 200:
#             print("Login successful!")
#             # Extract the sessionId from the login URL
#             session_id = login_url.split('sessionId=')[1]  # Assuming the sessionId is in the URL
#             print(f"Session ID: {session_id}")
#         else:
#             print(f"Login failed. Status code: {login_response.status_code}")
#     else:
#         print("Failed to retrieve login URL. Response:", data)
# else:
#     print(f"Failed to get login URL. Status code: {response.status_code}")
