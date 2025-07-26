    import requests
    login_url = "https://fi.money/wealth-mcp-login?token=mcp-session-b5901770-45a2-4421-892c-ff873e4dcd06%7C1753531659.vBcvLK9GsNaMu3ZjZ6smVzKyjt64DVcar4eZ68Ol5KQ%3D"

    phone_number = "2222222222"  

    # Simulate login
    login_params = {
        "phone_number": phone_number
    }

    # Send the GET request to simulate the login
    login_response = requests.get(login_url, params=login_params)

    print(login_response,'+++++')
    # Check if login was successful
    if login_response.status_code == 200:
        print("Login successful!")
        # Extract the session ID from the URL (assuming it's included in the response body or URL)
        session_id = login_url.split('token=')[1].split('%7C')[0]  # Extract the session ID from the token
        print(f"Session ID: {session_id}")
    else:
        print(f"Login failed. Status code: {login_response.status_code}")
        print(f"Response: {login_response.text}")
