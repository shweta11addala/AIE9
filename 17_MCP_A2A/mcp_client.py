"""
Simple MCP client for the Cat Shop server.
Handles OAuth automatically and tests all tools.
"""
import asyncio
import hashlib
import base64
import secrets
import json
import urllib.parse
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import webbrowser


BASE_URL = "http://localhost:8001"
REDIRECT_URI = "http://localhost:9999/callback"

auth_code = None


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global auth_code
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        auth_code = params.get("code", [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h2>Logged in! You can close this tab and return to the terminal.</h2>")

    def log_message(self, *args):
        pass


def get_token():
    global auth_code

    # 1. Register client
    reg = json.dumps({
        "client_name": "mcp-test-client",
        "redirect_uris": [REDIRECT_URI],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    }).encode()
    req = urllib.request.Request(f"{BASE_URL}/register", data=reg, headers={"Content-Type": "application/json"})
    client = json.loads(urllib.request.urlopen(req).read())
    client_id = client["client_id"]
    client_secret = client["client_secret"]

    # 2. Build PKCE + auth URL
    code_verifier = secrets.token_urlsafe(64)
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).rstrip(b"=").decode()
    state = secrets.token_hex(16)

    params = urllib.parse.urlencode({
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "scope": "read write",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    })
    auth_url = f"{BASE_URL}/authorize?{params}"

    # 3. Start local callback server
    server = HTTPServer(("localhost", 9999), CallbackHandler)
    thread = threading.Thread(target=server.handle_request)
    thread.start()

    # 4. Open browser for login
    print(f"\nOpening browser for login...")
    webbrowser.open(auth_url)
    thread.join(timeout=60)
    server.server_close()

    if not auth_code:
        raise RuntimeError("No auth code received")

    # 5. Exchange code for token
    body = urllib.parse.urlencode({
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": REDIRECT_URI,
        "client_id": client_id,
        "client_secret": client_secret,
        "code_verifier": code_verifier,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/token", data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    token_resp = json.loads(urllib.request.urlopen(req).read())
    return token_resp["access_token"]


def mcp_call(session_id, token, method, params=None):
    payload = json.dumps({
        "jsonrpc": "2.0", "id": 1,
        "method": method,
        "params": params or {}
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/mcp", data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": session_id,
        }
    )
    resp = urllib.request.urlopen(req).read().decode()
    for line in resp.splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:].strip())


def init_session(token):
    payload = json.dumps({
        "jsonrpc": "2.0", "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "mcp-client", "version": "1.0"}
        }
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/mcp", data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
    )
    import http.client
    # Use urllib but capture headers
    class HeaderCapture(urllib.request.HTTPHandler):
        session_id = None
        def http_open(self, req):
            resp = super().http_open(req)
            HeaderCapture.session_id = resp.headers.get("mcp-session-id")
            return resp

    opener = urllib.request.build_opener(HeaderCapture)
    opener.open(req)
    return HeaderCapture.session_id


def call_tool(session_id, token, tool_name, arguments=None):
    result = mcp_call(session_id, token, "tools/call", {
        "name": tool_name,
        "arguments": arguments or {}
    })
    content = result["result"]["content"]
    # Try parsing each content item individually, then combine
    parsed = []
    for c in content:
        if "text" not in c:
            continue
        text = c["text"].strip()
        if not text:
            continue
        try:
            val = json.loads(text)
            if isinstance(val, list):
                parsed.extend(val)
            else:
                parsed.append(val)
        except json.JSONDecodeError:
            pass
    if len(parsed) == 1:
        return parsed[0]
    return parsed


def main():
    print("=== Cat Shop MCP Client ===\n")

    print("Step 1: Getting OAuth token...")
    token = get_token()
    print(f"✅ Authenticated!\n")

    print("Step 2: Starting MCP session...")
    session_id = init_session(token)
    print(f"✅ Session: {session_id}\n")

    print("=" * 40)
    print("📦 LIST ALL PRODUCTS")
    print("=" * 40)
    products = call_tool(session_id, token, "list_products")
    for p in products:
        print(f"  [{p['id']}] {p['name']} - ${p['price']} ({p['category']})")

    print("\n" + "=" * 40)
    print("🔎 SEARCH PRODUCTS: 'salmon'")
    print("=" * 40)
    results = call_tool(session_id, token, "search_products", {"query": "salmon"})
    if isinstance(results, list):
        for p in results:
            print(f"  [{p.get('id', '?')}] {p.get('name', p.get('message', p))}")
    else:
        print(f"  {results}")

    print("\n" + "=" * 40)
    print("🔍 GET PRODUCT #1")
    print("=" * 40)
    product = call_tool(session_id, token, "get_product", {"product_id": 1})
    print(f"  {product['name']}: {product['description']} - ${product['price']}")

    print("\n" + "=" * 40)
    print("🛒 ADD TO CART")
    print("=" * 40)
    result = call_tool(session_id, token, "add_to_cart", {"product_id": 1, "quantity": 2})
    print(f"  {result['message']}")
    result = call_tool(session_id, token, "add_to_cart", {"product_id": 6, "quantity": 1})
    print(f"  {result['message']}")

    print("\n" + "=" * 40)
    print("👀 VIEW CART")
    print("=" * 40)
    cart = call_tool(session_id, token, "view_cart")
    for item in cart["items"]:
        print(f"  {item['name']} x{item['quantity']} = ${item['subtotal']}")
    print(f"  TOTAL: ${cart['total']}")

    print("\n" + "=" * 40)
    print("❌ REMOVE ITEM #6")
    print("=" * 40)
    result = call_tool(session_id, token, "remove_from_cart", {"product_id": 6})
    print(f"  {result['message']}")

    print("\n" + "=" * 40)
    print("💳 CHECKOUT")
    print("=" * 40)
    order = call_tool(session_id, token, "checkout")
    print(f"  {order['message']}")
    print(f"  Order ID: {order['order_id']} | Total: ${order['total']}")

    print("\n✅ All tools tested successfully!")


if __name__ == "__main__":
    main()
