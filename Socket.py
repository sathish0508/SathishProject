import asyncio
import websockets
import json
import socket
from aiohttp import web

# Get current machine IP address
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # This does not need to be reachable, just forces IP detection
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

HOST_IP = get_local_ip()

# Store connected clients
connected_clients = set()

# WebSocket handler
async def handle_client(websocket, path):
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            print(f"Received from client: {message}")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.remove(websocket)

# Broadcast trigger
async def broadcast_attendance_trigger():
    if connected_clients:
        message = json.dumps({"type": "attendance_trigger"})
        await asyncio.gather(*[client.send(message) for client in connected_clients])
        print("✅ Attendance trigger sent.")
    else:
        print("⚠️ No connected clients.")

# HTTP POST trigger
async def handle_trigger(request):
    await broadcast_attendance_trigger()
    return web.Response(text="Attendance trigger sent successfully", status=200)

# aiohttp setup
app = web.Application()
app.router.add_post('/trigger', handle_trigger)

# Start servers
async def main():
    websocket_server = await websockets.serve(handle_client, HOST_IP, 9098)
    print(f"✅ WebSocket running on ws://{HOST_IP}:9098")

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8081)
    await site.start()
    print(f"✅ HTTP trigger available at http://{HOST_IP}:8081/trigger")

    await asyncio.Event().wait()

# Run
await main()
