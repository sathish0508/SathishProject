import asyncio
import websockets
import json
import socket
import os
from aiohttp import web

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

HOST_IP = get_local_ip()
HTTP_PORT = int(os.environ.get("PORT", 8081))

connected_clients = set()

async def handle_client(websocket, path):
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            print(f"Received from client: {message}")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.remove(websocket)

async def broadcast_attendance_trigger():
    if connected_clients:
        message = json.dumps({"type": "attendance_trigger"})
        await asyncio.gather(*[client.send(message) for client in connected_clients])
        print("✅ Attendance trigger sent.")
    else:
        print("⚠️ No connected clients.")

async def handle_trigger(request):
    await broadcast_attendance_trigger()
    return web.Response(text="Attendance trigger sent successfully", status=200)

app = web.Application()
app.router.add_post('/trigger', handle_trigger)

async def main():
    websocket_server = await websockets.serve(handle_client, "0.0.0.0", 9098)
    print(f"✅ WebSocket running on ws://0.0.0.0:9098")

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', HTTP_PORT)
    await site.start()
    print(f"✅ HTTP trigger available at http://{HOST_IP}:{HTTP_PORT}/trigger")

    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
