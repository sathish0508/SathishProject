from aiohttp import web
import asyncio
import json
import socket
import aiohttp

connected_clients = set()

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    connected_clients.add(ws)
    
    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            print(f"Received: {msg.data}")
        elif msg.type == aiohttp.WSMsgType.ERROR:
            print('ws connection closed with exception %s' % ws.exception())
    
    connected_clients.remove(ws)
    return ws

async def trigger_handler(request):
    message = json.dumps({"type": "attendance_trigger"})
    for ws in connected_clients:
        await ws.send_str(message)
    return web.Response(text="Attendance trigger sent")

app = web.Application()
app.router.add_get('/ws', websocket_handler)
app.router.add_post('/trigger', trigger_handler)

if __name__ == '__main__':
    web.run_app(app, port=8080)
