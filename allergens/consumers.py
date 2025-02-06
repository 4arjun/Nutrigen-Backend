import json
from channels.generic.websocket import AsyncWebsocketConsumer

class MyWebSocketConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        await self.channel_layer.group_add("notifications",self.channel_name)


    async def disconnect(self, close_code):
        await self.channel_layer.group_discard("notifications",self.channel_name)

    async def receive(self, text_data):
        # Handle the message received from WebSocket
        text_data_json = json.loads(text_data)
        message = text_data_json['message']

        # Send message to room group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': message
            }
        )

    async def chat_message(self, event):
        # Handle the event sent to the group
        message = event['message']

        # Send message to WebSocket
        await self.send(text_data=json.dumps({
            'message': message
        }))
