# socket.io client 2.2
Realtime application framework (client) http://socket.io

## Usage
```typescript
var socket = io('http://localhost');
socket.on('connect', function(){});
socket.on('event', function(data){});
socket.on('disconnect', function(){});
```

## Resources
* socket.io client on [github](https://github.com/socketio/socket.io-client)
* socket.io [homepage](https://socket.io/)