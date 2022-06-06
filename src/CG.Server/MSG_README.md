<a href="cg://source/CG.Server/README.md" class="mat-raised-button mat-primary">Back</a>
# MSG Server

The MSG Server provide multiplayer online connection for your game.

Players can create game room, join game room, and broadcast messages.

## Getting Started

To initialize the msg server connection.
```typescript
/*
When initializing msg-client, you need to provide a version of your message system.
Players can only see other players in the same version of msg-client.
When your message definition changes, you need to give a different msg-version,
so that the players in the old version of game do not receive unknown messages.
*/
CG.Server.msg.initialize('v1');
```
*Be awared that the msg-client will not connect until the user sign in glt-server.*

The msg-client will start the connection when initializing, and login msg-server with
the user info provided by glt-server. Listen to LOGIN event to start communicating with
msg-server.

```typescript
GLT.Server.msg.once(LoginEvent.TYPE, (event: LoginEvent) => {
    // the player are loggedin, and gets a unique code that is different every time he plays the game
    console.log('My player code this time is ' + event.player.code);
});
```

To access the gameroom service.

```typescript
// create a new game room
GLT.Server.msg.gameroom.createRoom(
    'hellozone',    // the game zone it belongs to 
    'cool room',// the name of the room
    null,       // the password
    3,          // the max players it can contain
    1,          // the initial state
    {},         // the initial state json
    true        // allow guest
)
.then(fullroom => {
    // the room has a unique code
    console.log('create room: ' + fullroom.room.code);
});

// to list 10 rooms with a filter that queries only the rooms with state==1
GLT.Server.msg.gameroom.listRooms('hellozone', 10, [PropFilter.equal('state', 1)])
.then(list => {
    // the room has a unique code
    console.log('we found ' + list.length + ' rooms');
});

// to join a room
GLT.Server.msg.gameroom.joinRoom(roomCode)
.then(fullroom => {
    console.log('now i am in a room: ' + fullroom.room.code);
});

// listenToZone to receive RoomCreatedEvent, RoomClosedEvent, and BroadcastEvent.BROADCAST_TO_ZONE
msg.gameroom.listenToZone('hellozone');

msg.on(RoomCreatedEvent.TYPE, (event: RoomCreatedEvent) => {
    console.log('room created: ' + event.roomAndStatus.room.code);
});
```

To broadcast messages.

```typescript
// send messages to all players in the room that I am currently in.
GLT.Server.msg.gameroom.broadcastInRoom({target: 'enemy', skill: 'kick', damage: 100});

// send messages to all players in the room that I am currently in.
GLT.Server.msg.gameroom.broadcastToRoom({love: 100, give: 'ring'});

// and to receive messages
GLT.Server.msg.on(BroadcastEvent.TYPE.BROADCAST_IN_ROOM, (event: BroadcastEvent) => {
    console.log('data received: ' + event.data);
});
```

## Authors
**[Haskasu](/profile/Haskasu)**