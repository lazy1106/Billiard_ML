<a href="cg://source/CG.Server/README.md" class="mat-raised-button mat-primary">Back</a>
# GLT Server

The GLT server provides a database for your games that helps manage users, scores, and states.

## Getting Started

To initialize the library.
```typescript
CG.Server.onReady((user: CG.GLT.commands.vo.User) => {
    // The library is ready, and have the user ready.
    if(user.isLocalGuest()) {
        // if the user has not signed in, he is treated as a local guest
        console.log('You are not signed in yet.');
    } else {
        console.log('Welcome back ' + user.nickname);
    }
});
// Optionally, you can place a default user panel on the top-right corner of the app.
// This AuthPanel privodes a interface to help user sign in, edit profile etc.
CG.React.renderComponent(AuthPanel);
```

If player must login, do the followings.
```typescript
CG.Server.onReadyAndAuth((user: CG.GLT.commands.vo.User) => {
    if(user.isLocalGuest()) {
        console.log('You are not signed in yet.');
    } else {
        console.log('Welcome back ' + user.nickname);
    }
});
// tell auth panel to remove the option of login as a guest
CG.React.renderComponent(AuthPanel, {
    mustLogin: true,
    disableGuest: true
});
```

To manually open the login or profile window.
```typescript
// to open the window, we need to provide the functions a click event of a button
CG.GLT.auth.loginUser(event)
    .then(user => {
        console.log('Hello ' + user.nickname);
    })
    .catch(error => {
        console.error('login failed: ' + error);
    });

CG.GLT.auth.userProfile(event)
    .then(user => {
        console.log('Hello again ' + user.nickname);
    });
```

To access score service.
```typescript
/*
 submit a score of 10 in 'challenge' category, also specify that
 - use the highest score on leaderboard
 - this score will last forever (TimeRange.HISTORY)
 */
CG.GLT.commands.scoreService.submitScore('challenge', 10, SubmitType.KEEP_HIGHEST, TimeRange.HISTORY).submit();

/*
 Alternatively, you can open a dialog to help you submit the score,
 and to show the submit progress.
 This dialog also provides login button if the user has not logged in.
 */
CG.Server.components.showSubmitScoreDialog('You got 10 scores', 'challenge', 10, SubmitType.KEEP_HIGHEST, TimeRange.HISTORY, true)
    .then(() => {
        console.log('score submitted');
    });

// show a leaderboard of top 10
CG.Server.components.showHighScoreBoard('challenge', [TimeRange.HISTORY], OrderType.HIGH_TO_LOW, 10);
```

To access the state service.
```typescript
// set a user state
CG.GLT.commands.stateService.setState(user.username, 'profile', 'favorite food', 'Lynel horn').submit();

// get a user state
CG.GLT.commands.stateService.getState(user.username, 'profile', 'favorite food', userState => {
    console.log('userState: ' + userState.key + ' => ' + userState.value);
}).submit();
```

## Authors
**[Haskasu](/profile/Haskasu)**