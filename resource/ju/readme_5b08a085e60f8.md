# lz-string ver 1.4.4
LZ-based compression algorithm for JavaScript.

## Usage
```typescript
let content: string = 'Some secret texts that need to be compressed.';
let encoded: string = LZString.compressToBase64(content);
let decoded: string = LZString.decompressFromBase64(encoded);

console.log('encoded: ' + encoded);
console.log('decoded: ' + decoded);
```

## Resources
* lz-string on [github](https://github.com/pieroxy/lz-string)
* lz-string [homepage](http://pieroxy.net/blog/pages/lz-string/index.html)