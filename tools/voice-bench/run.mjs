// Runs a FoxVoice worklet source offline: node run.mjs <worklet.js> <in.f32> <out.f32> [targetHz] [formant]
import fs from 'node:fs';
const [,, src, inp, outp, semis = '8', formant = '1.17'] = process.argv;
let Cls; globalThis.sampleRate = 48000;
globalThis.AudioWorkletProcessor = class {}; globalThis.registerProcessor = (n, c) => { Cls = c; };
new Function(fs.readFileSync(src, 'utf8'))();
const node = new Cls();
const x = new Float32Array(fs.readFileSync(inp).buffer.slice(0));
const y = new Float32Array(x.length + 48000);
const params = { pitch: new Float32Array([Math.pow(2, +semis / 12)]), target: new Float32Array([+semis > 50 ? +semis : 205]), formant: new Float32Array([+formant]) };
const t0 = performance.now();
for (let i = 0; i + 128 <= y.length; i += 128) {
  const inb = new Float32Array(128); if (i < x.length) inb.set(x.subarray(i, Math.min(i + 128, x.length)));
  const out = [[new Float32Array(128)]];
  node.process([[inb]], out, params);
  y.set(out[0][0], i);
}
const ms = performance.now() - t0;
fs.writeFileSync(outp, Buffer.from(y.buffer));
console.error(`processed ${(x.length / 48000).toFixed(1)}s in ${ms.toFixed(0)}ms (${(ms / (x.length / 48)).toFixed(3)} of realtime)`);
