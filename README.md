# Glitched

A webcam filter that runs in your browser. Your face becomes a 3D fox, your voice becomes a woman's, and the
fox can drop into a green wire mesh or glitch out. Record it all to a clip.

It's an experiment. The fox is modelled on Apple's 🦊 emoji.

## Use it

1. Open the page in **Chrome** and click **Start camera**. Allow the camera and the mic.
2. Talk. Your voice comes back as a woman's, live. **Wear headphones**, or the speakers feed back into the mic
   and turn the voice to garble.
3. **🦊 Fox mode** (or `F`) turns on the mask, glitch and hear-my-voice together.
4. Other switches: tap a key to turn it on or off, or hold the key to flip it only while held.
   - `M` fox mask
   - `V` woman's voice
   - `W` wire mesh
   - `G` glitch
5. **Camera** picks which camera to use and switches live.
6. **Voice pitch** is where your voice lands, in Hz (205 by default). It learns your own pitch in the first few
   seconds of talking. **Voice size** lifts the vocal tract; lower sounds more natural.
7. **Record** saves the video and your masked voice.

## How it works

- **Face:** MediaPipe Face Landmarker tracks your face and gives ARKit-style expression values (jaw open, blinks,
  gaze, smile, brows). The fox model carries shape keys with the same names, so each expression drives the fox
  directly.
- **Fox:** a textured head made with image-to-3D, then rigged in Blender by `rig/rig_fox.py`:
  - jaw, blinks, gaze, smile and brows, plus springy ears;
  - the mouth and tongue are smooth separate pieces laid on the muzzle, so the mouth stays clean at every
    opening.
- **Voice:** an AudioWorklet phase vocoder.
  - It moves every harmonic up on its own, so it never has to guess your pitch moment to moment (which is what
    makes simple voice changers break up).
  - It lifts the voice's resonances separately, which is what makes it read as a woman rather than a sped-up man.
  - A noise gate keeps room hiss out, and a leveler keeps words even.

## Files

- `index.html` is the whole app: no build step. three.js and MediaPipe load from CDNs.
- `assets/fox.glb` is the rigged fox.
- `assets/fox-source.png` is the picture the 3D model was generated from.
- `rig/rig_fox.py` rigs a static head .glb into the fox. Run it with the Blender `bpy` module:
  `python3 rig_fox.py in.glb out.glb [preview_dir]`.
