# Rebuilding Glitched from scratch

Everything needed to rebuild the fox, run the app and test the voice is in this repo. The sections below go in
pipeline order.

## 1. The 3D head (image → mesh)

- `assets/fox-source.png` was turned into a textured mesh with **Hunyuan3D v3** on fal:
  - endpoint `fal-ai/hunyuan3d-v3/image-to-3d`
  - input `input_image_url`, with `generate_type: "Normal"` (textured)
  - costs about $0.40–0.55 per run and needs a `FAL_KEY`
- The exact mesh that came back is kept in `source/fox-raw.glb` (about 500k triangles, 4K texture), so you do
  **not** need to regenerate it.
- A fresh generation gives a different mesh. The rig's automatic landmark finding (eyes, nose, mouth line) should
  still work, but check the preview stills.

## 2. Rigging (mesh → expressive fox)

- **Needs:**
  - Python 3.11 with Blender as a module: `pip install bpy==4.2.0` (headless is fine)
  - `numpy`
- **Run:**
  ```
  python3 rig/rig_fox.py source/fox-raw.glb assets/fox.glb previews/
  ```
- **What it does:**
  - normalises the head;
  - welds the seam-split vertices and decimates to about 90k triangles;
  - finds the eyes, nose and mouth line from the texture and shape;
  - adds the ARKit-named shape keys (37+);
  - builds the mouth and tongue as separate smooth meshes;
  - exports `assets/fox.glb` with a 2048 JPEG texture;
  - writes preview stills, including mouth close-ups `m_0.0`–`m_1.0` (Cycles; EEVEE needs a GPU/EGL).
- **Lessons baked in:**
  - Don't cut the mouth into the generated skin: its surface is bumpy, so any cut edge looks jagged. The mouth
    and tongue are smooth curve-built meshes laid on the skin instead.
  - The page nudges those two pieces toward the camera (a vertex shader bias) so the skin can't poke through.
  - Remeshing the head to get clean mouth topology made craters and texture specks. Don't.

## 3. The app

- `index.html` is the whole app, with no build step; serve the folder over HTTPS (or localhost) for camera
  access.
- **Face:**
  - MediaPipe Face Landmarker 0.10.14 (from a CDN);
  - its 52 blendshape names equal the model's shape-key names, so each one drives the fox directly;
  - jaw open has a small dead zone plus easing.
- **Voice** (the `FOX_VOICE` AudioWorklet in the page):
  - a peak-locked phase vocoder: 2048-point FFT, 4× overlap (4096 above 60 kHz);
  - a cepstral envelope (about 1.8 ms lifter) for the formant ("voice size") shift;
  - learns the speaker's average pitch and moves it to a target in Hz (default 205);
  - a noise gate follows the room's noise floor;
  - a compressor plus makeup gain after the EQ.
- **Why not simpler:** the first version was a pitch-synchronous (PSOLA) shifter. It garbled 10–23% of voiced
  speech whenever its pitch tracking slipped, and a fixed +8 semitone shift left deep voices in the male range.

## 4. Testing the voice without a microphone

`tools/voice-bench/` runs the page's worklet offline in Node and scores it:
```
# extract the worklet from the page
python3 -c "s=open('index.html').read(); a=s.index('const FOX_VOICE = \`')+19; open('worklet.js','w').write(s[a:s.index('\`;',a)])"
# 48 kHz mono float32 speech in, processed audio out (target Hz, voice size)
node tools/voice-bench/run.mjs worklet.js speech.f32 out.f32 205 1.18
# roughness in voiced speech (share of garbled frames) and output pitch; the worklet's latency is 2560 samples
python3 tools/voice-bench/rough.py out.f32:2560:speech.f32
```
- Test speech used during development: LibriSpeech clips from `librosa/data` on GitHub:
  - `audio/5703-47212-0000.hq.ogg` (deep male)
  - `audio/3436-172162-0000.hq.ogg` (mid male)
  - `audio/198-209-0000.hq.ogg` (female reference)
- Resample the clips to 48 kHz first.
- Latest scores: 0.4–1.8% garbled frames, output pitch 182–208 Hz for the two male voices.
