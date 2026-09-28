"""Rig a static textured fox head (.glb from image-to-3D) with ARKit-named face shapes.

python3 rig_fox.py in.glb out.glb [preview_dir]

- normalises the head: front faces glTF +Z, height 1, centred
- finds eyes + nose from the texture (dark pixels on the front surface)
- cuts a mouth slit under the nose and adds a dark mouth interior behind it
- adds ARKit-named shape keys (MediaPipe emits the same names), plus ear keys
- exports a .glb with morph targets, and optionally renders preview stills
"""
import sys, math
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

inp, out = sys.argv[1], sys.argv[2]
preview = sys.argv[3] if len(sys.argv) > 3 else None

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=inp)
meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
for o in bpy.context.scene.objects: o.select_set(False)
for o in meshes: o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1: bpy.ops.object.join()
head = bpy.context.view_layer.objects.active
head.name = 'FoxHead'
bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
for o in list(bpy.context.scene.objects):
    if o != head: bpy.data.objects.remove(o, do_unlink=True)
# weld the vertices the importer split along texture seams, so the skin is one continuous surface
# (UVs live on face corners, so welding keeps the texture mapping intact)
_bm = bmesh.new(); _bm.from_mesh(head.data)
nb = len(_bm.verts); bmesh.ops.remove_doubles(_bm, verts=_bm.verts, dist=1e-5)
print('welded seams:', nb, '->', len(_bm.verts), 'verts; open edges left:', sum(1 for e in _bm.edges if len(e.link_faces) == 1))
_bm.to_mesh(head.data); _bm.free(); head.data.update()
# web weight: ~90k triangles (face shapes add a full copy of positions per shape)
TARGET_TRIS = 90000
ntris = sum(len(p.vertices) - 2 for p in head.data.polygons)
if ntris > TARGET_TRIS:
    mod = head.modifiers.new('dec', 'DECIMATE'); mod.ratio = TARGET_TRIS / ntris; mod.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier='dec')
    print('decimated', ntris, '->', sum(len(p.vertices) - 2 for p in head.data.polygons))
me = head.data

# ---------------------------------------------------------------- texture colour per vertex
def base_image(obj):
    for slot in obj.material_slots:
        m = slot.material
        if not m or not m.use_nodes: continue
        for n in m.node_tree.nodes:
            if n.type == 'BSDF_PRINCIPLED':
                link = n.inputs['Base Color'].links
                if link and link[0].from_node.type == 'TEX_IMAGE': return link[0].from_node.image
        for n in m.node_tree.nodes:
            if n.type == 'TEX_IMAGE' and n.image: return n.image
    return None

def vertex_colors(obj):
    img = base_image(obj)
    nv = len(obj.data.vertices)
    if img is None or not obj.data.uv_layers: return np.full((nv, 3), 0.5)
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, img.channels)[:, :, :3]
    uv = np.zeros(len(obj.data.loops) * 2); obj.data.uv_layers.active.data.foreach_get('uv', uv); uv = uv.reshape(-1, 2)
    vi = np.zeros(len(obj.data.loops), dtype=np.int64); obj.data.loops.foreach_get('vertex_index', vi)
    xs = np.clip((uv[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(int); ys = np.clip((uv[:, 1] % 1.0) * (h - 1), 0, h - 1).astype(int)
    c = px[ys, xs]
    acc = np.zeros((nv, 3)); cnt = np.zeros(nv)
    np.add.at(acc, vi, c); np.add.at(cnt, vi, 1)
    return acc / np.maximum(cnt, 1)[:, None]

def coords():
    co = np.zeros(len(me.vertices) * 3); me.vertices.foreach_get('co', co); return co.reshape(-1, 3)

def set_coords(a):
    me.vertices.foreach_set('co', a.reshape(-1)); me.update()

# ---------------------------------------------------------------- normalise: Blender -Y = front (glTF +Z), Z up, height 1
co = coords()
mn, mx = co.min(0), co.max(0)
co = (co - (mn + mx) / 2) / (mx[2] - mn[2])
set_coords(co)
col = vertex_colors(head)
lum = col @ np.array([0.3, 0.59, 0.11])
dark = lum < 0.12
# the snout sticks out furthest: the extreme |y| point in the lower-centre column is the front
col_mask = (np.abs(co[:, 0]) < 0.12 * (co[:, 0].max() - co[:, 0].min())) & (co[:, 2] < 0.1)
yy = co[col_mask, 1] if col_mask.sum() > 10 else co[:, 1]
side = 1.0 if yy.max() > -yy.min() else -1.0
if side > 0:
    co[:, 0] *= -1; co[:, 1] *= -1; set_coords(co)   # rotate 180 about Z
print('front flipped:', side > 0, 'dark verts:', int(dark.sum()))

nrm = np.zeros(len(me.vertices) * 3); me.vertices.foreach_get('normal', nrm); nrm = nrm.reshape(-1, 3)
front = (nrm[:, 1] < -0.25) & (co[:, 1] < 0.0)
width = co[:, 0].max() - co[:, 0].min()

# ---------------------------------------------------------------- features from the texture
d = dark & front
nose_c = co[d][np.argmin(co[d][:, 1])]                        # most forward dark point = nose tip
nose = d & (np.linalg.norm(co[:, [0, 2]] - nose_c[[0, 2]], axis=1) < 0.09)
# eyes are dark blue, not black: a looser threshold, upper face only, away from the nose
eyes = (lum < 0.33) & front & ~nose & (co[:, 2] > nose_c[2] + 0.08) & (np.abs(co[:, 0]) > 0.04)
def blob(mask):
    p = co[mask]; c = (p.min(0) + p.max(0)) / 2; r = (p.max(0) - p.min(0)) / 2
    return c, np.maximum(r, 0.015)
eyeL_c, eyeL_r = blob(eyes & (co[:, 0] > 0))                  # subject's left = +X when facing -Y
eyeR_c, eyeR_r = blob(eyes & (co[:, 0] < 0))
nose_min_z = co[nose][:, 2].min()
print('nose', nose_c.round(3), 'eyeL', eyeL_c.round(3), eyeL_r.round(3), 'eyeR', eyeR_c.round(3))

# muzzle: light fur around and below the nose
light = lum > 0.55
muzzle_w = 0.5 * (np.ptp(co[front & light & (co[:, 2] < nose_c[2]) , 0]) if (front & light & (co[:, 2] < nose_c[2])).sum() > 20 else width * 0.4)
# the lip line: the deepest groove in the centre column a little below the nose
col = front & (np.abs(co[:, 0]) < 0.02) & (co[:, 2] < nose_min_z - 0.03) & (co[:, 2] > nose_min_z - 0.11)
if col.sum() > 8:
    order = np.argsort(co[col, 2]); zc, yc = co[col, 2][order], co[col, 1][order]
    mouth_z = float(zc[np.argmax(yc)])            # most recessed point (largest y = furthest back)
else:
    mouth_z = nose_min_z - 0.07
print('mouth line z', round(mouth_z, 3), 'nose bottom z', round(float(nose_min_z), 3))
mouth_hw = min(muzzle_w * 0.55, abs(eyeL_c[0] - eyeR_c[0]) * 0.55)

co = coords()
nrm = np.zeros(len(me.vertices) * 3); me.vertices.foreach_get('normal', nrm); nrm = nrm.reshape(-1, 3)
front = (nrm[:, 1] < -0.2) & (co[:, 1] < 0.05)

def smooth(t): t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)
def ellipse_w(c, r, grow=1.0, core=0.0):
    # eyes sit in sockets whose walls face sideways, so select by position (front half), not normal
    dd = np.sqrt(((co[:, 0] - c[0]) / r[0]) ** 2 + ((co[:, 2] - c[2]) / r[2]) ** 2)
    return smooth((grow - dd) / max(grow - core, 1e-3)) * (co[:, 1] < c[1] + 0.12)

# ---------------------------------------------------------------- shape keys
head.shape_key_add(name='Basis', from_mix=False)
skin_disp = {}
def key(name, disp):
    k = head.shape_key_add(name=name, from_mix=False)
    k.data.foreach_set('co', (co + disp).reshape(-1)); skin_disp[name] = disp; return k

zero = np.zeros_like(co)
# jaw: everything under the mouth line in the muzzle hinges open about a point behind the muzzle
near0 = (np.abs(co[:, 0]) < 0.03) & (np.abs(co[:, 2] - mouth_z) < 0.03) & front
surf_y = co[near0, 1].min() if near0.sum() else nose_c[1] + 0.05
# only the front of the lower muzzle and chin swings; the back of the head stays put
prof = np.cos(np.pi / 2 * np.clip(np.abs(co[:, 0]) / mouth_hw, 0, 1)) ** 1.3     # round: centre opens most, corners stay shut
below = smooth((mouth_z - co[:, 2]) / 0.025)                    # skin under the lip line: 0 at the line, 1 a little below
jaw_w = (prof * below
         * smooth((surf_y + 0.28 - co[:, 1]) / 0.12) * smooth((co[:, 2] - (mouth_z - 0.2)) / 0.06))
pivot = np.array([0, surf_y + 0.24, mouth_z + 0.02])
def rot_x(p, ang):
    c, s = math.cos(ang), math.sin(ang); q = p - pivot
    return pivot + np.stack([q[:, 0], q[:, 1] * c - q[:, 2] * s, q[:, 1] * s + q[:, 2] * c], 1)
jaw_d = (rot_x(co, math.radians(27)) - co) * jaw_w[:, None]   # +angle: the front of the jaw drops (~0.08 at the lip)
up_w = prof * smooth(1 - (co[:, 2] - mouth_z) / 0.05) * (co[:, 1] < surf_y + 0.15) * (co[:, 2] >= mouth_z)
jaw_d[:, 2] += 0.016 * up_w                                  # the upper lip lifts a little as the mouth opens
key('jawOpen', jaw_d)
key('jawLeft', np.stack([jaw_w * 0.04, 0 * jaw_w, 0 * jaw_w], 1))
key('jawRight', np.stack([-jaw_w * 0.04, 0 * jaw_w, 0 * jaw_w], 1))

for side_name, c, r in (('Left', eyeL_c, eyeL_r), ('Right', eyeR_c, eyeR_r)):
    w = ellipse_w(c, r * 1.1, 1.9, 1.05)
    out_dir = 1.0 if side_name == 'Left' else -1.0
    # close: the eye sits in a socket, so bring the eye surface out flush with the fur around it
    # while collapsing it onto a lid line a little below centre
    dd = np.sqrt(((co[:, 0] - c[0]) / r[0]) ** 2 + ((co[:, 2] - c[2]) / r[2]) ** 2)
    ring = (dd > 1.35) & (dd < 1.7) & front
    ring_y = np.median(co[ring, 1]) if ring.sum() > 5 else c[1]
    inner = smooth((1.35 - dd) / 0.3) * (co[:, 1] < c[1] + 0.12)
    lid_z = c[2] - r[2] * 0.2
    d = zero.copy(); d[:, 2] = (lid_z - co[:, 2]) * np.maximum(w * 0.97, inner)
    d[:, 1] = (ring_y - 0.004 - co[:, 1]) * inner
    key('eyeBlink' + side_name, d)
    d = zero.copy(); d[:, 2] = (co[:, 2] - c[2]) * 0.3 * w; key('eyeWide' + side_name, d)
    d = zero.copy(); d[:, 2] = (c[2] - co[:, 2]) * 0.45 * w; key('eyeSquint' + side_name, d)
    g = ellipse_w(c, r, 1.8, 1.0)
    for look, vec in (('Out', (out_dir, 0)), ('In', (-out_dir, 0)), ('Up', (0, 1)), ('Down', (0, -1))):
        d = zero.copy(); d[:, 0] = vec[0] * r[0] * 0.55 * g; d[:, 2] = vec[1] * r[2] * 0.3 * g
        key('eyeLook' + look + side_name, d)
    # brow above each eye
    bw = smooth(1 - np.sqrt(((co[:, 0] - c[0]) / (r[0] * 2.0)) ** 2 + ((co[:, 2] - (c[2] + r[2] * 2.0)) / (r[2] * 1.2)) ** 2)) * front
    d = zero.copy(); d[:, 2] = -0.02 * bw; key('browDown' + side_name, d)
    d = zero.copy(); d[:, 2] = 0.025 * bw; key('browOuterUp' + side_name, d)
    # smile: the mouth corner on that side lifts and pulls back
    cw = smooth(1 - np.sqrt(((co[:, 0] - out_dir * mouth_hw) / (mouth_hw * 0.7)) ** 2 + ((co[:, 2] - mouth_z) / 0.06) ** 2)) * front
    d = zero.copy(); d[:, 2] = 0.025 * cw; d[:, 0] = out_dir * 0.015 * cw; d[:, 1] = 0.01 * cw; key('mouthSmile' + side_name, d)
    d = zero.copy(); d[:, 2] = -0.02 * cw; key('mouthFrown' + side_name, d)
    # cheek puff on that side
    chw = smooth(1 - np.sqrt(((co[:, 0] - out_dir * width * 0.3) / (width * 0.18)) ** 2 + ((co[:, 2] - mouth_z) / 0.14) ** 2))
    d = zero.copy(); d[:, 0] = out_dir * 0.03 * chw; key('cheekSquint' + side_name, d * 0.5)
    nw = smooth(1 - np.linalg.norm((co[:, [0, 2]] - np.array([out_dir * 0.04, nose_c[2] + 0.02])) / 0.07, axis=1)) * front
    d = zero.copy(); d[:, 2] = 0.015 * nw; key('noseSneer' + side_name, d)

bi = smooth(1 - np.sqrt((co[:, 0] / (abs(eyeL_c[0] - eyeR_c[0]) * 0.6)) ** 2 + ((co[:, 2] - (eyeL_c[2] + eyeL_r[2] * 2.2)) / 0.08) ** 2)) * front
d = zero.copy(); d[:, 2] = 0.03 * bi; key('browInnerUp', d)
pw = smooth(1 - np.sqrt((co[:, 0] / (mouth_hw * 1.2)) ** 2 + ((co[:, 2] - mouth_z) / 0.09) ** 2)) * front
d = zero.copy(); d[:, 0] = -co[:, 0] * 0.35 * pw; d[:, 1] = -0.03 * pw; key('mouthPucker', d)
d = zero.copy(); d[:, 0] = -co[:, 0] * 0.2 * pw; d[:, 1] = -0.02 * pw; d[:, 2] = (co[:, 2] - mouth_z) * 0.3 * pw; key('mouthFunnel', d)
cheeks = smooth(1 - np.minimum(np.abs(np.abs(co[:, 0]) - width * 0.3) / (width * 0.18), 1)) * smooth(1 - np.abs(co[:, 2] - mouth_z) / 0.16)
d = zero.copy(); d[:, 0] = np.sign(co[:, 0]) * 0.035 * cheeks; key('cheekPuff', d)

# ears: the top of the head, split by side, tilt back (driven by head motion in the page)
top = co[:, 2].max()
for side_name, s in (('Left', 1.0), ('Right', -1.0)):
    ew = smooth((co[:, 2] - (top - 0.34)) / 0.2) * smooth((s * co[:, 0] - 0.05) / 0.08)
    base = np.array([s * width * 0.25, 0.0, top - 0.34])
    q = co - base; ang = math.radians(-28)
    rot = np.stack([q[:, 0], q[:, 1] * math.cos(ang) - q[:, 2] * math.sin(ang), q[:, 1] * math.sin(ang) + q[:, 2] * math.cos(ang)], 1) + base
    key('earBack' + side_name, (rot - co) * ew[:, None])
    ang = math.radians(18) * s
    rot = np.stack([q[:, 0] * math.cos(ang) - q[:, 2] * math.sin(ang), q[:, 1], q[:, 0] * math.sin(ang) + q[:, 2] * math.cos(ang)], 1) + base
    key('earOut' + side_name, (rot - co) * ew[:, None])

# ---------------------------------------------------------------- mouth interior: dark cavity behind the slit
# ---------------------------------------------------------------- the mouth: smooth shapes glued to the skin
# The skin is never cut. The opening is its own mesh lying just in front of the muzzle, defined by exact
# curves (so its edge is smooth at every opening); at rest it is a thin clean lip line. Every other face
# shape moves it with the skin point under it, so it stays glued through smiles, puckers and jaw sideways.
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree
polys = [tuple(pl.vertices) for pl in me.polygons]
def projector(verts):
    bvh = BVHTree.FromPolygons([Vector(v) for v in verts], polys)
    def y_at(x, z, fallback):
        hit = bvh.ray_cast(Vector((x, -3.0, z)), Vector((0, 1, 0)))
        return hit[0].y if hit[0] is not None else fallback
    return y_at
y_rest, y_open = projector(co), projector(co + skin_disp['jawOpen'])
kd = KDTree(int(front.sum())); fidx = np.nonzero(front)[0]
for n, i in enumerate(fidx): kd.insert(Vector(co[i]), n)
kd.balance()
def nearest_skin(p): return fidx[kd.find(Vector(p))[1]]
def skin_dz(x, z, name):
    i = nearest_skin((x, y_rest(x, z, surf_y), z)); return skin_disp[name][i]

HW = mouth_hw * 0.78                                             # Apple's opening is rounder than the lip line is long
U, V = 72, 12
xs = HW * np.sin(np.pi / 2 * np.linspace(-1, 1, U + 1))      # denser toward the corners, where the curve turns
def c_(x): return 0.006 * (x / HW) ** 2                        # the lip line turns up a touch at the corners
def g_(x): return np.sqrt(np.clip(1 - (x / HW) ** 2, 0, 1))
drop = np.array([skin_dz(x, mouth_z - 0.03, 'jawOpen')[2] for x in xs])      # where the lower lip goes
lift = np.array([skin_dz(x, mouth_z + 0.008, 'jawOpen')[2] for x in xs])     # where the upper lip goes
def clean(a, sigma=5.0):
    # the skin samples are bumpy: make the lip curves symmetric and smooth
    a = (a + a[::-1]) / 2
    k = np.exp(-0.5 * (np.arange(-15, 16) / sigma) ** 2); k /= k.sum()
    return np.convolve(np.pad(a, 15, mode='edge'), k, mode='valid')
drop, lift = clean(drop), clean(lift)
def grid(top, bot, yoff, bulge=0.0, xscale=1.0):
    pts = []
    for j in range(V + 1):
        t = j / V
        for i, x in enumerate(xs):
            z = top[i] + (bot[i] - top[i]) * t
            pts.append((x * xscale, z, yoff, bulge * (1 - (2 * t - 1) ** 2) * g_(x)))
    return pts
def build(name, rest_tb, open_tb, color, rough, yoff, bulge=0.0, collapse=False, xscale=1.0):
    rest_pts = grid(*rest_tb, yoff, bulge, xscale); open_pts = grid(*open_tb, yoff, bulge, xscale)
    def lay(pts, y_at):
        # depth from a smooth quadratic fitted to the skin under the shape, never behind the skin itself
        P = np.array([(x, z) for x, z, _, _ in pts]); hit = np.array([y_at(x, z, surf_y) for x, z in P])
        A = np.stack([np.ones(len(P)), P[:, 0], P[:, 1], P[:, 0] ** 2, P[:, 0] * P[:, 1], P[:, 1] ** 2], 1)
        coef = np.linalg.lstsq(A, hit, rcond=None)[0]; fit = A @ coef
        y = np.minimum(fit, hit - 0.0015)
        return np.array([(x, y[i] - yo - b, z) for i, (x, z, yo, b) in enumerate(pts)])
    rv = lay(rest_pts, y_rest)
    if collapse: rv[:] = rv.mean(0)                             # hidden at rest: a single point
    ov = lay(open_pts, y_open)
    faces = [(j * (U + 1) + i, j * (U + 1) + i + 1, (j + 1) * (U + 1) + i + 1, (j + 1) * (U + 1) + i) for j in range(V) for i in range(U)]
    m = bpy.data.meshes.new(name); m.from_pydata([tuple(v) for v in rv], [], faces); m.update()
    ob = bpy.data.objects.new(name, m); bpy.context.scene.collection.objects.link(ob)
    mt = bpy.data.materials.new(name); mt.use_nodes = True
    bs = mt.node_tree.nodes['Principled BSDF']; bs.inputs['Base Color'].default_value = color; bs.inputs['Roughness'].default_value = rough
    m.materials.append(mt)
    for pl in m.polygons: pl.use_smooth = True
    ob.shape_key_add(name='Basis', from_mix=False)
    # glue: every skin shape moves each point like the skin point under it
    near = [nearest_skin(v) for v in (ov if collapse else rv)]
    for kname, disp in skin_disp.items():
        k = ob.shape_key_add(name=kname, from_mix=False)
        tgt = ov if kname == 'jawOpen' else rv + disp[near]
        k.data.foreach_set('co', tgt.reshape(-1))
    return ob
rest_mouth = (mouth_z + c_(xs) + 0.0042 * g_(xs), mouth_z + c_(xs) - 0.0042 * g_(xs))
open_mouth = (mouth_z + c_(xs) + lift, mouth_z + c_(xs) + drop * g_(xs) ** 0.35)
build('MouthInterior', rest_mouth, open_mouth, (0.12, 0.018, 0.028, 1), 0.75, 0.0025)
# tongue: a rounded hump resting on the floor of the mouth, about half the opening tall, 60% as wide
TW = 0.62
xt = xs * TW
bot_full = np.interp(xt, xs, open_mouth[1]); top_full = np.interp(xt, xs, open_mouth[0])
h0 = float((open_mouth[0] - open_mouth[1])[U // 2]); zb0 = float(open_mouth[1][U // 2])
bot_t = bot_full + 0.1 * (top_full - bot_full)
dome = zb0 + 0.1 * h0 + 0.5 * h0 * np.sqrt(np.clip(1 - (xt / (HW * TW)) ** 2, 0, 1)) ** 0.7
top_t = np.maximum(np.minimum(dome, top_full - 0.18 * (top_full - bot_full)), bot_t)
build('Tongue', rest_mouth, (top_t, bot_t), (0.87, 0.38, 0.44, 1), 0.4, 0.0045, bulge=0.005, collapse=True, xscale=TW)
print('mouth: opening', round(h0, 3), 'wide', round(2 * HW, 3))

print('shape keys:', [k.name for k in me.shape_keys.key_blocks])
img = base_image(head)
if img is not None and max(img.size) > 2048: img.scale(2048, 2048)
bpy.ops.export_scene.gltf(filepath=out, export_format='GLB', export_morph=True, export_morph_normal=False, export_apply=False, export_yup=True,
                          export_image_format='JPEG', export_jpeg_quality=88)
print('exported', out)

# ---------------------------------------------------------------- previews
if preview:
    import os
    os.makedirs(preview, exist_ok=True)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'; sc.cycles.samples = 16; sc.cycles.device = 'CPU'
    sc.render.resolution_x = sc.render.resolution_y = 512
    sc.render.film_transparent = False
    world = bpy.data.worlds.new('w'); sc.world = world; world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.9, 0.9, 0.9, 1); world.node_tree.nodes['Background'].inputs['Strength'].default_value = 1.0
    cam_data = bpy.data.cameras.new('cam'); cam = bpy.data.objects.new('cam', cam_data); sc.collection.objects.link(cam); sc.camera = cam
    cam_data.lens = 85
    light = bpy.data.lights.new('key', 'SUN'); light.energy = 3; lo = bpy.data.objects.new('key', light); sc.collection.objects.link(lo)
    lo.rotation_euler = (math.radians(50), 0, math.radians(20))
    def shoot(name, keys, yaw=0):
        for ob in bpy.context.scene.objects:
            if ob.type != 'MESH' or not ob.data.shape_keys: continue
            for kb in ob.data.shape_keys.key_blocks[1:]: kb.value = keys.get(kb.name, 0)
        r = 4.2
        cam.location = (r * math.sin(math.radians(yaw)), -r * math.cos(math.radians(yaw)), 0.05)
        cam.rotation_euler = (math.radians(90), 0, math.radians(yaw))
        sc.render.filepath = os.path.join(preview, name + '.png'); bpy.ops.render.render(write_still=True)
    shoot('rest', {}); shoot('jaw', {'jawOpen': 1}); shoot('jawhalf', {'jawOpen': 0.5}); shoot('jawq', {'jawOpen': 0.8}, yaw=30); shoot('blink', {'eyeBlinkLeft': 1, 'eyeBlinkRight': 1})
    shoot('lookleft', {'eyeLookOutLeft': 1, 'eyeLookInRight': 1}); shoot('smile', {'mouthSmileLeft': 1, 'mouthSmileRight': 1, 'jawOpen': 0.3})
    shoot('turn', {}, yaw=35); shoot('ears', {'earBackLeft': 1, 'earBackRight': 1}, yaw=60)
    # mouth close-ups
    cam_data.lens = 260
    for j in (0.0, 0.2, 0.5, 1.0):
        for ob in bpy.context.scene.objects:
            if ob.type == 'MESH' and ob.data.shape_keys:
                for kb in ob.data.shape_keys.key_blocks[1:]: kb.value = j if kb.name == 'jawOpen' else 0
        cam.location = (0, -4.2, mouth_z - 0.02); cam.rotation_euler = (math.radians(90), 0, 0)
        sc.render.filepath = os.path.join(preview, f'm_{j}.png'); bpy.ops.render.render(write_still=True)
    print('previews in', preview)
