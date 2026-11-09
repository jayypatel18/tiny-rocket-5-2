"""The gouache stroke painter. Turns a composed reference into a painting.

API (everything is float32; pictures H x W x 3, layers H x W x 4 premultiplied):

  paint(ref, alpha=None, direction=None, regions=None, seed=0, passes=None,
        scale=1.0, size=None, spill=0.0, grain=0.05, under=0.12)
      ref        H x W x 3 composed reference: layout, values and light already right.
      alpha      optional H x W coverage. With it the result is a premultiplied RGBA
                 layer, otherwise an RGB picture.
      direction  optional H x W angle field in radians (0 = along +x, pi/2 = down the
                 screen). Strokes drag along it. None: strokes follow the contours of
                 the reference (flow_from).
      regions    optional H x W int32 labels. A stroke never paints outside the
                 region its centre lies in (crisp edges where asked).
      passes     list of dicts, big brushes first (see PASSES, LOOSE, FINE). Keys:
                 w, l (brush width and stroke length in px), dens (coverage), op
                 (opacity), keep (colour difference a stroke may not cross: small
                 keeps edges crisp), detail (0 = everywhere, 1 = only where the
                 reference has detail), jh, jv (hue and value jitter), dry (dry
                 brush breakup toward the stroke end), pool (darker stroke rim).
      scale      multiplies every brush size (use the layer resolution factor).
      size       optional H x W map that scales brushes locally (perspective).
      spill      0: the output alpha is exactly `alpha` (crisp silhouette).
                 > 0: strokes may run up to about `spill` px over the edge, so the
                 silhouette is brushy (foliage, grass crests, clouds, mist).
      grain      paper grain strength. under: softness of the underpainting.

  strokes(layer, xs, ys, ang, w, l, cols, op=1.0, bend=0.0, seed=0, dry=0.4,
          pool=0.0, taper=0.0, streak=0.10, field=None, origin=(0, 0))
      Explicit strokes on a premultiplied RGBA layer, in place: blades, leaves,
      flowers, cloud wisps, accents. Each stroke starts at (xs, ys) and runs along
      `ang` for length `l`, bending by `bend` radians, width `w`, thinning by
      `taper` (0..1) toward its end. Every argument may be a scalar or an array.
      cols is (3,) or (n, 3). Strokes are painted in array order.

  flow_from(ref, sigma=6)   angle field along the contours of a picture
  paper(h, w, seed=0)       paper tooth texture 0..1
  bleed(rgb, alpha)         spread colour outward from a shape (for painting layers)
  smoothstep(a, b, x)       numpy smoothstep

Deterministic from `seed`. A 2880 x 1620 layer paints in well under 40 s.
"""
import math

import numpy as np
from numba import njit
from scipy import ndimage

from tr2 import core

NB = 40  # bristles across a brush

PASSES = [
    dict(w=64, l=190, dens=1.3, op=0.92, keep=0.55, detail=0.0, jh=0.030, jv=0.07, dry=0.7, pool=0.05),
    dict(w=30, l=95, dens=1.5, op=0.88, keep=0.34, detail=0.0, jh=0.030, jv=0.07, dry=0.8, pool=0.06),
    dict(w=14, l=46, dens=1.5, op=0.88, keep=0.20, detail=0.55, jh=0.025, jv=0.06, dry=0.8, pool=0.07),
    dict(w=6.5, l=22, dens=1.3, op=0.90, keep=0.12, detail=0.85, jh=0.020, jv=0.05, dry=0.6, pool=0.05),
]
# Soft things (clouds, mist, far hills): bigger, wetter, fewer small strokes.
LOOSE = [
    dict(w=90, l=280, dens=1.3, op=0.90, keep=0.60, detail=0.0, jh=0.020, jv=0.04, dry=0.5, pool=0.03),
    dict(w=40, l=130, dens=1.4, op=0.85, keep=0.40, detail=0.0, jh=0.020, jv=0.04, dry=0.7, pool=0.04),
    dict(w=18, l=60, dens=1.2, op=0.85, keep=0.22, detail=0.7, jh=0.015, jv=0.04, dry=0.7, pool=0.05),
]
# Small crisp subjects (houses, stones): no big brushes.
FINE = [
    dict(w=22, l=60, dens=1.5, op=0.90, keep=0.30, detail=0.0, jh=0.025, jv=0.06, dry=0.7, pool=0.06),
    dict(w=10, l=30, dens=1.6, op=0.90, keep=0.16, detail=0.4, jh=0.025, jv=0.05, dry=0.7, pool=0.06),
    dict(w=4.5, l=14, dens=1.5, op=0.92, keep=0.10, detail=0.8, jh=0.020, jv=0.04, dry=0.5, pool=0.05),
]


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, np.float32) - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def paper(h, w, seed=0):
    """Paper tooth: fine fibre noise plus a slow mottling, 0..1."""
    fine = core.fbm(h, w, 5.0, 3, seed=seed * 7 + 11)
    mid = core.fbm(h, w, 34.0, 3, seed=seed * 7 + 12, aspect=1.6)
    p = 0.72 * fine + 0.28 * mid
    p = (p - p.mean()) / (p.std() + 1e-6)
    return np.clip(0.5 + 0.22 * p, 0, 1).astype(np.float32)


def flow_from(ref, sigma=6.0):
    """Angle field that runs ALONG the contours of the picture (structure tensor)."""
    lum = core.gblur(np.asarray(ref, np.float32).mean(axis=2), max(1.0, sigma * 0.5))
    gx = ndimage.sobel(lum, axis=1, mode="nearest")
    gy = ndimage.sobel(lum, axis=0, mode="nearest")
    jxx = core.gblur(gx * gx, sigma * 2)
    jyy = core.gblur(gy * gy, sigma * 2)
    jxy = core.gblur(gx * gy, sigma * 2)
    ang = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    return ang.astype(np.float32)


def bleed(rgb, alpha, thresh=0.5):
    """Extend the colours of a shape outward so strokes at its edge sample paint,
    not the empty background."""
    inside = alpha >= thresh
    if inside.all() or not inside.any():
        return np.ascontiguousarray(rgb, np.float32)
    idx = ndimage.distance_transform_edt(~inside, return_distances=False, return_indices=True)
    return np.ascontiguousarray(rgb[idx[0], idx[1]], np.float32)


@njit(cache=True, fastmath=True)
def _kernel(canvas, acan, ref, field, pap, regions,
            sx, sy, sang, sbend, sw, sl, sop, scol, sseed,
            mode_ref, use_field, use_acan, use_regions, premul,
            jit_h, jit_v, dry, pool, keep, taper, streak, paper_k):
    H = canvas.shape[0]
    W = canvas.shape[1]
    K = 4
    px = np.empty(K + 1)
    py = np.empty(K + 1)
    prof = np.empty(NB)
    col = np.empty(3)
    base = np.empty(3)
    for n in range(sx.shape[0]):
        x = sx[n]
        y = sy[n]
        hw = 0.5 * sw[n]
        L = sl[n]
        op = sop[n]
        s = sseed[n]
        # bristle profile: random, lightly smoothed so streaks have some width
        prev = 0.5
        for i in range(NB):
            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            v = ((s >> 8) & 0xFFFF) / 65535.0
            prof[i] = 0.55 * v + 0.45 * prev
            prev = v
        step = L / K
        ix = min(max(int(x), 0), W - 1)
        iy = min(max(int(y), 0), H - 1)
        if mode_ref:
            # centred stroke: march both ways from the centre
            if use_field:
                a0 = field[iy, ix] + sang[n]
            else:
                a0 = sang[n]
            px[2] = x
            py[2] = y
            for sgn in (1, -1):
                cx = x
                cy = y
                a = a0 if sgn == 1 else a0 + math.pi
                for i in range(2):
                    if use_field and i > 0:
                        jx = min(max(int(cx), 0), W - 1)
                        jy = min(max(int(cy), 0), H - 1)
                        af = field[jy, jx]
                        if math.cos(af - a) < 0:
                            af += math.pi
                        a = af
                    else:
                        a = a + sgn * sbend[n] * 0.25
                    cx += math.cos(a) * step
                    cy += math.sin(a) * step
                    px[2 + sgn * (i + 1)] = cx
                    py[2 + sgn * (i + 1)] = cy
            for c in range(3):
                base[c] = ref[iy, ix, c]
        else:
            a = sang[n]
            cx = x
            cy = y
            px[0] = x
            py[0] = y
            for i in range(K):
                if use_field:
                    jx = min(max(int(cx), 0), W - 1)
                    jy = min(max(int(cy), 0), H - 1)
                    af = field[jy, jx]
                    if math.cos(af - a) < 0:
                        af += math.pi
                    a = af
                else:
                    a = sang[n] + sbend[n] * ((i + 0.5) / K)
                cx += math.cos(a) * step
                cy += math.sin(a) * step
                px[i + 1] = cx
                py[i + 1] = cy
            for c in range(3):
                base[c] = scol[n, c]
        # colour with jitter
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        r1 = ((s >> 8) & 0xFFFF) / 32767.5 - 1.0
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        r2 = ((s >> 8) & 0xFFFF) / 32767.5 - 1.0
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        r3 = ((s >> 8) & 0xFFFF) / 32767.5 - 1.0
        vv = 1.0 + jit_v * r1
        col[0] = base[0] * vv + jit_h * (r2 - 0.4 * r3)
        col[1] = base[1] * vv + jit_h * (0.8 * r3)
        col[2] = base[2] * vv + jit_h * (-r2 - 0.4 * r3)
        for c in range(3):
            if col[c] < 0.0:
                col[c] = 0.0
            elif col[c] > 1.0:
                col[c] = 1.0
        reg0 = regions[iy, ix] if use_regions else 0
        # bounding box
        xmin = px[0]
        xmax = px[0]
        ymin = py[0]
        ymax = py[0]
        for i in range(1, K + 1):
            if px[i] < xmin:
                xmin = px[i]
            if px[i] > xmax:
                xmax = px[i]
            if py[i] < ymin:
                ymin = py[i]
            if py[i] > ymax:
                ymax = py[i]
        i0 = max(int(xmin - hw - 1), 0)
        i1 = min(int(xmax + hw + 2), W)
        j0 = max(int(ymin - hw - 1), 0)
        j1 = min(int(ymax + hw + 2), H)
        soft = max(1.0, 0.22 * hw)
        for j in range(j0, j1):
            qy = j + 0.5
            for i in range(i0, i1):
                qx = i + 0.5
                best = 1e18
                bu = 0.0
                side = 1.0
                for k in range(K):
                    vx = px[k + 1] - px[k]
                    vy = py[k + 1] - py[k]
                    wx = qx - px[k]
                    wy = qy - py[k]
                    t = (wx * vx + wy * vy) / (vx * vx + vy * vy + 1e-9)
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                    dx = wx - t * vx
                    dy = wy - t * vy
                    d2 = dx * dx + dy * dy
                    if d2 < best:
                        best = d2
                        bu = (k + t) / K
                        side = 1.0 if (vx * dy - vy * dx) >= 0 else -1.0
                d = math.sqrt(best)
                hwu = hw * (1.0 - taper * bu)
                if hwu < 0.35:
                    hwu = 0.35
                if d >= hwu:
                    continue
                if use_regions and regions[j, i] != reg0:
                    continue
                v = side * d / hwu
                e = (hwu - d) / (soft if soft < hwu else hwu)
                if e > 1.0:
                    e = 1.0
                b = prof[int((v * 0.5 + 0.5) * (NB - 1) + 0.5)]
                load = 1.2 - dry * 1.5 * bu * math.sqrt(bu)
                cv = (load - 0.8 * b - paper_k * (pap[j, i] - 0.5)) * 3.0
                if cv <= 0.0:
                    continue
                if cv > 1.0:
                    cv = 1.0
                cov = e * cv
                if mode_ref and keep > 0.0:
                    dc = 0.0
                    for c in range(3):
                        q = abs(ref[j, i, c] - base[c])
                        if q > dc:
                            dc = q
                    tk = (dc - 0.5 * keep) / (0.5 * keep)
                    if tk >= 1.0:
                        continue
                    if tk > 0.0:
                        cov *= 1.0 - tk * tk * (3 - 2 * tk)
                a = cov * op
                if a < 0.004:
                    continue
                av = abs(v)
                tp = (av - 0.5) / 0.45
                if tp < 0.0:
                    tp = 0.0
                elif tp > 1.0:
                    tp = 1.0
                f = (1.0 - pool * tp * tp) * (1.0 + streak * (b - 0.5))
                if premul:
                    for c in range(3):
                        canvas[j, i, c] = canvas[j, i, c] * (1 - a) + min(col[c] * f, 1.0) * a
                    canvas[j, i, 3] = canvas[j, i, 3] * (1 - a) + a
                else:
                    for c in range(3):
                        canvas[j, i, c] = canvas[j, i, c] * (1 - a) + min(col[c] * f, 1.0) * a
                    if use_acan:
                        acan[j, i] = acan[j, i] * (1 - a) + a


def _f64(v, n):
    a = np.asarray(v, np.float64)
    if a.ndim == 0:
        a = np.full(n, float(a))
    return np.ascontiguousarray(a)


def paint(ref, alpha=None, direction=None, regions=None, seed=0, passes=None,
          scale=1.0, size=None, spill=0.0, grain=0.05, under=0.12):
    ref = np.ascontiguousarray(np.clip(ref, 0, 1), np.float32)
    h, w = ref.shape[:2]
    passes = PASSES if passes is None else passes
    if alpha is not None:
        alpha = np.ascontiguousarray(alpha, np.float32)
        ref = bleed(ref, alpha)
    field = flow_from(ref) if direction is None else np.ascontiguousarray(direction, np.float32)
    pap = paper(h, w, seed)
    w0 = passes[0]["w"] * scale
    canvas = np.ascontiguousarray(core.gblur(ref, max(0.5, under * w0)), np.float32)
    acan = np.zeros((h, w), np.float32)
    use_acan = alpha is not None and spill > 0
    use_regions = regions is not None
    reg = np.ascontiguousarray(regions, np.int32) if use_regions else np.zeros((1, 1), np.int32)
    lum = ref.mean(axis=2)
    g = np.hypot(ndimage.sobel(lum, axis=1, mode="nearest"), ndimage.sobel(lum, axis=0, mode="nearest"))
    g = core.gblur(g, 3.0 * scale)
    det = np.clip(g / (np.percentile(g, 97) + 1e-6), 0, 1) ** 0.6
    dummy = np.zeros((1, 3), np.float64)
    for k, p in enumerate(passes):
        r = core.rng("paint", seed, k)
        bw = p["w"] * scale
        bl = p["l"] * scale
        n = int(p["dens"] * h * w / (bw * bl * 0.8)) + 1
        xs = r.random(n) * w
        ys = r.random(n) * h
        xi = np.minimum(xs.astype(np.int64), w - 1)
        yi = np.minimum(ys.astype(np.int64), h - 1)
        ok = np.ones(n, bool)
        if alpha is not None:
            ok &= alpha[yi, xi] > 0.35
        if p.get("detail", 0) > 0:
            d0 = p["detail"]
            ok &= r.random(n) < (1 - d0) + d0 * det[yi, xi]
        xs, ys, xi, yi = xs[ok], ys[ok], xi[ok], yi[ok]
        m = len(xs)
        if m == 0:
            continue
        sc = size[yi, xi].astype(np.float64) if size is not None else 1.0
        sw = bw * sc * r.uniform(0.7, 1.3, m)
        sl = bl * sc * r.uniform(0.6, 1.4, m)
        sang = r.normal(0, 0.22, m)
        sbend = r.normal(0, 0.35, m)
        sop = np.full(m, p["op"], np.float64) * r.uniform(0.85, 1.0, m)
        sseed = r.integers(1, 2 ** 30, m).astype(np.int64)
        _kernel(canvas, acan, ref, field, pap, reg,
                np.ascontiguousarray(xs), np.ascontiguousarray(ys), sang, sbend, sw, sl, sop, dummy, sseed,
                True, True, use_acan, use_regions, False,
                p.get("jh", 0.03), p.get("jv", 0.06), p.get("dry", 0.7), p.get("pool", 0.05),
                p.get("keep", 0.3), 0.0, 0.10, 0.9)
    if grain > 0:
        mott = core.fbm(h, w, 260.0 * scale, 3, seed=seed + 991)
        canvas *= (1.0 + grain * 2.0 * (pap - 0.5) + 0.03 * (mott - 0.5))[..., None]
    np.clip(canvas, 0, 1, out=canvas)
    if alpha is None:
        return canvas
    if spill <= 0:
        a = alpha
    else:
        soft = core.gblur(alpha, spill)
        lim = smoothstep(0.06, 0.5, soft)
        inner = smoothstep(0.80, 0.98, core.gblur(alpha, spill * 0.5))
        a = np.maximum(acan * lim, inner)
    return core.premul(canvas, np.ascontiguousarray(a, np.float32))


def strokes(layer, xs, ys, ang, w, l, cols, op=1.0, bend=0.0, seed=0, dry=0.4,
            pool=0.0, taper=0.0, streak=0.10, field=None, origin=(0, 0), paper_k=0.5):
    """Explicit strokes on a premultiplied RGBA layer, in place (see module doc)."""
    xs = np.atleast_1d(np.asarray(xs, np.float64)) - origin[0]
    ys = np.atleast_1d(np.asarray(ys, np.float64)) - origin[1]
    n = len(xs)
    if n == 0:
        return layer
    cols = np.asarray(cols, np.float64)
    if cols.ndim == 1:
        cols = np.repeat(cols[None], n, 0)
    r = core.rng("strokes", seed, n)
    h, wd = layer.shape[:2]
    pap = paper(h, wd, seed) if paper_k > 0 else np.full((h, wd), 0.5, np.float32)
    use_field = field is not None
    fld = np.ascontiguousarray(field, np.float32) if use_field else np.zeros((1, 1), np.float32)
    _kernel(layer, np.zeros((1, 1), np.float32), np.zeros((1, 1, 3), np.float32), fld, pap,
            np.zeros((1, 1), np.int32),
            np.ascontiguousarray(xs), np.ascontiguousarray(ys), _f64(ang, n), _f64(bend, n),
            _f64(w, n), _f64(l, n), _f64(op, n), np.ascontiguousarray(cols),
            r.integers(1, 2 ** 30, n).astype(np.int64),
            False, use_field, False, False, True,
            0.0, 0.0, float(dry), float(pool), 0.0, float(taper), float(streak), float(paper_k))
    return layer
