"""Core of the film engine: buffers, noise, splines, the cel rasteriser, the
multiplane camera, blur, compositing and file output.

Conventions (every module follows these):
  * A picture is a float32 numpy array, H x W x 3, display referred, 0..1.
  * A layer is float32 H x W x 4, PREMULTIPLIED alpha.
  * Screen is W x H = 1920 x 1080, 24 fps. x grows right, y grows down.
  * Rendering is deterministic and random access: frame f always renders the
    same and can be rendered alone. Seed all randomness with rng(key...).
"""
import math
import zlib

import numpy as np
from numba import njit, prange
from PIL import Image, ImageChops, ImageDraw
from scipy import ndimage

W, H = 1920, 1080
FPS = 24


# ----------------------------------------------------------------- randomness
def seed_of(*key):
    return zlib.crc32(repr(key).encode()) & 0x7FFFFFFF


def rng(*key):
    """A numpy Generator seeded from any tuple of names and numbers."""
    return np.random.default_rng(seed_of(*key))


# ---------------------------------------------------------------------- noise
@njit(cache=True, fastmath=True)
def _h3(ix, iy, iz, seed):
    h = (ix * 374761393 + iy * 668265263 + iz * 1440662683 + seed * 974711) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / 16777215.0


@njit(cache=True, fastmath=True)
def _vn3(x, y, z, seed):
    x0 = math.floor(x)
    y0 = math.floor(y)
    z0 = math.floor(z)
    fx = x - x0
    fy = y - y0
    fz = z - z0
    ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    uz = fz * fz * fz * (fz * (fz * 6 - 15) + 10)
    ix = int(x0)
    iy = int(y0)
    iz = int(z0)
    a = _h3(ix, iy, iz, seed)
    b = _h3(ix + 1, iy, iz, seed)
    c = _h3(ix, iy + 1, iz, seed)
    d = _h3(ix + 1, iy + 1, iz, seed)
    e = _h3(ix, iy, iz + 1, seed)
    f = _h3(ix + 1, iy, iz + 1, seed)
    g = _h3(ix, iy + 1, iz + 1, seed)
    h = _h3(ix + 1, iy + 1, iz + 1, seed)
    k0 = a + (b - a) * ux
    k1 = c + (d - c) * ux
    k2 = e + (f - e) * ux
    k3 = g + (h - g) * ux
    m0 = k0 + (k1 - k0) * uy
    m1 = k2 + (k3 - k2) * uy
    return m0 + (m1 - m0) * uz


@njit(cache=True, fastmath=True, parallel=True)
def _fbm_grid(out, x0, y0, dx, dy, z, octaves, lac, gain, seed):
    hh, ww = out.shape
    for j in prange(hh):
        for i in range(ww):
            x = x0 + i * dx
            y = y0 + j * dy
            amp = 1.0
            fr = 1.0
            s = 0.0
            norm = 0.0
            for o in range(octaves):
                s += amp * _vn3(x * fr, y * fr, z * fr + o * 17.3, seed + o * 101)
                norm += amp
                amp *= gain
                fr *= lac
            out[j, i] = s / norm


def fbm(h, w, scale, octaves=5, seed=0, ox=0.0, oy=0.0, z=0.0, lac=2.0, gain=0.5, aspect=1.0):
    """Fractal value noise on a grid, 0..1. `scale` is the size in pixels of the
    largest feature. (ox, oy) scroll it in pixels, z animates it (any float).
    aspect > 1 stretches features horizontally."""
    out = np.empty((h, w), np.float32)
    _fbm_grid(out, ox / (scale * aspect), oy / scale, 1.0 / (scale * aspect), 1.0 / scale,
              float(z), octaves, lac, gain, int(seed))
    return out


@njit(cache=True, fastmath=True)
def _fbm_pts(xs, ys, z, octaves, lac, gain, seed, out):
    for k in range(xs.shape[0]):
        amp = 1.0
        fr = 1.0
        s = 0.0
        norm = 0.0
        for o in range(octaves):
            s += amp * _vn3(xs[k] * fr, ys[k] * fr, z * fr + o * 17.3, seed + o * 101)
            norm += amp
            amp *= gain
            fr *= lac
        out[k] = s / norm


def fbm_at(xs, ys, scale, octaves=4, seed=0, z=0.0, lac=2.0, gain=0.5):
    """Fractal noise sampled at arbitrary points (same field as fbm with ox=oy=0)."""
    xs = np.ascontiguousarray(np.asarray(xs, np.float64).ravel() / scale)
    ys = np.ascontiguousarray(np.asarray(ys, np.float64).ravel() / scale)
    out = np.empty(xs.shape[0], np.float64)
    _fbm_pts(xs, ys, float(z), octaves, lac, gain, int(seed), out)
    return out


def noise1(t, seed=0, octaves=2):
    """Smooth 1D noise in -1..1 for wobble, sway and camera breathing."""
    return float(fbm_at([t], [0.37], 1.0, octaves, seed)[0] * 2.0 - 1.0)


# -------------------------------------------------------------------- timing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def smooth(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def ease_in(t, p=2.0):
    return clamp(t) ** p


def ease_out(t, p=2.0):
    return 1 - (1 - clamp(t)) ** p


def ease_io(t, p=2.0):
    t = clamp(t)
    return 0.5 * (2 * t) ** p if t < 0.5 else 1 - 0.5 * (2 - 2 * t) ** p


def overshoot(t, amt=1.70158):
    """Ease out that passes the target and settles back (follow through)."""
    t = clamp(t) - 1
    return t * t * ((amt + 1) * t + amt) + 1


def span(t, a, b):
    """0 before a, 1 after b, linear between. Feed into an ease."""
    if b == a:
        return 1.0 if t >= a else 0.0
    return clamp((t - a) / (b - a))


def keys(t, kf, ease=smooth):
    """Animate through key poses. kf = [(time, value), ...] sorted by time.
    Values may be floats or numpy arrays. Holds before the first and after the
    last key. `ease` shapes every segment (use a tuple entry (time, value, ease)
    to override per key: that ease is used arriving INTO that key)."""
    if t <= kf[0][0]:
        return kf[0][1]
    for i in range(1, len(kf)):
        if t <= kf[i][0]:
            t0, v0 = kf[i - 1][0], kf[i - 1][1]
            t1, v1 = kf[i][0], kf[i][1]
            e = kf[i][2] if len(kf[i]) > 2 else ease
            u = e((t - t0) / (t1 - t0)) if t1 > t0 else 1.0
            return v0 + (v1 - v0) * u
    return kf[-1][1]


def on_n(f, n=2, start=0):
    """The drawing number for frame f when animating on twos (n=2) or threes."""
    return start + ((f - start) // n) * n


# -------------------------------------------------------------------- curves
def catmull(pts, n=12, closed=True, tension=0.5):
    """Smooth curve through control points. Returns (len*n, 2) points."""
    p = np.asarray(pts, np.float64)
    m = len(p)
    if closed:
        idx = np.arange(m)
        p0, p1, p2, p3 = p[(idx - 1) % m], p[idx], p[(idx + 1) % m], p[(idx + 2) % m]
    else:
        pe = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
        p0, p1, p2, p3 = pe[:-3], pe[1:-2], pe[2:-1], pe[3:]
    t = np.linspace(0, 1, n, endpoint=False)[None, :, None]
    a = 2 * tension
    m1 = (p2 - p0)[:, None, :] * a * 0.5
    m2 = (p3 - p1)[:, None, :] * a * 0.5
    P1 = p1[:, None, :]
    P2 = p2[:, None, :]
    t2 = t * t
    t3 = t2 * t
    out = (2 * t3 - 3 * t2 + 1) * P1 + (t3 - 2 * t2 + t) * m1 + (-2 * t3 + 3 * t2) * P2 + (t3 - t2) * m2
    out = out.reshape(-1, 2)
    if not closed:
        out = np.vstack([out, p[-1]])
    return out


def resample(path, n):
    """n points evenly spaced by arc length along an open polyline."""
    p = np.asarray(path, np.float64)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    if d[-1] <= 0:
        return np.repeat(p[:1], n, 0)
    s = np.linspace(0, d[-1], n)
    return np.c_[np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])]


def ellipse(cx, cy, rx, ry, n=48, rot=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(a) * rx
    y = np.sin(a) * ry
    c, s = math.cos(rot), math.sin(rot)
    return np.c_[cx + x * c - y * s, cy + x * s + y * c]


def xform(pts, scale=1.0, rot=0.0, tx=0.0, ty=0.0, sx=1.0, sy=1.0):
    """Scale (with optional squash sx, sy), rotate (radians), then translate."""
    p = np.asarray(pts, np.float64) * (scale * sx, scale * sy)
    c, s = math.cos(rot), math.sin(rot)
    return np.c_[p[:, 0] * c - p[:, 1] * s + tx, p[:, 0] * s + p[:, 1] * c + ty]


def wobble(pts, amp, seed, freq=0.08):
    """Hand tremor: displace a path by smooth noise (amp in px). Same seed gives
    the same drawing, so change the seed per drawing (on twos) for line boil."""
    p = np.asarray(pts, np.float64)
    if amp <= 0:
        return p
    nx = fbm_at(p[:, 0] * freq, p[:, 1] * freq, 1.0, 2, seed) - 0.5
    ny = fbm_at(p[:, 0] * freq + 31.7, p[:, 1] * freq - 12.9, 1.0, 2, seed + 7) - 0.5
    return p + np.c_[nx, ny] * (2 * amp)


# ------------------------------------------------------------- cel rasteriser
def _rgba8(c, a=1.0):
    return (int(round(clamp(c[0]) * 255)), int(round(clamp(c[1]) * 255)),
            int(round(clamp(c[2]) * 255)), int(round(clamp(a) * 255)))


class Cel:
    """A transparent animation cel, drawn supersampled with hard edges, then
    reduced for clean anti aliasing. Coordinates are screen pixels (floats);
    the cel covers the box (x0, y0, x0+w, y0+h) of the screen.

        c = Cel()                      # full frame
        c.fill(path, (r, g, b))        # flat colour shape
        m = c.mask(path)               # reusable clip mask of a shape
        c.fill(shadow_path, col, clip=m)   # hard shadow inside that shape
        c.crescent(path, col, (dx, dy))    # shadow or rim crescent inside a shape
        c.stroke(path, 3.0, INK)       # tapered ink line with pressure
        layer = c.rgba()               # premultiplied float layer at 1x
    """

    def __init__(self, w=W, h=H, ss=3, x0=0, y0=0):
        self.w, self.h, self.ss, self.x0, self.y0 = int(w), int(h), int(ss), x0, y0
        self.im = Image.new("RGBA", (self.w * self.ss, self.h * self.ss), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def _xy(self, pts):
        p = (np.asarray(pts, np.float64) - (self.x0, self.y0)) * self.ss
        return [(float(a), float(b)) for a, b in p]

    def mask(self, pts, clip=None):
        m = Image.new("L", self.im.size, 0)
        if len(pts) >= 3:
            ImageDraw.Draw(m).polygon(self._xy(pts), fill=255)
        if clip is not None:
            m = ImageChops.multiply(m, clip)
        return m

    def paste(self, color, m, alpha=1.0):
        """Lay a flat colour through a mask (alpha < 1 for glazes)."""
        if alpha < 1.0:
            m = m.point(lambda v: int(v * alpha))
        self.im.paste(_rgba8(color), (0, 0), m)
        if alpha < 1.0 or True:
            pass

    def fill(self, pts, color, clip=None, alpha=1.0):
        if len(pts) < 3:
            return
        if clip is None and alpha >= 1.0:
            self.d.polygon(self._xy(pts), fill=_rgba8(color))
        else:
            m = self.mask(pts, clip)
            if alpha < 1.0:
                m = m.point(lambda v: int(v * alpha))
            lay = Image.new("RGBA", self.im.size, _rgba8(color))
            self.im = Image.composite(lay, self.im, m) if alpha >= 1.0 else Image.alpha_composite(
                self.im, Image.merge("RGBA", (*lay.split()[:3], m)))
            self.d = ImageDraw.Draw(self.im)

    def crescent(self, pts, color, offset, clip=None, alpha=1.0):
        """The part of shape `pts` NOT covered by the same shape moved by
        `offset` (dx, dy): a hard edged crescent. Move the shape toward the
        light to get the shadow side, away from it to get a rim light."""
        m = self.mask(pts, clip)
        moved = self.mask(np.asarray(pts, np.float64) + np.asarray(offset, np.float64))
        m = ImageChops.subtract(m, moved)
        if alpha < 1.0:
            m = m.point(lambda v: int(v * alpha))
        self.im = Image.alpha_composite(self.im, Image.merge(
            "RGBA", (*Image.new("RGB", self.im.size, _rgba8(color)[:3]).split(), m)))
        self.d = ImageDraw.Draw(self.im)

    def stroke(self, pts, width, color, taper=(0.18, 0.22), pressure=None, closed=False,
               seed=None, clip=None, min_w=0.12):
        """An ink line. `width` is the full width in screen px at full pressure.
        taper = fraction of the length that swells in and thins out. `pressure`
        is an optional array (any length) of 0..1 values along the line. A seed
        adds a little natural pressure variation."""
        p = np.asarray(pts, np.float64)
        if len(p) < 2:
            return
        if closed:
            p = np.vstack([p, p[:1]])
        n = max(8, int(np.hypot(*np.diff(p, axis=0).T).sum() / 1.5))
        p = resample(p, n)
        s = np.linspace(0, 1, n)
        wv = np.ones(n)
        if not closed:
            a, b = taper
            if a > 0:
                wv *= np.clip(s / a, 0, 1) ** 0.6
            if b > 0:
                wv *= np.clip((1 - s) / b, 0, 1) ** 0.6
        if pressure is not None:
            pr = np.asarray(pressure, np.float64)
            wv *= np.interp(s, np.linspace(0, 1, len(pr)), pr)
        if seed is not None:
            wv *= 0.82 + 0.36 * fbm_at(s * 6.0, np.zeros(n) + 0.5, 1.0, 2, seed)
        wv = np.maximum(wv, min_w) * width * 0.5
        t = np.gradient(p, axis=0)
        t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
        nrm = np.c_[-t[:, 1], t[:, 0]]
        poly = np.vstack([p + nrm * wv[:, None], (p - nrm * wv[:, None])[::-1]])
        self.fill(poly, color, clip=clip)

    def dot(self, x, y, r, color, clip=None):
        self.fill(ellipse(x, y, r, r, 20), color, clip=clip)

    def rgba(self):
        """Premultiplied float32 layer at 1x, the size of the cel box."""
        im = self.im.reduce(self.ss) if self.ss > 1 else self.im
        a = np.asarray(im, np.float32) / 255.0
        a[..., :3] *= a[..., 3:4]
        return a

    def alpha(self):
        im = self.im.getchannel("A")
        im = im.reduce(self.ss) if self.ss > 1 else im
        return np.asarray(im, np.float32) / 255.0


# ---------------------------------------------------------------- compositing
def over(dst, layer, x0=0, y0=0, opacity=1.0):
    """Composite a premultiplied RGBA layer onto an RGB picture, in place.
    (x0, y0) is where the layer's top left corner lands on dst."""
    x0 = int(round(x0))
    y0 = int(round(y0))
    h, w = layer.shape[:2]
    X0, Y0 = max(x0, 0), max(y0, 0)
    X1, Y1 = min(x0 + w, dst.shape[1]), min(y0 + h, dst.shape[0])
    if X1 <= X0 or Y1 <= Y0:
        return dst
    src = layer[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    a = src[..., 3:4] * opacity
    d = dst[Y0:Y1, X0:X1]
    d *= (1 - a)
    d += src[..., :3] * opacity
    return dst


def over_rgba(dst, layer):
    """Premultiplied layer over premultiplied layer (same size), in place."""
    a = layer[..., 3:4]
    dst *= (1 - a)
    dst += layer
    return dst


def add_light(dst, glow, color=(1, 1, 1), amount=1.0):
    """Additive light with a soft shoulder (screen), so it never clips hard.
    glow is H x W (0..1) or H x W x 3."""
    g = glow[..., None] * np.asarray(color, np.float32) if glow.ndim == 2 else glow
    g = np.clip(g * amount, 0, 1)
    dst[:] = 1 - (1 - np.clip(dst, 0, 1)) * (1 - g)
    return dst


def premul(rgb, a):
    out = np.empty(rgb.shape[:2] + (4,), np.float32)
    out[..., :3] = rgb * a[..., None]
    out[..., 3] = a
    return out


# ----------------------------------------------------------------------- blur
def gblur(img, sigma):
    """Gaussian blur of a picture, layer or single channel. Fast for big sigma."""
    if sigma <= 0.3:
        return img
    img = np.asarray(img, np.float32)
    k = int(sigma // 3)
    if k >= 2:
        h, w = img.shape[:2]
        ph, pw = (-h) % k, (-w) % k
        pad = ((0, ph), (0, pw)) + ((0, 0),) * (img.ndim - 2)
        im = np.pad(img, pad, mode="edge")
        sh = (im.shape[0] // k, k, im.shape[1] // k, k) + im.shape[2:]
        small = im.reshape(sh).mean(axis=(1, 3))
        sg = (sigma / k, sigma / k) + (0,) * (img.ndim - 2)
        small = ndimage.gaussian_filter(small, sg, mode="nearest")
        zf = (k, k) + (1,) * (img.ndim - 2)
        big = ndimage.zoom(small, zf, order=1, mode="nearest", grid_mode=True)
        return np.ascontiguousarray(big[:h, :w])
    sg = (sigma, sigma) + (0,) * (img.ndim - 2)
    return ndimage.gaussian_filter(img, sg, mode="nearest")


# --------------------------------------------------------------- warp, camera
@njit(cache=True, fastmath=True, parallel=True)
def _warp(src, out, s, tx, ty):
    sh, sw, ch = src.shape
    oh, ow = out.shape[0], out.shape[1]
    inv = 1.0 / s
    for j in prange(oh):
        v = (j + 0.5 - ty) * inv - 0.5
        j0 = int(math.floor(v))
        fv = v - j0
        for i in range(ow):
            u = (i + 0.5 - tx) * inv - 0.5
            i0 = int(math.floor(u))
            fu = u - i0
            for c in range(ch):
                acc = 0.0
                if 0 <= j0 < sh:
                    if 0 <= i0 < sw:
                        acc += src[j0, i0, c] * (1 - fu) * (1 - fv)
                    if 0 <= i0 + 1 < sw:
                        acc += src[j0, i0 + 1, c] * fu * (1 - fv)
                if 0 <= j0 + 1 < sh:
                    if 0 <= i0 < sw:
                        acc += src[j0 + 1, i0, c] * (1 - fu) * fv
                    if 0 <= i0 + 1 < sw:
                        acc += src[j0 + 1, i0 + 1, c] * fu * fv
                out[j, i, c] = acc


def warp(src, s, tx, ty, out_hw=(H, W)):
    """Resample: source pixel (u, v) lands at screen (u*s + tx, v*s + ty).
    Bilinear, transparent outside the source. When s < 0.6 the source is
    pre blurred a little so it does not alias."""
    src = np.ascontiguousarray(src, np.float32)
    if src.ndim == 2:
        src = src[..., None]
    if s < 0.6:
        src = gblur(src, 0.35 / s)
    out = np.zeros(out_hw + (src.shape[2],), np.float32)
    _warp(src, out, float(s), float(tx), float(ty))
    return out


class Camera:
    """The multiplane camera.

    A set is painted in SET coordinates: one big composition. (cx0, cy0) is the
    point of the set that sits at screen centre for the default view.
    cx, cy   where the camera looks now, in set coordinates (character plane)
    zoom     lens zoom: scales every layer the same
    dolly    physical push in: near layers grow more than far ones
    Each layer has a parallax factor p: 0 = infinitely far (sky, never moves),
    1 = the character plane, > 1 = foreground in front of the character.
    """

    def __init__(self, cx0=W / 2, cy0=H / 2, cx=None, cy=None, zoom=1.0, dolly=1.0, shake=(0.0, 0.0),
                 tilt=0.0, pan=0.0):
        self.cx0, self.cy0 = cx0, cy0
        self.cx = cx0 if cx is None else cx
        self.cy = cy0 if cy is None else cy
        self.zoom, self.dolly, self.shake = zoom, dolly, shake
        # tilt / pan: the camera ROTATES up (tilt > 0) or right (pan > 0). Unlike moving
        # cx, cy (a crane or dolly, which gives parallax), a rotation shifts EVERY layer,
        # the sky included, by the same number of screen px: tilt px down, pan px left.
        self.tilt, self.pan = tilt, pan

    def xf(self, p=1.0):
        """(s, tx, ty): a set point (X, Y) on a layer of parallax p lands at
        screen (X*s + tx, Y*s + ty)."""
        s = self.zoom * (self.dolly ** p)
        ax = self.cx0 + (self.cx - self.cx0) * p
        ay = self.cy0 + (self.cy - self.cy0) * p
        tx = W / 2 - ax * s + self.shake[0] * (0.3 + 0.7 * min(p, 1.5)) - self.pan
        ty = H / 2 - ay * s + self.shake[1] * (0.3 + 0.7 * min(p, 1.5)) + self.tilt
        return s, tx, ty

    def pt(self, X, Y, p=1.0):
        s, tx, ty = self.xf(p)
        return X * s + tx, Y * s + ty

    def place(self, layer, origin=(0, 0), p=1.0, blur=0.0):
        """Render a painted layer (its top left at set coords `origin`) through
        the camera. Returns a screen sized layer. blur = defocus sigma in px."""
        s, tx, ty = self.xf(p)
        out = warp(layer, s, tx + origin[0] * s, ty + origin[1] * s)
        return gblur(out, blur) if blur > 0.3 else out


# ----------------------------------------------------------------- finishing
def to8(img, seed=0):
    """Float picture to uint8 with a little noise dither so skies never band."""
    r = rng("dither", seed)
    n = (r.random(img.shape[:2], dtype=np.float32) - 0.5)[..., None] / 255.0
    return (np.clip(img + n, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def save(path, img, seed=0):
    """Save a picture (H x W x 3 float) or a layer (H x W x 4, shown on grey)."""
    if img.ndim == 3 and img.shape[2] == 4:
        bg = np.full(img.shape[:2] + (3,), 0.5, np.float32)
        img = over(bg, img)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, 2)
    Image.fromarray(to8(img, seed)).save(path)


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def sheet(imgs, cols, cell_w=480, path=None):
    """Contact sheet from a list of pictures."""
    th = [Image.fromarray(to8(i)) for i in imgs]
    ch = int(cell_w * th[0].height / th[0].width)
    th = [t.resize((cell_w, ch), Image.LANCZOS) for t in th]
    rows = (len(th) + cols - 1) // cols
    out = Image.new("RGB", (cols * cell_w, rows * ch), (20, 20, 20))
    for k, t in enumerate(th):
        out.paste(t, ((k % cols) * cell_w, (k // cols) * ch))
    if path:
        out.save(path, quality=90)
    return out
