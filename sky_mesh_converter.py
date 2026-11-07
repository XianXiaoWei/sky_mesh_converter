#!/usr/bin/env python3
"""Sky 0.32.2 mesh/GLB converter; evidence and limitations: 逆向说明.md.
单文件版：动画逻辑已内置，无需其他转换脚本。
Standard library only. Little endian .mesh versions 25..32.
"""
from __future__ import annotations
import argparse, base64, hashlib, json, math, os, re, struct, sys, tempfile, zlib
import shutil, unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

VERSIONS = tuple(range(25, 33))
FLAG_ORDER = (
    "StripGeo",
    "StripAnim",
    "UncompAnim",
    "Add",
    "CompOcc",
    "CompEdges",
    "CompAdj",
    "ZipPos",
    "ZipUvs",
    "StripUv13",
    "StripNorm",
    "ForceIdx32",
    "CopyFrameDelay",
)
FLAG_DESCRIPTIONS = {
    "StripGeo": "剥离几何（由结构决定）",
    "StripAnim": "剥离动画片段，仍可保留骨骼蒙皮（由结构决定）",
    "UncompAnim": "动画轨道使用float，取消LZ4与16位量化",
    "Add": "叠加动画（只支持原始数据保留）",
    "CompOcc": "遮挡辅助数据（仅几何和蒙皮未变时保留）",
    "CompEdges": "边辅助数据（只支持原始数据保留）",
    "CompAdj": "邻接辅助数据（只支持原始数据保留）",
    "ZipPos": "位置量化为10/10/10位，另存1字节顶点属性",
    "ZipUvs": "UV0/UV2的4个分量分别量化为8位",
    "StripUv13": "剥离UV1/UV3",
    "StripNorm": "剥离法线",
    "ForceIdx32": "强制32位索引（旧版本必须使用）",
    "CopyFrameDelay": "动画帧复制延迟（只支持原始数据保留）",
}
REFERENCE_ONLY = {"Add", "CompEdges", "CompAdj", "CopyFrameDelay"}
EXT = "SKY_mesh_source"
EXTRA_GEOMETRY = "sky_extra_geometry"
MAX_DATA = 1024 * 1024 * 1024


class FormatError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise FormatError(message)


def fixed_name(s, n=64):
    b = s.encode("utf-8")
    require(len(b) < n, f"名称超过{n-1}字节: {s}")
    return b + b"\0" * (n - len(b))


def name_of(b):
    return b.split(b"\0", 1)[0].decode("utf-8", "replace")


def digest(b):
    return hashlib.sha256(b).hexdigest()


def split_name(path):
    s = Path(path).stem
    flags = []
    while True:
        match = next((f for f in FLAG_ORDER if s.endswith("_" + f)), None)
        if match:
            s = s[: -len(match) - 1]
            flags.append(match)
            continue
        # Export spelling requested by user: base._Flag_Flag.glb
        if s.endswith(".") and flags:
            s = s[:-1]
        break
    return s, set(flags)


def suffix(flags):
    return "".join("_" + f for f in FLAG_ORDER if f in flags)


class Reader:
    def __init__(self, data, label="mesh"):
        self.data = data
        self.p = 0
        self.label = label

    def take(self, n):
        require(
            0 <= n <= len(self.data) - self.p,
            f"{self.label}: 0x{self.p:x}需要{n}字节，剩余{len(self.data)-self.p}",
        )
        b = self.data[self.p : self.p + n]
        self.p += n
        return b

    def unpack(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))

    def u32(self):
        return self.unpack("I")[0]

    def byte(self):
        return self.take(1)[0]


def lz4_decode(src, expected):
    require(0 <= expected <= MAX_DATA, "LZ4解压长度超限")
    try:
        import lz4.block

        return lz4.block.decompress(src, uncompressed_size=expected)
    except ImportError:
        pass
    out = bytearray()
    p = 0

    def length(v):
        nonlocal p
        if v == 15:
            while True:
                require(p < len(src), "截断的LZ4长度")
                x = src[p]
                p += 1
                v += x
                if x != 255:
                    break
        return v

    while p < len(src):
        token = src[p]
        p += 1
        n = length(token >> 4)
        require(p + n <= len(src) and len(out) + n <= expected, "无效LZ4 literal")
        out.extend(src[p : p + n])
        p += n
        if p == len(src):
            break
        require(p + 2 <= len(src), "截断的LZ4偏移")
        off = src[p] | src[p + 1] << 8
        p += 2
        require(0 < off <= len(out), "无效LZ4回溯偏移")
        n = length(token & 15) + 4
        require(len(out) + n <= expected, "LZ4输出溢出")
        # Repeating slice handles overlapping matches without per-byte overhead.
        block = bytes(out[-off:])
        out.extend((block * ((n + off - 1) // off))[:n])
    require(len(out) == expected, "LZ4解压长度不匹配")
    return bytes(out)


def lz4_encode(data):
    try:
        import lz4.block

        return lz4.block.compress(data, store_size=False)
    except ImportError:
        # A standards-compliant all-literal block. Valid in the game's LZ4 decoder.
        n = len(data)
        out = bytearray([min(n, 15) << 4])
        if n >= 15:
            n -= 15
            while n >= 255:
                out.append(255)
                n -= 255
            out.append(n)
        return bytes(out) + data


@dataclass
class Lod:
    distance: float
    bounds: tuple
    quant_lo: tuple
    quant_hi: tuple
    vertices: int
    indices_count: int
    idx32: int
    counts: tuple
    has_normals: int
    has_secondary: int
    third: int
    zipped: tuple
    groups: tuple
    streams: dict = field(default_factory=dict)
    raw: bytes = b""


@dataclass
class Mesh:
    version: int
    name: str
    animated: int
    extra: int
    descriptors: bytes
    lods: list
    skeleton: bytes
    extra_data: bytes
    compressed: int
    raw: bytes
    filename_flags: set = field(default_factory=set)
    warnings: list = field(default_factory=list)


def read_lod(r, version, animated):
    start = r.p
    distance = r.unpack("f")[0]
    require(not math.isnan(distance), "LOD距离是NaN")
    bounds = r.unpack("12f" if version >= 28 else "6f")
    if version < 28:
        bounds = bounds + bounds
    lo = r.unpack("8f") if version > 28 else (0.0,) * 8
    hi = r.unpack("8f") if version > 28 else (1.0,) * 8
    n, ni = r.unpack("2I")
    require(n <= 10000000 and ni <= 60000000 and ni % 3 == 0, "无效顶点/三角形数量")
    idx32 = r.u32() if version > 29 else 1
    require(idx32 in (0, 1), "索引格式不是0/1")
    counts = r.unpack("5I")
    require(max(counts, default=0) <= 60000000, "辅助数据数量超限")
    require(max(counts[:2]) <= n, "顶点映射组数超过顶点数")
    if version >= 29:
        hn, hs, t = r.unpack("3B")
        zp, zu, z13 = r.unpack("3I")
        groups = r.unpack("4I")
        require(hn in (0, 1) and hs in (0, 1) and t in (0, 1), "无效LOD布尔字段")
        require(
            all(x in (0, n * 4) for x in (zp, zu, z13)), "压缩属性长度必须为0或4×顶点数"
        )
    else:
        hn, hs, t = 1, 1, 0
        groups = r.unpack("4I")
        zp = zu = z13 = 0
    lod = Lod(
        distance, bounds, lo, hi, n, ni, idx32, counts, hn, hs, t, (zp, zu, z13), groups
    )
    s = lod.streams
    unit = 4 if idx32 else 2

    def take(key, sz):
        s[key] = r.take(sz)

    if not zp:
        take("positions", n * 16)
    if hn:
        take("normals", n * 4)
    if not zu:
        take("uvs", n * 16)
    if animated:
        take("weights", n * 8)
    take("indices", ni * unit)
    if hs:
        take("secondary_indices", ni * unit)
    if counts[0]:
        take("position_map", n * unit)
    if counts[1]:
        take("normal_map", n * unit)
    if counts[2]:
        take("strips", counts[2] * unit)
    if counts[3]:
        take("metrics", counts[3] * 4)
    if counts[4]:
        take("pairs", counts[4] * 2 * unit)
    if version > 28:
        take("areas", ni // 3 * 4)
    if zp:
        take("zip_positions", n * 4)
        take("alpha", n)
    if zu:
        take("zip_uvs", n * 4)
    if z13:
        take("zip_uv13", n * 4)
    lod.raw = r.data[start : r.p]
    indices = struct.unpack("<" + ("I" if idx32 else "H") * ni, s["indices"])
    require(not indices or max(indices) < n, "三角形索引超出顶点范围")
    for key, count in (("position_map", counts[0]), ("normal_map", counts[1])):
        if count:
            require(
                all(
                    x[0] < count
                    for x in struct.iter_unpack("<I" if idx32 else "<H", s[key])
                ),
                f"{key}索引超出映射组范围",
            )
    return lod


def decode_extra_geometry(data):
    r = Reader(data, "附加几何")
    n, ni = r.unpack("2I")
    require(n <= 10000000 and ni <= 60000000 and ni % 3 == 0, "附加几何数量无效")
    positions = list(struct.iter_unpack("<3f", r.take(n * 12)))
    indices = [x[0] for x in struct.iter_unpack("<I", r.take(ni * 4))]
    require(r.p == len(data), "附加几何尾部有未解释字节")
    require(
        all(math.isfinite(x) for p in positions for x in p), "附加几何位置含非有限数"
    )
    require(all(i < n for i in indices), "附加几何索引越界")
    return {"POSITION": positions, "indices": indices}


def encode_extra_geometry(geom):
    positions, indices = geom["POSITION"], geom["indices"]
    require(len(positions) <= 10000000 and len(indices) <= 60000000, "附加几何数量超限")
    require(
        all(
            len(p) == 3
            and all(math.isfinite(x) and abs(x) <= 3.4028234663852886e38 for x in p)
            for p in positions
        ),
        "附加几何位置必须是有限float32 XYZ",
    )
    require(len(indices) % 3 == 0, "附加几何三角索引数量不是3的倍数")
    require(
        all(isinstance(i, int) and 0 <= i < len(positions) for i in indices),
        "附加几何索引越界",
    )
    return (
        struct.pack("<2I", len(positions), len(indices))
        + b"".join(struct.pack("<3f", *p) for p in positions)
        + pack_indices(indices, 1)
    )


def read_mesh(data, filename="model.mesh"):
    r = Reader(data)
    v = r.u32()
    require(v in VERSIONS, f"不支持版本0x{v:x}；该so正常接受0x19..0x20")
    name = name_of(r.take(64))
    nl = r.u32()
    require(nl <= 64, "LOD数量超限")
    animated, extra = r.unpack("2B")
    require(animated in (0, 1) and extra in (0, 1), "无效mesh结构标记")
    nm = r.u32() if v > 31 else 0
    require(nm <= 100000, "头部附加描述数量超限")
    descriptors = r.take(nm * 112)
    comp = r.u32() if v >= 27 else 0
    require(comp in (0, 1), "压缩类型不是0/1")
    if comp:
        cs, us = r.unpack("2I")
        payload = lz4_decode(r.take(cs), us)
        pr = Reader(payload, "LOD payload")
    else:
        pr = r
    lods = [read_lod(pr, v, animated) for _ in range(nl)]
    ex = b""
    if extra:
        start = pr.p
        np, ni = pr.unpack("2I")
        require(np <= 10000000 and ni <= 60000000 and ni % 3 == 0, "附加几何数量无效")
        pr.take(np * 12 + ni * 4)
        ex = pr.data[start : pr.p]
        decode_extra_geometry(ex)
    if comp:
        require(pr.p == len(payload), f"LOD payload尾部有{len(payload)-pr.p}未解释字节")
    skel = r.take(len(data) - r.p)
    if animated:
        read_skeleton(skel)
    else:
        require(not skel, "静态mesh尾部有未解释数据")
    m = Mesh(
        v,
        name,
        animated,
        extra,
        descriptors,
        lods,
        skel,
        ex,
        comp,
        data,
        split_name(filename)[1],
    )
    if v < 31:
        m.warnings.append("该版本会触发游戏Old mesh version提示")
    actual = detect_flags(m)
    for f in ("ZipPos", "ZipUvs", "StripNorm", "ForceIdx32", "CompOcc"):
        if (f in m.filename_flags) != (f in actual):
            m.warnings.append(f"文件名{f}与数据特征不一致；采用数据特征")
    return m


def read_skeleton(data):
    r = Reader(data, "skeleton")
    v = r.u32()
    require(7 <= v <= 11, f"骨骼版本{v}不在so正常范围7..11")
    name = name_of(r.take(64))
    n, na, nf = r.unpack("3I")
    comp = r.byte()
    pool = r.u32() if v >= 10 else n * 64
    require(n <= 255 and na <= 100000 and nf <= 10000000, "骨骼/动画数量超限")
    bones = []
    for i in range(n):
        bn = name_of(r.take(64))
        mat = r.unpack("16f")
        parent = r.u32() - 1
        require(-1 <= parent < n and parent != i, "骨骼父索引无效")
        if na:
            require(all(math.isfinite(x) for x in mat), "骨骼矩阵不是有限数")
        bones.append({"name": bn, "matrix": transpose(mat), "parent": parent})
    # Verify the hierarchy independently of bone storage order.
    for i in range(n):
        seen = set()
        j = i
        while j >= 0:
            require(j not in seen, "骨骼父子关系成环")
            seen.add(j)
            j = bones[j]["parent"]
    return {
        "version": v,
        "name": name,
        "bones": bones,
        "frame_count": na,
        "fps": nf,
        "compression": comp,
        "pool": pool,
        "tail": data[r.p :],
    }


def detect_flags(m):
    out = set()
    ls = m.lods
    if not ls:
        out.add("StripGeo")
    if m.animated:
        s = read_skeleton(m.skeleton)
        if not s["frame_count"]:
            out.add("StripAnim")
        if s["frame_count"] and not s["compression"]:
            out.add("UncompAnim")
    if ls:
        if all(l.zipped[0] for l in ls):
            out.add("ZipPos")
        if all(l.zipped[1] for l in ls):
            out.add("ZipUvs")
        if all(not l.has_normals for l in ls):
            out.add("StripNorm")
        if all(not l.zipped[2] for l in ls) and any(l.zipped[1] for l in ls):
            out.add("StripUv13")
        if all(l.idx32 for l in ls):
            out.add("ForceIdx32")
        if any(l.counts[2] for l in ls):
            out.add("CompOcc")
    # These resource request switches are not uniquely encoded by mesh bytes.
    out |= m.filename_flags & {"Add", "CompEdges", "CompAdj", "CopyFrameDelay"}
    return out


def identity():
    return [1.0 if i % 5 == 0 else 0.0 for i in range(16)]


def transpose(m):
    return [m[j * 4 + i] for i in range(4) for j in range(4)]


def mm(a, b):
    return [
        sum(a[i * 4 + k] * b[k * 4 + j] for k in range(4))
        for i in range(4)
        for j in range(4)
    ]


def inverse(m):
    a = [
        list(m[i * 4 : i * 4 + 4]) + [float(i == j) for j in range(4)] for i in range(4)
    ]
    for i in range(4):
        k = max(range(i, 4), key=lambda k: abs(a[k][i]))
        require(abs(a[k][i]) > 1e-12, "奇异矩阵不能求逆")
        a[i], a[k] = a[k], a[i]
        q = a[i][i]
        a[i] = [x / q for x in a[i]]
        for j in range(4):
            if j != i:
                q = a[j][i]
                a[j] = [x - q * y for x, y in zip(a[j], a[i])]
    return [x for row in a for x in row[4:]]


def normal(v):
    n = math.sqrt(sum(x * x for x in v))
    return tuple(x / n for x in v) if n > 1e-15 else (0.0, 1.0, 0.0)


def transform(m, v, w=1.0):
    return tuple(
        sum(m[i * 4 + j] * v[j] for j in range(3)) + m[i * 4 + 3] * w for i in range(3)
    )


def decode_lod(l):
    s = l.streams
    n = l.vertices
    out = {"POSITION": [], "TEXCOORD_0": [], "COLOR_0": []}
    if l.zipped[0]:
        lo = l.bounds[6:9]
        hi = l.bounds[9:12]
        for i, (pk,) in enumerate(struct.iter_unpack("<I", s["zip_positions"])):
            out["POSITION"].append(
                tuple(
                    lo[j] + ((pk >> (20 - 10 * j)) & 1023) / 1023 * (hi[j] - lo[j])
                    for j in range(3)
                )
            )
            out["COLOR_0"].append((1.0, 1.0, 1.0, s["alpha"][i] / 255))
    else:
        for x, y, z, c in struct.iter_unpack("<3fI", s["positions"]):
            out["POSITION"].append((x, y, z))
            out["COLOR_0"].append(tuple((c >> (8 * j) & 255) / 255 for j in range(4)))
    if l.has_normals:
        out["NORMAL"] = [
            normal(tuple(x / 127 for x in row[:3]))
            for row in struct.iter_unpack("4b", s["normals"])
        ]
    if l.zipped[1]:
        for channel, dims in ((0, (0, 1)), (2, (4, 5))):
            out["TEXCOORD_" + str(channel)] = [
                tuple(
                    l.quant_lo[d]
                    + b[j + channel] / 255 * (l.quant_hi[d] - l.quant_lo[d])
                    for j, d in enumerate(dims)
                )
                for b in struct.iter_unpack("4B", s["zip_uvs"])
            ]
    else:
        uvs = [struct.unpack("<8e", s["uvs"][i * 16 : i * 16 + 16]) for i in range(n)]
        for j in range(4):
            out["TEXCOORD_" + str(j)] = [tuple(x[j * 2 : j * 2 + 2]) for x in uvs]
    if l.zipped[2]:
        for channel, dims in ((1, (2, 3)), (3, (6, 7))):
            out["TEXCOORD_" + str(channel)] = [
                tuple(
                    l.quant_lo[d]
                    + b[j + (channel - 1)] / 255 * (l.quant_hi[d] - l.quant_lo[d])
                    for j, d in enumerate(dims)
                )
                for b in struct.iter_unpack("4B", s["zip_uv13"])
            ]
    if "weights" in s:
        out["JOINTS_0"] = []
        out["WEIGHTS_0"] = []
        for row in struct.iter_unpack("8B", s["weights"]):
            # Palette slot 0 is the mesh transform; bone i occupies slot i+1.
            # Keep it separate from bone 0, and merge repeated influences for glTF.
            merged = {}
            for joint, weight in zip(row[:4], row[4:]):
                if weight:
                    merged[joint - 1] = merged.get(joint - 1, 0) + weight
            total = sum(merged.values())
            require(total > 0, "蒙皮顶点无有效权重")
            pairs = list(merged.items())
            out["JOINTS_0"].append(tuple(j for j, w in pairs) + (0,) * (4 - len(pairs)))
            out["WEIGHTS_0"].append(
                tuple(w / total for j, w in pairs) + (0.0,) * (4 - len(pairs))
            )
    out["indices"] = [
        x[0] for x in struct.iter_unpack("<I" if l.idx32 else "<H", s["indices"])
    ]
    for k, rows in out.items():
        if k != "indices":
            require(
                all(math.isfinite(x) for row in rows for x in row), f"{k}包含非有限数"
            )
    return out


def smooth_normals(pos, indices):
    out = [[0.0, 0.0, 0.0] for _ in pos]
    for i in range(0, len(indices), 3):
        a, b, c = indices[i : i + 3]
        u = [pos[b][j] - pos[a][j] for j in range(3)]
        v = [pos[c][j] - pos[a][j] for j in range(3)]
        cross = (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
        for k in (a, b, c):
            for j in range(3):
                out[k][j] += cross[j]
    return [normal(x) for x in out]


class GlbBuilder:
    def __init__(self):
        self.bin = bytearray()
        self.doc = {
            "asset": {"version": "2.0", "generator": "Sky mesh converter (so 365939)"},
            "buffers": [],
            "bufferViews": [],
            "accessors": [],
            "meshes": [],
            "nodes": [],
            "scenes": [{"nodes": []}],
            "scene": 0,
        }

    def view(self, data, target=None):
        self.bin.extend(b"\0" * (-len(self.bin) % 4))
        off = len(self.bin)
        self.bin.extend(data)
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            v["target"] = target
        self.doc["bufferViews"].append(v)
        return len(self.doc["bufferViews"]) - 1

    def accessor(self, rows, typ, component=5126, target=None, bounds=False):
        fmt = {5126: "f", 5123: "H", 5125: "I", 5121: "B"}[component]
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[typ]
        raw = b"".join(struct.pack("<" + fmt * width, *row) for row in rows)
        v = self.view(raw, target)
        a = {
            "bufferView": v,
            "componentType": component,
            "count": len(rows),
            "type": typ,
        }
        if bounds and rows:
            a.update(
                min=[min(row[i] for row in rows) for i in range(width)],
                max=[max(row[i] for row in rows) for i in range(width)],
            )
        self.doc["accessors"].append(a)
        return len(self.doc["accessors"]) - 1

    def data(self):
        for key in ("meshes", "nodes", "accessors", "skins", "animations"):
            if self.doc.get(key) == []:
                self.doc.pop(key, None)
        for scene in self.doc.get("scenes", []):
            if scene.get("nodes") == []:
                scene.pop("nodes")
        self.bin.extend(b"\0" * (-len(self.bin) % 4))
        self.doc["buffers"] = [{"byteLength": len(self.bin)}]
        jb = json.dumps(
            self.doc, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
        jb += b" " * (-len(jb) % 4)
        return (
            struct.pack("<III", 0x46546C67, 2, 28 + len(jb) + len(self.bin))
            + struct.pack("<II", len(jb), 0x4E4F534A)
            + jb
            + struct.pack("<II", len(self.bin), 0x004E4942)
            + self.bin
        )


def builder_view(builder):
    view = object.__new__(Glb)
    view.doc = builder.doc
    view.bin = bytes(builder.bin)
    return view


def export_glb(m):
    g = GlbBuilder()
    d = g.doc
    nl = len(m.lods)
    animation = _sky_animation

    bones = []
    game_animation = None
    if m.animated:
        bones = read_skeleton(m.skeleton)["bones"]
        game_animation = animation.read_animation(m.skeleton)
    identity_slot = any(
        joint == 0 and weight > 0
        for lod in m.lods
        for row in struct.iter_unpack("8B", lod.streams.get("weights", b""))
        for joint, weight in zip(row[:4], row[4:])
    )
    worlds = []
    for b in bones:
        try:
            worlds.append(inverse(b["matrix"]))
        except FormatError:
            require(
                not read_skeleton(m.skeleton)["frame_count"], "动画骨骼bind矩阵无效"
            )
            b["matrix"] = identity()
            worlds.append(identity())
    for i, l in enumerate(m.lods):
        geom = decode_lod(l)
        if "JOINTS_0" in geom:
            require(bones, "蒙皮没有骨架")
            geom["JOINTS_0"] = [
                tuple(len(bones) if j == -1 else j for j in row)
                for row in geom["JOINTS_0"]
            ]
            require(
                all(
                    0 <= j < len(bones) + int(identity_slot)
                    for row in geom["JOINTS_0"]
                    for j in row
                ),
                "蒙皮骨骼索引越界",
            )
        uv_channels = [
            int(k.rsplit("_", 1)[1]) for k in geom if k.startswith("TEXCOORD_")
        ]
        # glTF requires continuous semantic indices; source extension retains absence.
        for channel in range(max(uv_channels, default=-1) + 1):
            geom.setdefault("TEXCOORD_" + str(channel), [(0.0, 0.0)] * l.vertices)
        attrs = {}
        # A stripped normal remains absent, so it is not mistaken for source data.
        for key, rows in geom.items():
            if key == "indices":
                continue
            typ = "VEC" + str(len(rows[0])) if rows else "VEC3"
            if not rows:
                continue
            attrs[key] = g.accessor(
                rows, typ, 5123 if key == "JOINTS_0" else 5126, 34962, key == "POSITION"
            )
        ids = geom["indices"]
        ia = g.accessor(
            [(i,) for i in ids],
            "SCALAR",
            5125 if max(ids, default=0) > 65535 else 5123,
            34963,
        )
        d["meshes"].append(
            {
                "name": m.name + f"_LOD{i}",
                "primitives": [{"attributes": attrs, "indices": ia, "mode": 4}],
            }
        )
        node = {
            "name": m.name + f"_LOD{i}",
            "mesh": i,
            "extras": {
                "sky_lod": i,
                "sky_distance": l.distance if math.isfinite(l.distance) else "inf",
            },
        }
        if bones:
            node["skin"] = 0
        d["nodes"].append(node)
    for i, b in enumerate(bones):
        p = b["parent"]
        local = mm(inverse(worlds[p]), worlds[i]) if p >= 0 else worlds[i]
        node = {"name": b["name"], "matrix": transpose(local)}
        if game_animation and game_animation.clips:
            node = {"name": b["name"], **game_animation.bind[i]}
        children = [nl + j for j, c in enumerate(bones) if c["parent"] == i]
        if children:
            node["children"] = children
        d["nodes"].append(node)
    if nl:
        d["scenes"][0]["nodes"].append(0)
    if nl > 1:
        d["nodes"][0]["extensions"] = {"MSFT_lod": {"ids": list(range(1, nl))}}
        d["extensionsUsed"] = ["MSFT_lod"]
    if bones:
        skin_joints = list(range(nl, nl + len(bones)))
        bind_matrices = [transpose(b["matrix"]) for b in bones]
        roots = [nl + i for i, b in enumerate(bones) if b["parent"] < 0]
        if identity_slot:
            virtual = len(d["nodes"])
            d["nodes"].append(
                {"name": "SKY_mesh_transform", "extras": {"sky_identity_slot": True}}
            )
            skin_joints.append(virtual)
            bind_matrices.append(identity())
            roots.append(virtual)
        if len(roots) > 1:
            root = len(d["nodes"])
            d["nodes"].append({"name": "SKY_skeleton_root", "children": roots})
            roots = [root]
        ib = g.accessor(bind_matrices, "MAT4")
        d["skins"] = [{"joints": skin_joints, "inverseBindMatrices": ib}]
        d["scenes"][0]["nodes"] += roots
    if game_animation and game_animation.clips:
        animation.add_glb_animations(g, game_animation, nl)
    extra_node = None
    if m.extra:
        geom = decode_extra_geometry(m.extra_data)
        if geom["POSITION"] and geom["indices"]:
            pa = g.accessor(geom["POSITION"], "VEC3", 5126, 34962, True)
            ia = g.accessor([(i,) for i in geom["indices"]], "SCALAR", 5125, 34963)
            mesh_id = len(d["meshes"])
            d["meshes"].append(
                {
                    "name": m.name + "_附加几何",
                    "extras": {EXTRA_GEOMETRY: True},
                    "primitives": [
                        {"attributes": {"POSITION": pa}, "indices": ia, "mode": 4}
                    ],
                }
            )
            extra_node = len(d["nodes"])
            d["nodes"].append(
                {
                    "name": m.name + "_附加几何",
                    "mesh": mesh_id,
                    "extras": {EXTRA_GEOMETRY: True},
                }
            )
            d["scenes"].append({"name": "附加几何", "nodes": [extra_node]})
    # Retain original blocks for exact source recovery.
    source_view = g.view(m.raw)
    d.setdefault("extensionsUsed", []).append(EXT)
    d["extensions"] = {
        EXT: {
            "bufferView": source_view,
            "sha256": digest(m.raw),
            "version": m.version,
            "flags": sorted(detect_flags(m)),
            "filenameFlags": sorted(m.filename_flags),
            "coordinateSystem": "game-native",
            "animationClipsExported": bool(game_animation and game_animation.clips),
            "animationHash": animation.fingerprint(builder_view(g)),
            "extraGeometryExported": extra_node is not None,
        }
    }
    # Hash only the visible geometry, hierarchy and standard animation data.
    d["extensions"][EXT]["geometryHash"] = glb_fingerprint(d, bytes(g.bin))
    return g.data()


COMPONENTS = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}
WIDTHS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


class Glb:
    def __init__(self, data):
        require(len(data) >= 20, "GLB文件过短")
        magic, v, total = struct.unpack_from("<III", data)
        require(
            magic == 0x46546C67 and v == 2 and total == len(data), "无效GLB头/文件长度"
        )
        p = 12
        doc = None
        blob = None
        while p < len(data):
            require(p + 8 <= len(data), "GLB chunk头截断")
            n, t = struct.unpack_from("<II", data, p)
            p += 8
            require(n % 4 == 0 and p + n <= len(data), "GLB chunk长度无效")
            b = data[p : p + n]
            p += n
            if t == 0x4E4F534A:
                require(doc is None, "重复JSON chunk")
                doc = json.loads(b)
            elif t == 0x004E4942:
                require(blob is None, "重复BIN chunk")
                blob = b
        require(doc is not None, "没有GLB JSON chunk")
        self.doc = doc
        self.bin = blob or b""
        require(doc.get("asset", {}).get("version") == "2.0", "只接受glTF 2.0")
        required = set(doc.get("extensionsRequired", []))
        unsupported = required - {"KHR_mesh_quantization", EXT, "MSFT_lod"}
        require(
            not unsupported,
            f"不支持必要扩展: {sorted(unsupported)}；请先解压Draco/meshopt",
        )

    def view(self, i):
        v = self.doc["bufferViews"][i]
        require(v.get("buffer", 0) == 0, "GLB外部buffer不支持")
        off = v.get("byteOffset", 0)
        n = v["byteLength"]
        require(
            0 <= off <= len(self.bin) and 0 <= n <= len(self.bin) - off,
            "bufferView越界",
        )
        return self.bin[off : off + n]

    def accessor(self, i):
        a = self.doc["accessors"][i]
        fmt, size = COMPONENTS[a["componentType"]]
        width = WIDTHS[a["type"]]
        n = a["count"]
        require(0 <= n <= 60000000, "accessor数量超限")
        rows = [tuple(0 for _ in range(width)) for _ in range(n)]

        def read(view, count, offset, stride):
            b = self.view(view)
            require(
                offset >= 0
                and stride >= size * width
                and (
                    count == 0 or offset + (count - 1) * stride + size * width <= len(b)
                ),
                "accessor越界/步长无效",
            )
            return [
                struct.unpack_from("<" + fmt * width, b, offset + j * stride)
                for j in range(count)
            ]

        if "bufferView" in a:
            v = self.doc["bufferViews"][a["bufferView"]]
            rows = read(
                a["bufferView"],
                n,
                a.get("byteOffset", 0),
                v.get("byteStride", size * width),
            )
        if "sparse" in a:
            sp = a["sparse"]
            sc = sp["count"]
            ix = sp["indices"]
            vf = sp["values"]
            it, isz = COMPONENTS[ix["componentType"]]
            require(
                ix["componentType"] in (5121, 5123, 5125) and sc <= n, "无效sparse索引"
            )
            ib = self.view(ix["bufferView"])
            offset = ix.get("byteOffset", 0)
            require(offset >= 0 and offset + sc * isz <= len(ib), "sparse索引越界")
            indices = struct.unpack_from("<" + it * sc, ib, offset)
            require(
                all(0 <= x < n for x in indices)
                and all(x < y for x, y in zip(indices, indices[1:])),
                "sparse索引无序/越界",
            )
            vals = read(vf["bufferView"], sc, vf.get("byteOffset", 0), size * width)
            for j, row in zip(indices, vals):
                rows[j] = row
        if a.get("normalized") and a["componentType"] != 5126:
            maxv = {5120: 127, 5121: 255, 5122: 32767, 5123: 65535, 5125: 4294967295}[
                a["componentType"]
            ]
            rows = [tuple(max(-1.0, x / maxv) for x in row) for row in rows]
        return rows


def glb_fingerprint(doc, blob):
    # Fingerprint decoded accessors rather than padding/buffer offsets or extension data.
    obj = object.__new__(Glb)
    obj.doc = doc
    obj.bin = blob
    access = {}
    for mesh in doc.get("meshes", []):
        for p in mesh.get("primitives", []):
            for i in list(p.get("attributes", {}).values()) + (
                [p["indices"]] if "indices" in p else []
            ):
                access[str(i)] = obj.accessor(i)
    for s in doc.get("skins", []):
        if "inverseBindMatrices" in s:
            access[str(s["inverseBindMatrices"])] = obj.accessor(
                s["inverseBindMatrices"]
            )
    for a in doc.get("animations", []):
        for s in a.get("samplers", []):
            for k in ("input", "output"):
                access[str(s[k])] = obj.accessor(s[k])
    nodes = [
        {
            k: v
            for k, v in n.items()
            if k
            in (
                "mesh",
                "skin",
                "matrix",
                "translation",
                "rotation",
                "scale",
                "children",
            )
        }
        for n in doc.get("nodes", [])
    ]
    for result, node in zip(nodes, doc.get("nodes", [])):
        if EXTRA_GEOMETRY in node.get("extras", {}):
            result[EXTRA_GEOMETRY] = node["extras"][EXTRA_GEOMETRY]
    payload = {
        "meshes": doc.get("meshes", []),
        "skins": doc.get("skins", []),
        "nodes": nodes,
        "animations": doc.get("animations", []),
        "access": access,
    }
    return digest(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def node_matrix(node):
    if "matrix" in node:
        return transpose(node["matrix"])
    tx, ty, tz = node.get("translation", (0, 0, 0))
    x, y, z, w = node.get("rotation", (0, 0, 0, 1))
    sx, sy, sz = node.get("scale", (1, 1, 1))
    return [
        (1 - 2 * y * y - 2 * z * z) * sx,
        (2 * x * y - 2 * z * w) * sy,
        (2 * x * z + 2 * y * w) * sz,
        tx,
        (2 * x * y + 2 * z * w) * sx,
        (1 - 2 * x * x - 2 * z * z) * sy,
        (2 * y * z - 2 * x * w) * sz,
        ty,
        (2 * x * z - 2 * y * w) * sx,
        (2 * y * z + 2 * x * w) * sy,
        (1 - 2 * x * x - 2 * y * y) * sz,
        tz,
        0,
        0,
        0,
        1,
    ]


def extra_geometry_role(doc, node):
    extras = node.get("extras", {})
    if EXTRA_GEOMETRY not in extras and "mesh" in node:
        extras = doc.get("meshes", [])[node["mesh"]].get("extras", {})
    if EXTRA_GEOMETRY not in extras:
        return None
    role = extras[EXTRA_GEOMETRY]
    require(type(role) is bool, "sky_extra_geometry标记必须是true或false")
    return role


def import_glb(g, extra_only=False):
    d = g.doc
    nodes = d.get("nodes", [])
    parents = {}
    for i, n in enumerate(nodes):
        for j in n.get("children", []):
            require(j not in parents and 0 <= j < len(nodes), "节点存在多个父节点/越界")
            parents[j] = i
    worlds = {}

    def world(i, active=None):
        if i in worlds:
            return worlds[i]
        active = set() if active is None else active
        require(i not in active, "GLB节点成环")
        active.add(i)
        mat = node_matrix(nodes[i])
        worlds[i] = mm(world(parents[i], active), mat) if i in parents else mat
        active.remove(i)
        return worlds[i]

    for i in range(len(nodes)):
        world(i)
    roots = d.get("scenes", [{"nodes": list(range(len(nodes)))}])[
        d.get("scene", 0)
    ].get("nodes", [])
    active = set()

    def visit(i):
        if i in active:
            return
        active.add(i)
        for j in nodes[i].get("children", []):
            visit(j)

    for i in roots:
        visit(i)
    lod_ids = {
        j
        for n in nodes
        for j in n.get("extensions", {}).get("MSFT_lod", {}).get("ids", [])
    }
    tagged = any("sky_lod" in n.get("extras", {}) for n in nodes)
    selected = [
        (i, n)
        for i, n in enumerate(nodes)
        if "mesh" in n
        and (
            extra_geometry_role(d, n) is True
            if extra_only
            else extra_geometry_role(d, n) is None
            and (
                i in active
                or i in lod_ids
                or tagged
                and "sky_lod" in n.get("extras", {})
            )
        )
    ]
    if extra_only:
        animated_nodes = {
            ch.get("target", {}).get("node")
            for clip in d.get("animations", [])
            for ch in clip.get("channels", [])
        }
        for i, n in selected:
            require("skin" not in n, "附加几何只存静态XYZ，不能使用蒙皮")
            j = i
            while j >= 0:
                require(j not in animated_nodes, "附加几何或其父节点不能添加动画")
                j = parents.get(j, -1)
    groups = {}
    rigid_nodes = (
        [i for i, n in selected]
        if not extra_only
        and d.get("animations")
        and all("skin" not in n for i, n in selected)
        else []
    )
    has_skin = any(n.get("skin") is not None for _, n in selected)
    mixed_rigid_nodes = []
    skin_id = None
    mesh_world = identity()
    for i, n in selected:
        level = (
            0
            if extra_only
            else (
                n.get("extras", {}).get("sky_lod", 0)
                if tagged
                else (1 + sorted(lod_ids).index(i) if i in lod_ids else 0)
            )
        )
        geom = groups.setdefault(
            level,
            {
                "POSITION": [],
                "indices": [],
                "distance": float(n.get("extras", {}).get("sky_distance", "inf")),
            },
        )
        sk = n.get("skin")
        if sk is not None:
            require(skin_id is None or skin_id == sk, "多个独立骨架不能装入一个mesh")
            skin_id = sk
            if mesh_world != identity():
                require(mesh_world == worlds[i], "同一蒙皮有不同mesh节点变换")
            mesh_world = worlds[i]
        mesh = d["meshes"][n["mesh"]]
        for prim in mesh["primitives"]:
            require(
                not prim.get("targets"),
                "当前不生成游戏头部附加描述；未修改的来源GLB可精确保留",
            )
            require(
                "KHR_draco_mesh_compression" not in prim.get("extensions", {}),
                "先在建模软件解压Draco",
            )
            attrs = {k: g.accessor(v) for k, v in prim["attributes"].items()}
            require("POSITION" in attrs, "primitive缺少POSITION")
            if extra_only:
                require(
                    not any(k.startswith(("JOINTS_", "WEIGHTS_")) for k in attrs),
                    "附加几何不能使用蒙皮权重",
                )
                attrs = {"POSITION": attrs["POSITION"]}
            pos = attrs["POSITION"]
            require(all(len(p) == 3 for p in pos), "POSITION必须是XYZ")
            count = len(pos)
            if rigid_nodes:
                attrs["JOINTS_0"] = [(rigid_nodes.index(i), 0, 0, 0)] * count
                attrs["WEIGHTS_0"] = [(1.0, 0.0, 0.0, 0.0)] * count
            elif has_skin:
                skin_attrs = any(k.startswith(("JOINTS_", "WEIGHTS_")) for k in attrs)
                if sk is not None and skin_attrs:
                    require(
                        "JOINTS_0" in attrs and "WEIGHTS_0" in attrs,
                        "蒙皮缺少JOINTS_0/WEIGHTS_0",
                    )
                    joint_count = len(d["skins"][sk]["joints"])
                    require(
                        all(
                            len(row) == 4
                            and all(
                                isinstance(j, int) and 0 <= j < joint_count for j in row
                            )
                            for row in attrs["JOINTS_0"]
                        )
                        and all(len(row) == 4 for row in attrs["WEIGHTS_0"]),
                        "蒙皮分量数量错误或JOINTS_0超出GLB蒙皮范围",
                    )
                else:
                    # An unskinned primitive keeps its own node transform. A rigid
                    # joint also preserves attachments and independent animation.
                    if i not in mixed_rigid_nodes:
                        mixed_rigid_nodes.append(i)
                    attrs = {
                        k: v
                        for k, v in attrs.items()
                        if not k.startswith(("JOINTS_", "WEIGHTS_"))
                    }
                    # Negative markers exist only internally; input joint indices
                    # were validated above before these markers were introduced.
                    attrs["JOINTS_0"] = [(-i - 2,) * 4] * count
                    attrs["WEIGHTS_0"] = [(1.0, 0.0, 0.0, 0.0)] * count
            require(
                all(len(rows) == count for rows in attrs.values()), "属性顶点数量不一致"
            )
            ids = (
                [row[0] for row in g.accessor(prim["indices"])]
                if "indices" in prim
                else list(range(count))
            )
            require(
                all(isinstance(j, int) and 0 <= j < count for j in ids), "GLB索引越界"
            )
            mode = prim.get("mode", 4)
            if mode == 5:
                ids = [
                    v
                    for j in range(len(ids) - 2)
                    for v in (
                        (ids[j], ids[j + 1], ids[j + 2])
                        if j % 2 == 0
                        else (ids[j + 1], ids[j], ids[j + 2])
                    )
                    if len(set(ids[j : j + 3])) == 3
                ]
            elif mode == 6:
                ids = [
                    v
                    for j in range(1, len(ids) - 1)
                    for v in (ids[0], ids[j], ids[j + 1])
                ]
            else:
                require(mode == 4, "仅支持三角形/三角带/三角扇")
            require(len(ids) % 3 == 0, "三角索引数量不是3的倍数")
            mat = worlds[i]
            nm = transpose(inverse(mat))
            attrs["POSITION"] = [transform(mat, p) for p in pos]
            if "NORMAL" in attrs:
                attrs["NORMAL"] = [
                    normal(transform(nm, p, 0.0)) for p in attrs["NORMAL"]
                ]
            det = (
                mat[0] * (mat[5] * mat[10] - mat[6] * mat[9])
                - mat[1] * (mat[4] * mat[10] - mat[6] * mat[8])
                + mat[2] * (mat[4] * mat[9] - mat[5] * mat[8])
            )
            if det < 0:
                for j in range(0, len(ids), 3):
                    ids[j + 1], ids[j + 2] = ids[j + 2], ids[j + 1]
            base = len(geom["POSITION"])
            all_keys = set(geom) - {"indices", "distance"} | set(attrs)
            defaults = {
                "NORMAL": (0.0, 1.0, 0.0),
                "COLOR_0": (1.0, 1.0, 1.0, 1.0),
                "JOINTS_0": (0, 0, 0, 0),
                "WEIGHTS_0": (1.0, 0.0, 0.0, 0.0),
            }
            for k in all_keys:
                width = (
                    3
                    if k in ("NORMAL", "POSITION")
                    else 4 if k.startswith(("COLOR", "JOINTS", "WEIGHTS")) else 2
                )
                default = defaults.get(k, (0.0,) * width)
                if k not in geom:
                    geom[k] = [default] * base
                rows = attrs.get(k, [default] * count)
                if k == "COLOR_0":
                    rows = [tuple(x) + (1.0,) if len(x) == 3 else x for x in rows]
                geom[k].extend(rows)
            geom["indices"].extend(base + j for j in ids)
    if extra_only:
        for geom in groups.values():
            require(
                geom["POSITION"] and geom["indices"], "附加几何必须包含顶点和三角形"
            )
            encode_extra_geometry(geom)
        return list(groups.values()), [], False
    bones = []
    if not selected:
        skins = d.get("skins", [])
        require(len(skins) <= 1, "动画资源存在多个独立骨架")
        if skins:
            skin_id = 0
        elif d.get("animations"):
            rigid_nodes = sorted(active)
    if skin_id is not None:
        skin = d["skins"][skin_id]
        original_joints = skin["joints"]
        virtual = [
            j
            for j in original_joints
            if nodes[j].get("extras", {}).get("sky_identity_slot") is True
        ]
        require(len(virtual) <= 1, "蒙皮有多个保留的mesh变换槽")
        js = [j for j in original_joints if j not in virtual]
        require(len(js) <= 255, "游戏单mesh最多255根可索引骨骼")
        mapping = {j: i for i, j in enumerate(js)}
        ib = (
            g.accessor(skin["inverseBindMatrices"])
            if "inverseBindMatrices" in skin
            else [identity() for _ in original_joints]
        )
        require(len(ib) == len(original_joints), "inverseBindMatrices数量不一致")
        for j, row in zip(original_joints, ib):
            if j in virtual:
                require(not nodes[j].get("children"), "保留的mesh变换槽不能有子节点")
                require(
                    all(
                        abs(x - y) < 1e-6
                        for x, y in zip(
                            mm(transpose(row), inverse(mesh_world)), identity()
                        )
                    )
                    and all(abs(x - y) < 1e-6 for x, y in zip(worlds[j], identity())),
                    "保留的mesh变换槽不能修改变换",
                )
                require(
                    not any(
                        ch.get("target", {}).get("node") == j
                        for clip in d.get("animations", [])
                        for ch in clip.get("channels", [])
                    ),
                    "保留的mesh变换槽不能添加动画",
                )
                continue
            p = parents.get(j, -1)
            while p >= 0 and p not in mapping:
                p = parents.get(p, -1)
            bones.append(
                {
                    "name": nodes[j].get("name", f"bone{mapping[j]}"),
                    "parent": mapping.get(p, -1),
                    "matrix": mm(transpose(row), inverse(mesh_world)),
                    "node": j,
                }
            )
        animated_nodes = {
            ch.get("target", {}).get("node")
            for clip in d.get("animations", [])
            for ch in clip.get("channels", [])
        }
        attached_or_animated = set(original_joints) | animated_nodes
        moving_rigid_nodes = []
        for j in mixed_rigid_nodes:
            p = j
            while p >= 0:
                if p in attached_or_animated:
                    moving_rigid_nodes.append(j)
                    break
                p = parents.get(p, -1)
        # World-static geometry uses palette slot 0, leaving all 255 bone slots
        # available. Only attachments/animated rigid parts need an extra bone.
        rigid_mapping = {j: len(bones) + k for k, j in enumerate(moving_rigid_nodes)}
        hierarchy_mapping = {**mapping, **rigid_mapping}
        for j in moving_rigid_nodes:
            p = parents.get(j, -1)
            while p >= 0 and p not in hierarchy_mapping:
                p = parents.get(p, -1)
            bones.append(
                {
                    "name": f"SKY_rigid_mesh_{j}",
                    "parent": hierarchy_mapping.get(p, -1),
                    "matrix": inverse(worlds[j]),
                    "node": j,
                }
            )
        require(len(bones) <= 255, "游戏单mesh最多255根可索引骨骼（含刚性部件）")
        remap_joints = {i: mapping.get(j, -1) for i, j in enumerate(original_joints)}
        remap_joints.update(
            {-j - 2: rigid_mapping.get(j, -1) for j in mixed_rigid_nodes}
        )
        for geom in groups.values():
            if "JOINTS_0" in geom:
                require(
                    all(
                        isinstance(j, int) and j in remap_joints
                        for row in geom["JOINTS_0"]
                        for j in row
                    ),
                    "JOINTS_0超出GLB蒙皮范围",
                )
                geom["JOINTS_0"] = [
                    tuple(remap_joints[j] for j in row) for row in geom["JOINTS_0"]
                ]
    if rigid_nodes:
        require(len(rigid_nodes) <= 255, "游戏单mesh最多255根可索引骨骼")
        mapping = {j: i for i, j in enumerate(rigid_nodes)}
        for j in rigid_nodes:
            parent = parents.get(j, -1)
            while parent >= 0 and parent not in mapping:
                parent = parents.get(parent, -1)
            bones.append(
                {
                    "name": nodes[j].get("name", f"bone{mapping[j]}"),
                    "parent": mapping.get(parent, -1),
                    "matrix": inverse(worlds[j]),
                    "node": j,
                }
            )
    require(
        groups
        or bones
        or any(extra_geometry_role(d, n) is not None for n in nodes if "mesh" in n),
        "GLB场景没有可转换的几何或骨架",
    )
    for geom in groups.values():
        require(geom["POSITION"], "空几何不能转换")
        require(
            "JOINTS_1" not in geom and "WEIGHTS_1" not in geom,
            "超过4个权重不能无损转换，请在建模软件限制为4个",
        )
        if bones:
            require(
                "JOINTS_0" in geom and "WEIGHTS_0" in geom, "蒙皮缺少JOINTS_0/WEIGHTS_0"
            )
        for k, rows in geom.items():
            if isinstance(rows, list) and k != "indices":
                require(
                    all(math.isfinite(v) for row in rows for v in row), f"{k}含非有限数"
                )
    return [groups[i] for i in sorted(groups)], bones, bool(d.get("animations"))


def bounds_of(pos):
    return tuple(min(p[i] for p in pos) for i in range(3)) + tuple(
        max(p[i] for p in pos) for i in range(3)
    )


def pack_indices(ids, idx32):
    return struct.pack("<" + ("I" if idx32 else "H") * len(ids), *ids)


def area_stream(pos, ids):
    out = []
    acc = 0.0
    for i in range(0, len(ids), 3):
        a, b, c = (pos[j] for j in ids[i : i + 3])
        u = [a[j] - c[j] for j in range(3)]
        v = [b[j] - c[j] for j in range(3)]
        cross = (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
        acc += math.sqrt(sum(x * x for x in cross)) * 0.5
        out.append(acc)
    return struct.pack("<" + "f" * len(out), *out)


def quantize(x, lo, hi, maximum):
    return (
        max(0, min(maximum, int((x - lo) / (hi - lo) * maximum + 0.5)))
        if hi > lo
        else 0
    )


def skin_equivalent(a, b):
    """Compare effective influences, including duplicate joints and palette slot 0."""
    if ("JOINTS_0" in a) != ("JOINTS_0" in b):
        return False
    if "JOINTS_0" not in a:
        return True
    for aj, aw, bj, bw in zip(
        a["JOINTS_0"], a["WEIGHTS_0"], b["JOINTS_0"], b["WEIGHTS_0"]
    ):
        aa, bb = {}, {}
        for out, js, ws in ((aa, aj, aw), (bb, bj, bw)):
            total = sum(ws)
            if total <= 0:
                return False
            for j, w in zip(js, ws):
                if w > 0:
                    out[j] = out.get(j, 0.0) + w / total
        if aa.keys() != bb.keys() or any(abs(aa[j] - bb[j]) > 1e-6 for j in aa):
            return False
    return True


def vertex_maps(pos, norms, source_groups=None, source_positions=None):
    """Position welding and position/normal smoothing groups; never UV welding."""
    pmap, nmap, pm, nm = {}, {}, [], []
    for i, (p, norm) in enumerate(zip(pos, norms)):
        # Float equality treats signed zero as the same position/direction.
        pk = struct.unpack("<3f", struct.pack("<3f", *p))
        # Quantization may place distinct authored positions on the same grid point.
        # An unchanged source still supplies their original position groups.
        normal_position = source_positions[i] if source_positions is not None else pk
        nk = (
            (normal_position, source_groups[i])
            if source_groups is not None
            else (
                normal_position,
                struct.unpack("<3f", struct.pack("<3f", *normal(norm))),
            )
        )
        if pk not in pmap:
            pmap[pk] = len(pmap)
        if nk not in nmap:
            nmap[nk] = len(nmap)
        pm.append(pmap[pk])
        nm.append(nmap[nk])
    return pm, nm, len(pmap), len(nmap)


def build_lod(geom, version, flags, bones, reference=None):
    pos = geom["POSITION"]
    ids = geom["indices"]
    n = len(pos)
    ni = len(ids)
    require(ni > 0, "mesh必须有三角形")
    zp = "ZipPos" in flags
    zu = "ZipUvs" in flags
    hn = "StripNorm" not in flags
    require(
        version >= 29 or not (zp or zu or not hn),
        "0x19..0x1C不支持ZipPos/ZipUvs/StripNorm",
    )
    require(
        not zp
        or all(
            all(abs(x - 1.0) < 1e-6 for x in row[:3]) for row in geom.get("COLOR_0", [])
        ),
        "ZipPos仅保留顶点颜色alpha；RGB有色顶点请取消ZipPos",
    )
    idx32 = int(version <= 29 or "ForceIdx32" in flags or n > 65536 or max(ids) > 65535)
    bo = bounds_of(pos)
    uvs = [geom.get("TEXCOORD_" + str(i), [(0.0, 0.0)] * n) for i in range(4)]
    lo = tuple(min(row[j] for row in uvs[i]) for i in range(4) for j in range(2))
    hi = tuple(max(row[j] for row in uvs[i]) for i in range(4) for j in range(2))
    z13 = zu and "StripUv13" not in flags
    s = {}
    groups = (0, 0, 0, 0)
    counts = (0, 0, 0, 0, 0)
    third = 0
    hs = int(version < 29)
    same = False
    same_positions = same_normals = same_skin = False
    norms = geom.get("NORMAL") or smooth_normals(pos, ids)
    refgeom = None
    if reference:
        refgeom = decode_lod(reference)
        same = reference.vertices == n and refgeom["indices"] == ids
        if same:
            same_positions = all(
                struct.pack("<3f", *p) == struct.pack("<3f", *q)
                for p, q in zip(pos, refgeom["POSITION"])
            )
            same_normals = ("NORMAL" not in geom and not reference.has_normals) or (
                "NORMAL" in geom
                and reference.has_normals
                and all(
                    max(abs(x - y) for x, y in zip(normal(p), q)) < 1e-6
                    for p, q in zip(geom["NORMAL"], refgeom["NORMAL"])
                )
            )
            same_skin = skin_equivalent(geom, refgeom)
        if same:
            s = {
                k: v
                for k, v in reference.streams.items()
                if k
                in (
                    "secondary_indices",
                    "position_map",
                    "normal_map",
                    "strips",
                    "metrics",
                    "pairs",
                )
            }
            groups = reference.groups
            counts = reference.counts
            third = reference.third
            # Versions 25..28 always contain this stream, without a presence flag.
            hs = int(version < 29 or reference.has_secondary)
            if reference.idx32 != idx32:
                for key in (
                    "secondary_indices",
                    "position_map",
                    "normal_map",
                    "strips",
                    "pairs",
                ):
                    if key in s:
                        vals = [
                            x[0]
                            for x in struct.iter_unpack(
                                "<I" if reference.idx32 else "<H", s[key]
                            )
                        ]
                        restart = 0xFFFFFFFF if reference.idx32 else 0xFFFF
                        if key == "strips":
                            vals = [
                                (0xFFFFFFFF if idx32 else 0xFFFF) if x == restart else x
                                for x in vals
                            ]
                        s[key] = pack_indices(vals, idx32)
        elif bones:
            groups = reference.groups if sum(reference.groups) == n else (n, 0, 0, 0)
    if "CompOcc" in flags:
        require(
            same and same_positions and same_skin and reference.counts[2] > 0,
            "CompOcc尚不能重新烘焙；仅能保留位置、索引和蒙皮未变的来源数据。"
            "修改几何后请取消CompOcc",
        )
    else:
        counts = counts[:2] + (0, 0, counts[4])
        s.pop("strips", None)
        s.pop("metrics", None)
    if any(f in flags for f in ("CompEdges", "CompAdj")):
        require(
            same and same_positions and same_normals and same_skin,
            "边/邻接数据尚不能重新烘焙；需要位置、法线、索引和蒙皮未变的来源数据",
        )
    # Keep authored smoothing groups when the source omitted its normal stream.
    # UV seams do not split these groups. Moving coincident vertices apart does.
    keep_normal_groups = same and "NORMAL" not in geom and reference.counts[1] > 0
    if not same or not same_positions or not same_normals:
        source_groups = (
            [
                x[0]
                for x in struct.iter_unpack(
                    "<I" if reference.idx32 else "<H", reference.streams["normal_map"]
                )
            ]
            if keep_normal_groups
            else None
        )
        keep_positions = same and same_positions and reference.counts[0] > 0
        source_positions = (
            [
                x[0]
                for x in struct.iter_unpack(
                    "<I" if reference.idx32 else "<H", reference.streams["position_map"]
                )
            ]
            if keep_positions
            else None
        )
        pm, nm, np, nn = vertex_maps(pos, norms, source_groups, source_positions)
        counts = (
            counts[0] if keep_positions else np,
            nn,
            counts[2],
            counts[3],
            counts[4],
        )
        if not keep_positions:
            s["position_map"] = pack_indices(pm, idx32)
        s["normal_map"] = pack_indices(nm, idx32)
    if zp:
        s["zip_positions"] = b"".join(
            struct.pack(
                "<I",
                sum(
                    quantize(p[j], bo[j], bo[j + 3], 1023) << (20 - 10 * j)
                    for j in range(3)
                ),
            )
            for p in pos
        )
        s["alpha"] = bytes(
            quantize(row[3], 0, 1, 255)
            for row in geom.get("COLOR_0", [(1.0, 1.0, 1.0, 1.0)] * n)
        )
    else:
        cols = geom.get("COLOR_0", [(1.0, 1.0, 1.0, 1.0)] * n)
        s["positions"] = b"".join(
            struct.pack(
                "<3fI",
                *p,
                sum(quantize(col[j], 0, 1, 255) << (j * 8) for j in range(4)),
            )
            for p, col in zip(pos, cols)
        )
    if hn:
        s["normals"] = b"".join(
            struct.pack(
                "4b",
                *(max(-127, min(127, int(round(x * 127)))) for x in normal(norm)),
                0,
            )
            for norm in norms
        )
    if zu:
        s["zip_uvs"] = bytes(
            quantize(uvs[ch][i][j], lo[ch * 2 + j], hi[ch * 2 + j], 255)
            for i in range(n)
            for ch in (0, 2)
            for j in range(2)
        )
        if z13:
            s["zip_uv13"] = bytes(
                quantize(uvs[ch][i][j], lo[ch * 2 + j], hi[ch * 2 + j], 255)
                for i in range(n)
                for ch in (1, 3)
                for j in range(2)
            )
    else:
        if "StripUv13" in flags:
            uvs[1] = [(0.0, 0.0)] * n
            uvs[3] = [(0.0, 0.0)] * n
        require(
            all(abs(x) <= 65504 for rows in uvs for row in rows for x in row),
            "UV超过half浮点范围65504",
        )
        s["uvs"] = b"".join(
            struct.pack("<8e", *(x for ch in range(4) for x in uvs[ch][i]))
            for i in range(n)
        )
    if bones:
        rows = []
        groupcounts = [0] * 4
        for joints, weights in zip(geom["JOINTS_0"], geom["WEIGHTS_0"]):
            require(
                all(isinstance(j, int) and -1 <= j < len(bones) for j in joints),
                "JOINTS_0超出骨骼范围",
            )
            require(all(w >= 0 for w in weights) and sum(weights) > 0, "无效蒙皮权重")
            total = sum(weights)
            ws = [w / total * 255 for w in weights]
            wi = [int(x) for x in ws]
            for j in sorted(range(4), key=lambda j: ws[j] - wi[j], reverse=True)[
                : 255 - sum(wi)
            ]:
                wi[j] += 1
            js = [j + 1 if w else 0 for j, w in zip(joints, wi)]
            rows.append(bytes(js + wi))
            groupcounts[max(1, sum(bool(w) for w in wi)) - 1] += 1
        s["weights"] = b"".join(rows)
        # Game count fields partition vertices by number of influences. Reorder if needed.
        order = sorted(range(n), key=lambda i: sum(bool(x) for x in rows[i][4:]))
        if order != list(range(n)):
            require(
                not any(f in flags for f in ("CompOcc", "CompEdges", "CompAdj")),
                "参考顶点的权重组发生变化；无法保留拓扑辅助数据",
            )
            remap = {old: new for new, old in enumerate(order)}
            gg = {
                k: (
                    [v[i] for i in order]
                    if isinstance(v, list) and k not in ("indices", "distance")
                    else v
                )
                for k, v in geom.items()
            }
            gg["indices"] = [remap[i] for i in ids]
            gg["_sorted"] = True
            return build_lod(gg, version, flags, bones, None)
        groups = tuple(groupcounts)
    s["indices"] = pack_indices(ids, idx32)
    if hs and "secondary_indices" not in s:
        s["secondary_indices"] = s["indices"]
    s["areas"] = area_stream(pos, ids)
    return Lod(
        geom.get("distance", math.inf),
        bo + bo,
        lo,
        hi,
        n,
        ni,
        idx32,
        counts,
        int(hn),
        hs,
        third,
        (n * 4 if zp else 0, n * 4 if zu else 0, n * 4 if z13 else 0),
        groups,
        s,
    )


def encode_lod(l, version, animated):
    out = bytearray(struct.pack("<f", l.distance))
    out.extend(
        struct.pack(
            "<" + ("12f" if version >= 28 else "6f"),
            *(l.bounds if version >= 28 else l.bounds[:6]),
        )
    )
    if version > 28:
        out.extend(struct.pack("<16f", *l.quant_lo, *l.quant_hi))
    out.extend(struct.pack("<2I", l.vertices, l.indices_count))
    if version > 29:
        out.extend(struct.pack("<I", l.idx32))
    out.extend(struct.pack("<5I", *l.counts))
    if version >= 29:
        out.extend(
            struct.pack(
                "<3B7I", l.has_normals, l.has_secondary, l.third, *l.zipped, *l.groups
            )
        )
    else:
        out.extend(struct.pack("<4I", *l.groups))
    keys = []
    if not l.zipped[0]:
        keys.append("positions")
    if l.has_normals:
        keys.append("normals")
    if not l.zipped[1]:
        keys.append("uvs")
    if animated:
        keys.append("weights")
    keys.append("indices")
    if l.has_secondary:
        keys.append("secondary_indices")
    for i, k in enumerate(("position_map", "normal_map", "strips", "metrics", "pairs")):
        if l.counts[i]:
            keys.append(k)
    if version > 28:
        keys.append("areas")
    if l.zipped[0]:
        keys.extend(("zip_positions", "alpha"))
    if l.zipped[1]:
        keys.append("zip_uvs")
    if l.zipped[2]:
        keys.append("zip_uv13")
    for k in keys:
        out.extend(l.streams[k])
    return bytes(out)


def encode_mesh(m, compressed=None):
    v = m.version
    comp = m.compressed if compressed is None else int(compressed)
    require(v >= 27 or not comp, "0x19/0x1A没有外层压缩字段")
    require(bool(m.extra) == bool(m.extra_data), "附加几何结构标记与数据不一致")
    if m.extra:
        decode_extra_geometry(m.extra_data)
    out = bytearray(
        struct.pack("<I", v)
        + fixed_name(m.name)
        + struct.pack("<I2B", len(m.lods), m.animated, m.extra)
    )
    if v > 31:
        require(len(m.descriptors) % 112 == 0, "无效头部附加描述")
        out.extend(struct.pack("<I", len(m.descriptors) // 112) + m.descriptors)
    else:
        require(not m.descriptors, "该版本没有头部附加描述字段")
    payload = b"".join(encode_lod(l, v, m.animated) for l in m.lods) + m.extra_data
    if v >= 27:
        out.extend(struct.pack("<I", comp))
    if comp:
        block = lz4_encode(payload)
        out.extend(struct.pack("<2I", len(block), len(payload)) + block)
    else:
        out.extend(payload)
    out.extend(m.skeleton)
    result = bytes(out)
    read_mesh(result, m.name + ".mesh")
    return result


def new_skeleton(name, bones):
    # StripAnim: bones/names/parents retained; game skips bind matrices on this path.
    out = bytearray(
        struct.pack("<I", 11)
        + fixed_name(name)
        + struct.pack(
            "<3IBI",
            len(bones),
            0,
            30,
            2,
            sum(len(b["name"].encode()) + 1 for b in bones),
        )
    )
    for b in bones:
        out.extend(
            fixed_name(b["name"])
            + struct.pack("<16fI", *transpose(b["matrix"]), b["parent"] + 1)
        )
    return bytes(out)


def source_mesh(g):
    e = g.doc.get("extensions", {}).get(EXT)
    if not e:
        return None, None
    raw = g.view(e["bufferView"])
    require(digest(raw) == e["sha256"], "GLB内嵌mesh校验和不匹配")
    m = read_mesh(raw, "source" + suffix(set(e.get("filenameFlags", []))) + ".mesh")
    return m, e


def source_lods_unchanged(g, reference):
    # Legacy geometryHash values omit LOD extras. Compare against the embedded
    # source as well so old GLBs remain eligible for exact byte restoration.
    nodes = g.doc.get("nodes", [])
    if len(nodes) < len(reference.lods):
        return False
    for i, node in enumerate(nodes):
        extras = node.get("extras", {})
        actual = {k: extras[k] for k in ("sky_lod", "sky_distance") if k in extras}
        expected = {}
        if i < len(reference.lods):
            distance = reference.lods[i].distance
            expected = {
                "sky_lod": i,
                "sky_distance": distance if math.isfinite(distance) else "inf",
            }
        if actual != expected:
            return False
    return True


def convert_glb(g, filename, version, flags=None, compress=True, fps=None):
    ref, meta = source_mesh(g)
    flags = set(
        detect_flags(ref)
        if flags is None and ref
        else split_name(filename)[1] if flags is None else flags
    )
    require(flags <= set(FLAG_ORDER), f"未知参数: {flags-set(FLAG_ORDER)}")
    same = bool(
        ref
        and meta
        and glb_fingerprint(g.doc, g.bin) == meta["geometryHash"]
        and source_lods_unchanged(g, ref)
    )
    if (
        same
        and version == ref.version
        and flags == set(meta["flags"])
        and bool(compress) == bool(ref.compressed)
        and fps is None
    ):
        return ref.raw, flags, "完整原始数据恢复（字节一致）"
    geoms, bones, has_animations = import_glb(g)
    extra_geoms, _, _ = import_glb(g, extra_only=True)
    extra_roles = [
        extra_geometry_role(g.doc, n) for n in g.doc.get("nodes", []) if "mesh" in n
    ]
    extra_data = (
        encode_extra_geometry(extra_geoms[0])
        if extra_geoms
        else (
            b""
            if any(role is not None for role in extra_roles)
            else ref.extra_data if ref else b""
        )
    )
    animation = _sky_animation

    if ref:
        changed_refs = (flags ^ detect_flags(ref)) & REFERENCE_ONLY
        require(not changed_refs, f"这些参数没有可用原始数据: {sorted(changed_refs)}")
    else:
        require(
            not (flags & (REFERENCE_ONLY | {"CompOcc"})),
            f"这些烘焙参数需要原始游戏参考: {sorted(flags & (REFERENCE_ONLY | {'CompOcc'}))}",
        )
    if bones:
        skeleton = new_skeleton(ref.name if ref else split_name(filename)[0], bones)
        reuse = bool(
            ref
            and ref.animated
            and len(bones) == len(read_skeleton(ref.skeleton)["bones"])
            and fps is None
            and (
                (not has_animations and not meta.get("animationClipsExported"))
                or meta
                and animation.fingerprint(g) == meta.get("animationHash")
            )
        )
        if reuse:
            sb = read_skeleton(ref.skeleton)["bones"]
            require(
                [b["name"] for b in sb] == [b["name"] for b in bones],
                "原始骨架名字/顺序已改变；不能复用原始动画",
            )
            skeleton = ref.skeleton
            sv = read_skeleton(skeleton)["version"]
            offset = 4 + 64 + 12 + 1 + (4 if sv >= 10 else 0)
            out = bytearray(skeleton)
            for i, b in enumerate(bones):
                out[offset + i * 132 : offset + (i + 1) * 132] = fixed_name(
                    b["name"]
                ) + struct.pack("<16fI", *transpose(b["matrix"]), b["parent"] + 1)
            skeleton = bytes(out)
            ga = animation.read_animation(skeleton)
            desired = 0 if "UncompAnim" in flags else 2
            if ga.clips and ga.compression != desired:
                skeleton = animation.encode_animation(skeleton, ga, desired)
        elif has_animations:
            ga = animation.from_glb(g, bones, fps)
            skeleton = animation.encode_animation(
                skeleton, ga, 0 if "UncompAnim" in flags else 2
            )
        if read_skeleton(skeleton)["frame_count"]:
            flags.discard("StripAnim")
        else:
            flags.add("StripAnim")
            flags.discard("UncompAnim")
    else:
        require(not has_animations, "动画未关联到可转换的几何")
        skeleton = b""
        flags.discard("StripAnim")
        flags.discard("UncompAnim")
    if geoms:
        flags.discard("StripGeo")
    else:
        flags.add("StripGeo")
        flags.difference_update(
            {"CompOcc", "ZipPos", "ZipUvs", "StripUv13", "StripNorm", "ForceIdx32"}
        )
    if geoms and (
        version <= 29
        or any(
            len(gm["POSITION"]) > 65536 or max(gm["indices"], default=0) > 65535
            for gm in geoms
        )
    ):
        flags.add("ForceIdx32")
    else:
        if "ForceIdx32" not in flags:
            flags.discard("ForceIdx32")
    refs = ref.lods if ref else []
    lods = [
        build_lod(geom, version, flags, bones, refs[i] if i < len(refs) else None)
        for i, geom in enumerate(geoms)
    ]
    name = ref.name if ref else split_name(filename)[0]
    m = Mesh(
        version,
        name,
        int(bool(bones)),
        int(bool(extra_data)),
        ref.descriptors if ref and same else b"",
        lods,
        skeleton,
        extra_data,
        int(compress and version >= 27),
        b"",
        flags,
    )
    require(
        not ref or not ref.descriptors or same,
        "该mesh含头部附加描述；修改后暂不能重新生成",
    )
    return encode_mesh(m), flags, "已重新编码"


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"输出已存在，未覆盖: {path}")
    fd, tmp = tempfile.mkstemp(prefix=".sky-convert-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def convert_file(path, output=None, version=32, flags=None, compress=True, fps=None):
    path = Path(path)
    require(path.is_file(), f"文件不存在: {path}")
    if path.suffix.lower() == ".mesh":
        m = read_mesh(path.read_bytes(), path.name)
        fs = detect_flags(m)
        base, _ = split_name(path)
        out = (
            Path(output)
            if output
            else path.with_name(base + ("." + suffix(fs) if fs else "") + ".glb")
        )
        result = export_glb(m)
        # Verify the exported GLB parser and embedded source before publishing.
        g = Glb(result)
        src, meta = source_mesh(g)
        require(src.raw == m.raw, "来源mesh保留失败")
        atomic_write(out, result)
        print(
            f'版本: 0x{m.version:02X} | LOD: {len(m.lods)} | 骨骼: {len(read_skeleton(m.skeleton)["bones"]) if m.animated else 0}'
        )
        for w in m.warnings:
            print("核对:", w)
        if m.animated and read_skeleton(m.skeleton)["frame_count"]:
            print("游戏动画已导出为标准glTF动画轨道，原始数据同时保留。")
        if m.extra:
            eg = decode_extra_geometry(m.extra_data)
            print(
                f'附加几何: {len(eg["POSITION"])}顶点 / {len(eg["indices"]) // 3}三角形；GLB的“附加几何”场景可单独编辑。'
            )
        print("完成:", out)
        return out
    require(path.suffix.lower() == ".glb", "输入必须是.mesh或.glb")
    g = Glb(path.read_bytes())
    result, fs, status = convert_glb(g, path.name, version, flags, compress, fps)
    base, _ = split_name(path)
    out = Path(output) if output else path.with_name(base + suffix(fs) + ".mesh")
    atomic_write(out, result)
    print(status)
    print("完成:", out)
    return out


def _menu_color(text, color="white", bold=False, enabled=None):
    if enabled is None:
        enabled = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
    if not enabled:
        return text
    code = {"white": "37", "green": "32", "red": "31"}[color]
    return f'\x1b[{"1;" if bold else ""}{code}m{text}\x1b[0m'


def _menu_width(text):
    return sum(
        (
            0
            if unicodedata.combining(ch)
            else 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        )
        for ch in text
    )


def _menu_clip(text, width):
    if _menu_width(text) <= width:
        return text
    out = ""
    for ch in text:
        if _menu_width(out + ch) > max(0, width - 1):
            break
        out += ch
    return out + "…" if width else ""


def _menu_pad(text, width):
    text = _menu_clip(text, width)
    return text + " " * max(0, width - _menu_width(text))


def _menu_wrap(text, width, count=2):
    lines, current = [], ""
    for ch in text:
        if _menu_width(current + ch) > width and current:
            lines.append(current)
            current = ""
            if len(lines) == count - 1:
                return lines + [_menu_clip(text[sum(len(x) for x in lines) :], width)]
        current += ch
    return lines + [current]


class _MenuTerminal:
    """ANSI font styling and optional cbreak input, with a plain-input fallback."""

    def __init__(self):
        self.raw = False
        self.color = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
        self.row_map = {}
        self.size = shutil.get_terminal_size((88, 28))

    def __enter__(self):
        if (
            os.name == "posix"
            and sys.stdin.isatty()
            and sys.stdout.isatty()
            and os.environ.get("TERM") != "dumb"
        ):
            import termios, tty

            self.fd = sys.stdin.fileno()
            self.saved = termios.tcgetattr(self.fd)
            try:
                tty.setcbreak(self.fd)
                self.raw = True
                sys.stdout.write("\x1b[?1049h\x1b[2J\x1b[H\x1b[?1003h\x1b[?1006h")
                sys.stdout.flush()
            except BaseException:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
                self.raw = False
                raise
        return self

    def __exit__(self, *args):
        if self.raw:
            import termios

            try:
                sys.stdout.write("\x1b[?1003l\x1b[?1006l\x1b[0m\x1b[?25h\x1b[?1049l")
                sys.stdout.flush()
            finally:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def draw(self, title, rows, focus, controls, message="", error=False, buffer=""):
        self.size = shutil.get_terminal_size((88, 28))
        self.width = max(12, min(94, self.size.columns - 8))
        page_size = max(1, self.size.lines - 12)
        start = focus // page_size * page_size
        visible = rows[start : start + page_size]
        paint = lambda s, color="white", bold=False: _menu_color(
            s, color, bold, self.color
        )
        lines = [
            "  " + paint(_menu_clip("SKY  ·  Mesh ↔ GLB", self.width), bold=True),
            "  " + paint(_menu_clip(title, self.width), bold=True),
        ]
        lines.append("  ┌" + "─" * (self.width + 2) + "┐")
        self.row_map = {}
        for i, row in enumerate(visible, start):
            color = (
                "red"
                if row.get("locked")
                else "green" if row.get("selected") else "white"
            )
            mark = "●" if row.get("selected") else "○"
            label_width = min(16, max(5, self.width - 22))
            text = f'{"›" if i == focus else " "} {i + 1:2}. {mark} {_menu_pad(row["label"], label_width)} {row.get("status", "")}'
            self.row_map[len(lines) + 1] = i
            lines.append(
                "  │ " + paint(_menu_pad(text, self.width), color, i == focus) + " │"
            )
        lines.append("  └" + "─" * (self.width + 2) + "┘")
        detail = rows[focus].get("description", "")
        lines.extend("  " + paint(s) for s in _menu_wrap(detail, self.width))
        lines.extend("  " + paint(_menu_clip(s, self.width)) for s in controls)
        if len(rows) > page_size:
            message = (
                message
                or f"显示 {start + 1}–{start + len(visible)} / {len(rows)} 项；上下键或滚轮翻页"
            )
        lines.append(
            "  "
            + paint(_menu_clip(message or " ", self.width), "red" if error else "white")
        )
        if self.raw:
            lines.append(
                "  "
                + paint(
                    "输入 › " + _menu_clip(buffer, max(1, self.width - 7)), bold=True
                )
            )
            sys.stdout.write("\x1b[H" + "\n".join(lines) + "\x1b[J")
            sys.stdout.flush()
        else:
            print("\n" + "\n".join(lines))

    def read_event(self):
        if not self.raw:
            return "line", input("  输入 › ")
        import select

        while not select.select([self.fd], [], [], 0.2)[0]:
            if shutil.get_terminal_size((88, 28)) != self.size:
                return "resize", None
        ch = os.read(self.fd, 1)
        if not ch or ch == b"\x04":
            raise EOFError
        if ch == b"\x03":
            raise KeyboardInterrupt
        if ch in (b"\r", b"\n"):
            return "enter", None
        if ch == b" ":
            return "space", None
        if ch in (b"\x7f", b"\x08"):
            return "backspace", None
        if ch == b"\t":
            return "down", None
        if ch != b"\x1b":
            return "char", ch.decode("ascii", "replace")
        sequence = b""
        if select.select([self.fd], [], [], 0.04)[0]:
            sequence += os.read(self.fd, 1)
        if sequence in (b"[", b"O"):
            while len(sequence) < 64 and select.select([self.fd], [], [], 0.04)[0]:
                part = os.read(self.fd, 1)
                sequence += part
                if 0x40 <= part[0] <= 0x7E:
                    break
        key = {
            b"[A": "up",
            b"OA": "up",
            b"[B": "down",
            b"OB": "down",
            b"[C": "down",
            b"[D": "up",
            b"[H": "home",
            b"[1~": "home",
            b"OH": "home",
            b"[F": "end",
            b"[4~": "end",
            b"OF": "end",
            b"[5~": "page_up",
            b"[6~": "page_down",
        }.get(sequence)
        if key:
            return key, None
        mouse = re.fullmatch(rb"\[<(\d+);(\d+);(\d+)([Mm])", sequence)
        if mouse:
            button, x, y = (int(v) for v in mouse.groups()[:3])
            return "mouse", (button, x, y, mouse.group(4) == b"M")
        return "escape", None

    def navigate(self, event, focus, total):
        kind, value = event
        if kind in ("up", "down", "page_up", "page_down"):
            step = max(1, self.size.lines - 12) if kind.startswith("page") else 1
            target = focus + (-step if kind in ("up", "page_up") else step)
            return (
                min(total - 1, max(0, target))
                if kind.startswith("page")
                else target % total
            ), False
        if kind == "home":
            return 0, False
        if kind == "end":
            return total - 1, False
        if kind == "mouse":
            button, x, y, pressed = value
            if button & 64:
                return (
                    min(total - 1, max(0, focus + (-1 if button & 1 == 0 else 1))),
                    False,
                )
            if 3 <= x <= self.width + 6 and y in self.row_map:
                return self.row_map[y], pressed and button & 35 == 0
        return focus, False


def _menu_choice(
    title, labels, descriptions, default=0, back="q", values=None, message=""
):
    focus, buffer = default, ""
    with _MenuTerminal() as ui:
        while True:
            rows = [
                {
                    "label": label,
                    "selected": i == focus,
                    "status": "当前选择" if i == focus else "",
                    "description": descriptions[i],
                }
                for i, label in enumerate(labels)
            ]
            ui.draw(
                title,
                rows,
                focus,
                [
                    "↑ / ↓ 定位  ·  输入编号或版本值",
                    f"回车确认  ·  {back} {'返回参数表' if back == 'b' else '退出'}",
                ],
                message,
                error=bool(message),
                buffer=buffer,
            )
            event = ui.read_event()
            kind, value = event
            old_focus = focus
            focus, clicked = ui.navigate(event, focus, len(rows))
            if (
                kind in ("up", "down", "home", "end", "page_up", "page_down")
                or clicked
                or focus != old_focus
            ):
                buffer = ""
            if kind == "line":
                return value.strip()
            if kind == "escape":
                return back
            if kind == "enter":
                return buffer.strip() or str(focus + 1)
            if kind == "backspace":
                buffer = buffer[:-1]
            elif kind == "char":
                if value.lower() == back and not buffer:
                    return back
                if value in "0123456789xXabcdefABCDEF":
                    buffer += value
                else:
                    message = (
                        "请输入菜单编号" if values is None else "请输入菜单编号或版本号"
                    )
            if buffer:
                try:
                    number = (
                        int(buffer, 0)
                        if buffer.lower().startswith("0x")
                        else int(buffer)
                    )
                    if buffer.isdigit() and 1 <= number <= len(rows):
                        focus = number - 1
                    elif values and number in values:
                        focus = values.index(number)
                except ValueError:
                    pass


def flags_menu(g, filename, initial=None):
    ref, meta = source_mesh(g)
    geoms, bones, animations = import_glb(g)
    selected = detect_flags(ref) if ref else split_name(filename)[1]
    selected = set(selected)
    if initial is not None:
        selected = set(initial)
    locked = {}
    if bones:
        keep_legacy = (
            ref
            and ref.animated
            and not meta.get("animationClipsExported")
            and read_skeleton(ref.skeleton)["frame_count"]
        )
        locked["StripAnim"] = not (animations or keep_legacy)
    else:
        locked["StripAnim"] = False
    locked["StripGeo"] = not bool(geoms)
    if not geoms:
        locked.update(
            {
                f: False
                for f in (
                    "CompOcc",
                    "ZipPos",
                    "ZipUvs",
                    "StripUv13",
                    "StripNorm",
                    "ForceIdx32",
                )
            }
        )
    if not bones or locked["StripAnim"]:
        locked["UncompAnim"] = False
    if any(
        len(x["POSITION"]) > 65536 or max(x["indices"], default=0) > 65535
        for x in geoms
    ):
        locked["ForceIdx32"] = True
    for f in REFERENCE_ONLY:
        if ref and f in detect_flags(ref):
            locked[f] = True
    for f, v in locked.items():
        if v:
            selected.add(f)
        else:
            selected.discard(f)
    unavailable = (REFERENCE_ONLY | {"CompOcc"}) - (detect_flags(ref) if ref else set())
    for f in unavailable:
        selected.discard(f)
    focus = next(
        (
            i
            for i, f in enumerate(FLAG_ORDER)
            if f not in locked and f not in unavailable
        ),
        0,
    )
    buffer, message, error = "", "", False

    def toggle(answer):
        nonlocal focus, message, error
        try:
            items = [int(x) for x in re.split(r"[ ,，]+", answer.strip())]
        except ValueError:
            message, error = "请输入菜单编号；只有 y 确认后进入版本选择", True
            return
        notices = []
        error = False
        for i in items:
            if not 1 <= i <= len(FLAG_ORDER):
                notices.append(f"无效编号：{i}")
                error = True
                continue
            focus = i - 1
            f = FLAG_ORDER[focus]
            if f in locked or f in unavailable:
                notices.append(f"{f} 已锁定，不能修改")
                error = True
            else:
                if f in selected:
                    selected.remove(f)
                else:
                    selected.add(f)
                notices.append(f'{f} {"已选择" if f in selected else "已取消"}')
        message = "；".join(notices)

    with _MenuTerminal() as ui:
        while True:
            rows = []
            for f in FLAG_ORDER:
                status = (
                    "来源保留 · 锁定"
                    if f in locked and f in REFERENCE_ONLY
                    else (
                        "结构固定 · 锁定"
                        if f in locked
                        else (
                            "无来源 · 锁定"
                            if f in unavailable
                            else "已选择" if f in selected else "未选择"
                        )
                    )
                )
                rows.append(
                    {
                        "label": f,
                        "selected": f in selected,
                        "locked": f in locked or f in unavailable,
                        "status": status,
                        "description": f"{f}：{FLAG_DESCRIPTIONS[f]}",
                    }
                )
            ui.draw(
                "GLB → mesh  /  参数多选",
                rows,
                focus,
                [
                    "编号 + 空格/回车切换  ·  ↑↓定位，空格切换",
                    "回车只刷新  ·  y 确认  ·  白色未选 / 绿色已选 / 红色锁定",
                ],
                message,
                error,
                buffer,
            )
            event = ui.read_event()
            kind, value = event
            old_focus = focus
            focus, clicked = ui.navigate(event, focus, len(rows))
            if (
                kind in ("up", "down", "home", "end", "page_up", "page_down")
                or clicked
                or focus != old_focus
            ):
                buffer = ""
            if clicked:
                toggle(str(focus + 1))
            if kind == "line":
                if value.strip().lower() == "y":
                    return selected
                if value.strip():
                    toggle(value)
                elif value and " " in value:
                    toggle(str(focus + 1))
                else:
                    message, error = "状态已刷新；输入 y 才会进入版本选择", False
            elif kind == "char" and value.lower() == "y":
                if buffer:
                    toggle(buffer)
                    buffer = ""
                    if error:
                        continue
                return selected
            elif kind in ("enter", "space"):
                if buffer:
                    toggle(buffer)
                    buffer = ""
                elif kind == "space":
                    toggle(str(focus + 1))
                else:
                    message, error = "状态已刷新；按 y 才会进入版本选择", False
            elif kind == "backspace":
                buffer = buffer[:-1]
            elif kind == "char":
                if value.isdigit() or value == ",":
                    buffer += value
                    if buffer.isdigit() and 1 <= int(buffer) <= len(rows):
                        focus = int(buffer) - 1
                else:
                    message, error = "请输入编号，用空格切换；按 y 确认", True
            elif kind == "escape":
                buffer = ""
                message, error = "已清空输入；按 y 确认参数", False


def version_menu(flags):
    notice = ""
    while True:
        labels, descriptions = [], []
        for i, v in enumerate(VERSIONS, 1):
            note = (
                "（旧布局，32位索引，不支持压缩属性）"
                if v <= 28
                else (
                    "（32位索引）"
                    if v == 29
                    else "（APK全部样本版本）" if v == 32 else ""
                )
            )
            labels.append(f"0x{v:02X} / {v}")
            descriptions.append(f"0x{v:02X}：{note or '支持外层LZ4与可选属性'}")
        a = _menu_choice(
            "GLB → mesh  /  版本单选",
            labels,
            descriptions,
            7,
            "b",
            list(VERSIONS),
            notice,
        )
        if a.lower() == "b":
            return None
        try:
            v = (
                32
                if not a
                else (
                    VERSIONS[int(a) - 1]
                    if a.isdigit() and 1 <= int(a) <= 8
                    else int(a, 0)
                )
            )
        except (ValueError, IndexError):
            notice = "无效版本，请输入菜单编号或 0x19–0x20"
            continue
        if v not in VERSIONS:
            notice = "版本不受该 so 正常支持，请选择 0x19–0x20"
            continue
        if v < 29 and flags & {"ZipPos", "ZipUvs", "StripNorm"}:
            notice = "请先返回参数表取消 ZipPos / ZipUvs / StripNorm"
            continue
        return v


def interactive_menu():
    notice = ""
    while True:
        choice = _menu_choice(
            "转换菜单  /  libBootloader.so 365939",
            ["glb转mesh", "mesh转glb"],
            [
                "GLB → mesh：选择参数，确认后选择目标版本",
                "mesh → GLB：自动识别版本与数据特征",
            ],
            message=notice,
        ).lower()
        if choice in ("q", "quit", "退出"):
            return
        if choice not in ("1", "2"):
            notice = "请输入 1 或 2，输入 q 退出"
            continue
        notice = ""
        try:
            print(
                _menu_color(
                    f"\n  当前任务：{'GLB → mesh' if choice == '1' else 'mesh → GLB'}",
                    bold=True,
                )
            )
            path = Path(input("  文件路径 › ").strip().strip("\"'"))
            require(
                path.suffix.lower() == (".glb" if choice == "1" else ".mesh"),
                "文件类型与菜单不一致",
            )
            if choice == "1":
                g = Glb(path.read_bytes())
                fs = None
                while True:
                    fs = flags_menu(g, path.name, fs)
                    v = version_menu(fs)
                    if v is not None:
                        break
                convert_file(path, version=v, flags=fs, compress=v >= 27)
            else:
                convert_file(path)
        except (OSError, ValueError, KeyError, IndexError, struct.error) as e:
            notice = f"转换失败：{e}"
            print(_menu_color(notice, "red"))


def main(argv=None):
    p = argparse.ArgumentParser(description="Sky mesh↔GLB，无参数启动双选项中文菜单")
    p.add_argument("input", nargs="?")
    p.add_argument("-o", "--output")
    p.add_argument("--version", type=lambda x: int(x, 0), choices=VERSIONS, default=32)
    p.add_argument(
        "--flags", help="逗号分隔参数；空字符串表示取消所有可选参数；不指定则自动识别"
    )
    p.add_argument("--uncompressed", action="store_true")
    p.add_argument("--fps", type=int, help="动画采样帧率1—240；默认沿用sky_fps，否则30")
    p.add_argument("--inspect", action="store_true", help="检查mesh结构，不生成文件")
    a = p.parse_args(argv)
    try:
        if not a.input:
            interactive_menu()
            return 0
        if a.inspect:
            m = read_mesh(Path(a.input).read_bytes(), a.input)
            print(
                json.dumps(
                    {
                        "version": m.version,
                        "name": m.name,
                        "compressed": bool(m.compressed),
                        "animated": bool(m.animated),
                        "extraGeometry": bool(m.extra),
                        "extraGeometryCounts": (
                            dict(
                                zip(
                                    ("vertices", "indices"),
                                    struct.unpack_from("<2I", m.extra_data),
                                )
                            )
                            if m.extra
                            else None
                        ),
                        "flags": sorted(detect_flags(m)),
                        "filenameFlags": sorted(m.filename_flags),
                        "lods": [
                            {
                                "vertices": l.vertices,
                                "triangles": l.indices_count // 3,
                                "idx32": bool(l.idx32),
                                "counts": l.counts,
                                "zipLengths": l.zipped,
                            }
                            for l in m.lods
                        ],
                        "warnings": m.warnings,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        fs = (
            None
            if a.flags is None
            else {s.strip().lstrip("_") for s in a.flags.split(",") if s.strip()}
        )
        if a.fps is not None:
            require(1 <= a.fps <= 240, "动画帧率必须在1—240之间")
        convert_file(a.input, a.output, a.version, fs, not a.uncompressed, a.fps)
        return 0
    except (OSError, ValueError, KeyError, IndexError, struct.error) as e:
        print("错误:", e, file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\n已退出")
        return 0


# 动画实现：与已验证的动画模块保持一致。
"""Sky skeleton animation format, verified against ARM64 readers E622A4/E62424.
TRS order on disk: scale vec3, quaternion xyzw, translation vec3.
"""

c = sys.modules[__name__]


@dataclass
class Clip:
    start: int
    end: int
    flags: int
    bounds: tuple
    quant_min: tuple
    quant_extent: tuple
    mask: bytes
    tracks: dict
    name: str = ""


@dataclass
class Animation:
    fps: int
    frame_count: int
    bind: list
    clips: list
    compression: int
    raw_payload: bytes = b""


def unit_quaternion(row):
    length = math.sqrt(sum(x * x for x in row))
    c.require(length > 1e-12, "零长度动画四元数")
    return tuple(x / length for x in row)


def read_trs(r):
    scale = r.unpack("3f")
    rotation = r.unpack("4f")
    translation = r.unpack("3f")
    c.require(
        all(math.isfinite(x) for x in scale + rotation + translation),
        "动画TRS含非有限数",
    )
    return {
        "scale": scale,
        "rotation": unit_quaternion(rotation),
        "translation": translation,
    }


def read_animation(skeleton):
    s = c.read_skeleton(skeleton)
    count = s["frame_count"]
    fps = s["fps"]
    n = len(s["bones"])
    comp = s["compression"]
    v = s["version"]
    if not count:
        c.require(not s["tail"], "StripAnim骨骼资源有意外尾部")
        return Animation(fps, 0, [], [], comp)
    c.require(comp in (0, 1, 2) and fps > 0, "无效动画压缩模式/帧率")
    r = c.Reader(s["tail"], "animation outer")
    outer_bind = [read_trs(r) for _ in range(n)]
    if comp:
        cs = r.u32()
        us = r.u32() if v >= 9 else 0x300000
        # v7/8's LZ4 output length is not stored; cannot use it as exact length.
        c.require(v >= 9, "骨骼动画v7/8的无长度LZ4尚不支持")
        payload = c.lz4_decode(r.take(cs), us)
        c.require(r.p == len(r.data), "动画压缩块尾部有多余数据")
    else:
        payload = r.take(len(r.data) - r.p)
    r = c.Reader(payload, "animation payload")
    nc, np, nq, nt = r.unpack("4I")
    ntq = r.u32() if v >= 10 else 0
    nm = r.u32()
    c.require(nc <= 100000 and max(np, nq, nt, ntq, nm) <= 100000000, "动画池数量超限")
    bind = [read_trs(r) for _ in range(n)]
    clips = []
    used = [0, 0, 0, 0, 0]
    paths = ("scale", "rotation", "translation")
    for ci in range(nc):
        start, end, flags = r.unpack("3I")
        c.require(start <= end < count, "动画片段帧范围无效")
        bounds = r.unpack("6f") if v > 8 else (0.0,) * 6
        qmin = r.unpack("3f") if v >= 11 else (-20.0,) * 3
        qext = r.unpack("3f") if v >= 11 else (40.0,) * 3
        mask = r.take(n)
        c.require(all(x < 64 for x in mask), "骨骼通道掩码含未知位")
        c.require(
            all(
                not (x & (1 << j) and x & (1 << (j + 3)))
                for j in range(3)
                for x in mask
            ),
            "骨骼通道同时标成常量和逐帧",
        )
        tracks = {}

        def read_value(path):
            if path == "scale":
                used[0] += 1
                return r.unpack("3f")
            if path == "rotation":
                used[1] += 1
                return (
                    tuple(x * (2.0 / 65535.0) - 1.0 for x in r.unpack("4H"))
                    if comp == 2
                    else r.unpack("4f")
                )
            if comp == 2 and flags & 1:
                used[3] += 1
                return tuple(
                    qmin[j] + x / 65535.0 * qext[j]
                    for j, x in enumerate(r.unpack("3H"))
                )
            used[2] += 1
            return r.unpack("3f")

        for j, path in enumerate(paths):
            for bi, m in enumerate(mask):
                if m & (1 << (j + 3)):
                    tracks[(bi, path)] = [read_value(path)]
                    used[4] += 1
                elif m & (1 << j):
                    tracks[(bi, path)] = []
                    used[4] += 1
        for frame in range(end - start + 1):
            for j, path in enumerate(paths):
                for bi, m in enumerate(mask):
                    if m & (1 << j):
                        tracks[(bi, path)].append(read_value(path))
        for (bi, path), rows in tracks.items():
            c.require(
                all(math.isfinite(x) for row in rows for x in row), "动画通道含非有限数"
            )
            if path == "rotation":
                tracks[(bi, path)] = [unit_quaternion(row) for row in rows]
        clips.append(
            Clip(
                start,
                end,
                flags,
                bounds,
                qmin,
                qext,
                mask,
                tracks,
                f"frames_{start}_{end}",
            )
        )
    c.require(r.p == len(payload), f"动画payload剩余{len(payload)-r.p}字节")
    c.require(
        used == [np, nq, nt, ntq, nm],
        f"动画池声明与实际不一致: {used} != {[np,nq,nt,ntq,nm]}",
    )
    return Animation(fps, count, bind, clips, comp, payload)


def add_glb_animations(builder, animation, bone_offset):
    doc = builder.doc
    doc["animations"] = []
    for clip in animation.clips:
        duration = (clip.end - clip.start) / animation.fps
        data = {
            "name": clip.name,
            "samplers": [],
            "channels": [],
            "extras": {
                "sky_start": clip.start,
                "sky_end": clip.end,
                "sky_fps": animation.fps,
                "sky_flags": clip.flags,
            },
        }
        for (bi, path), rows in sorted(clip.tracks.items()):
            if len(rows) == 1:
                times = [0.0, duration] if duration else [0.0]
                values = [rows[0]] * len(times)
            else:
                times = [i / animation.fps for i in range(len(rows))]
                values = rows
            if path == "rotation":
                values = [unit_quaternion(row) for row in values]
                for j in range(1, len(values)):
                    if sum(x * y for x, y in zip(values[j - 1], values[j])) < 0:
                        values[j] = tuple(-x for x in values[j])
            time_acc = builder.accessor([(t,) for t in times], "SCALAR", bounds=True)
            val_acc = builder.accessor(values, "VEC4" if path == "rotation" else "VEC3")
            si = len(data["samplers"])
            data["samplers"].append(
                {"input": time_acc, "output": val_acc, "interpolation": "LINEAR"}
            )
            data["channels"].append(
                {"sampler": si, "target": {"node": bone_offset + bi, "path": path}}
            )
        # A static clip still needs one channel in glTF.
        if not data["channels"] and animation.bind:
            path = "translation"
            values = [animation.bind[0][path]] * (2 if duration else 1)
            times = [0.0, duration] if duration else [0.0]
            data["samplers"].append(
                {
                    "input": builder.accessor(
                        [(t,) for t in times], "SCALAR", bounds=True
                    ),
                    "output": builder.accessor(values, "VEC3"),
                }
            )
            data["channels"].append(
                {"sampler": 0, "target": {"node": bone_offset, "path": path}}
            )
        doc["animations"].append(data)


def pack_trs(trs):
    return struct.pack(
        "<10f", *trs["scale"], *unit_quaternion(trs["rotation"]), *trs["translation"]
    )


def encode_animation(skeleton, animation, compression=2):
    """Write complete v11 skeleton with either float/raw or Q16/LZ4 tracks."""
    s = c.read_skeleton(skeleton)
    bones = s["bones"]
    n = len(bones)
    c.require(compression in (0, 1, 2), "不支持动画压缩模式")
    if not animation.clips:
        return c.new_skeleton(s["name"], bones)
    c.require(
        len(animation.bind) == n and 1 <= animation.fps <= 240, "动画bind/fps无效"
    )
    bodies = []
    counts = [0, 0, 0, 0, 0]
    for clip in animation.clips:
        nf = clip.end - clip.start + 1
        c.require(nf > 0, "空动画片段")
        mask = [0] * n
        for (bi, path), rows in clip.tracks.items():
            c.require(
                0 <= bi < n and path in ("scale", "rotation", "translation"),
                "动画通道类型/骨骼索引错误",
            )
            c.require(len(rows) in (1, nf), "动画行数与片段帧数不符")
            j = ("scale", "rotation", "translation").index(path)
            mask[bi] |= 1 << (j + 3 if len(rows) == 1 else j)
        translations = [
            row
            for (bi, path), rows in clip.tracks.items()
            if path == "translation"
            for row in rows
        ]
        qmin = (
            tuple(min(row[j] for row in translations) for j in range(3))
            if translations
            else (0.0,) * 3
        )
        qext = (
            tuple(max(row[j] for row in translations) - qmin[j] for j in range(3))
            if translations
            else (0.0,) * 3
        )
        flags = (clip.flags | 1) if compression == 2 else (clip.flags & ~1)
        body = bytearray(
            struct.pack(
                "<3I12f", clip.start, clip.end, flags, *clip.bounds, *qmin, *qext
            )
            + bytes(mask)
        )

        def write_value(path, row):
            c.require(all(math.isfinite(x) for x in row), "动画数据非有限")
            if path == "scale":
                counts[0] += 1
                body.extend(struct.pack("<3f", *row))
            elif path == "rotation":
                counts[1] += 1
                row = unit_quaternion(row)
                if compression == 2:
                    body.extend(
                        struct.pack("<4H", *(c.quantize(x, -1, 1, 65535) for x in row))
                    )
                else:
                    body.extend(struct.pack("<4f", *row))
            elif compression == 2:
                counts[3] += 1
                body.extend(
                    struct.pack(
                        "<3H",
                        *(
                            c.quantize(row[j], qmin[j], qmin[j] + qext[j], 65535)
                            for j in range(3)
                        ),
                    )
                )
            else:
                counts[2] += 1
                body.extend(struct.pack("<3f", *row))

        for j, path in enumerate(("scale", "rotation", "translation")):
            for bi, m in enumerate(mask):
                if m & (1 << (j + 3)):
                    write_value(path, clip.tracks[(bi, path)][0])
                    counts[4] += 1
                elif m & (1 << j):
                    counts[4] += 1
        for fi in range(nf):
            for j, path in enumerate(("scale", "rotation", "translation")):
                for bi, m in enumerate(mask):
                    if m & (1 << j):
                        write_value(path, clip.tracks[(bi, path)][fi])
        bodies.append(body)
    payload = (
        struct.pack("<6I", len(bodies), *counts)
        + b"".join(pack_trs(t) for t in animation.bind)
        + b"".join(bodies)
    )
    frame_count = max(clip.end for clip in animation.clips) + 1
    out = bytearray(
        struct.pack("<I", 11)
        + c.fixed_name(s["name"])
        + struct.pack(
            "<3IBI",
            n,
            frame_count,
            animation.fps,
            compression,
            sum(len(b["name"].encode()) + 1 for b in bones),
        )
    )
    for b in bones:
        out.extend(
            c.fixed_name(b["name"])
            + struct.pack("<16fI", *c.transpose(b["matrix"]), b["parent"] + 1)
        )
    out.extend(b"".join(pack_trs(t) for t in animation.bind))
    if compression:
        block = c.lz4_encode(payload)
        out.extend(struct.pack("<2I", len(block), len(payload)) + block)
    else:
        out.extend(payload)
    result = bytes(out)
    read_animation(result)
    return result


def decompose_trs(matrix):
    c.require(
        len(matrix) == 16 and all(math.isfinite(x) for x in matrix), "无效TRS矩阵"
    )
    c.require(
        max(abs(matrix[i] - v) for i, v in zip((12, 13, 14, 15), (0.0, 0.0, 0.0, 1.0)))
        < 1e-5,
        "非仿射骨骼矩阵",
    )
    columns = [tuple(matrix[i * 4 + j] for i in range(3)) for j in range(3)]
    scales = [math.sqrt(sum(x * x for x in col)) for col in columns]
    c.require(min(scales) > 1e-10, "零缩放骨骼无法分解")
    determinant = (
        matrix[0] * (matrix[5] * matrix[10] - matrix[6] * matrix[9])
        - matrix[1] * (matrix[4] * matrix[10] - matrix[6] * matrix[8])
        + matrix[2] * (matrix[4] * matrix[9] - matrix[5] * matrix[8])
    )
    if determinant < 0:
        scales[0] = -scales[0]
    cols = [tuple(x / scales[j] for x in col) for j, col in enumerate(columns)]
    c.require(
        max(
            abs(sum(x * y for x, y in zip(cols[j], cols[k])))
            for j, k in ((0, 1), (0, 2), (1, 2))
        )
        < 1e-4,
        "骨骼变换含剪切，不能编码成游戏TRS",
    )
    m = [[cols[j][i] for j in range(3)] for i in range(3)]
    trace = sum(m[i][i] for i in range(3))
    if trace > 0:
        t = math.sqrt(trace + 1) * 2
        q = (
            (m[2][1] - m[1][2]) / t,
            (m[0][2] - m[2][0]) / t,
            (m[1][0] - m[0][1]) / t,
            0.25 * t,
        )
    else:
        i = max(range(3), key=lambda j: m[j][j])
        j = (i + 1) % 3
        k = (i + 2) % 3
        t = math.sqrt(max(0.0, 1 + m[i][i] - m[j][j] - m[k][k])) * 2
        vals = [0.0] * 4
        vals[i] = 0.25 * t
        vals[j] = (m[j][i] + m[i][j]) / t
        vals[k] = (m[k][i] + m[i][k]) / t
        vals[3] = (m[k][j] - m[j][k]) / t
        q = tuple(vals)
    return {
        "scale": tuple(scales),
        "rotation": unit_quaternion(q),
        "translation": (matrix[3], matrix[7], matrix[11]),
    }


def slerp(a, b, t):
    a = unit_quaternion(a)
    b = unit_quaternion(b)
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0:
        b = tuple(-x for x in b)
        dot = -dot
    if dot > 0.9995:
        return unit_quaternion(tuple(x + (y - x) * t for x, y in zip(a, b)))
    theta = math.acos(min(1.0, dot))
    den = math.sin(theta)
    return tuple(
        (x * math.sin((1 - t) * theta) + y * math.sin(t * theta)) / den
        for x, y in zip(a, b)
    )


def sampler_value(times, values, mode, path, t):
    import bisect

    n = len(times)
    c.require(mode in ("LINEAR", "STEP", "CUBICSPLINE"), f"未知动画插值: {mode}")
    c.require(
        n > 0 and all(times[j] < times[j + 1] for j in range(n - 1)),
        "动画时间必须严格递增",
    )
    expected = n * 3 if mode == "CUBICSPLINE" else n
    c.require(len(values) == expected, "动画输入输出数量不符")

    def key(i):
        return values[i * 3 + 1] if mode == "CUBICSPLINE" else values[i]

    if t <= times[0]:
        return key(0)
    if t >= times[-1]:
        return key(n - 1)
    i = bisect.bisect_right(times, t) - 1
    dt = times[i + 1] - times[i]
    u = (t - times[i]) / dt
    a = key(i)
    b = key(i + 1)
    if mode == "STEP":
        return a
    if mode == "LINEAR":
        return (
            slerp(a, b, u)
            if path == "rotation"
            else tuple(x + (y - x) * u for x, y in zip(a, b))
        )
    c.require(mode == "CUBICSPLINE", f"未知动画插值: {mode}")
    h00 = 2 * u**3 - 3 * u * u + 1
    h10 = u**3 - 2 * u * u + u
    h01 = -2 * u**3 + 3 * u * u
    h11 = u**3 - u * u
    row = tuple(
        h00 * a[j]
        + h10 * dt * values[i * 3 + 2][j]
        + h01 * b[j]
        + h11 * dt * values[(i + 1) * 3][j]
        for j in range(len(a))
    )
    return unit_quaternion(row) if path == "rotation" else row


def from_glb(g, bones, fps=None):
    if fps is None:
        rates = {
            clip.get("extras", {}).get("sky_fps")
            for clip in g.doc.get("animations", [])
            if "sky_fps" in clip.get("extras", {})
        }
        c.require(len(rates) <= 1, "动画片段sky_fps不一致，请用--fps指定统一采样帧率")
        fps = next(iter(rates), 30)
    c.require(isinstance(fps, int) and 1 <= fps <= 240, "动画采样帧率必须为1—240整数")
    nodes = g.doc.get("nodes", [])
    parents = {child: i for i, n in enumerate(nodes) for child in n.get("children", [])}
    js = [b["node"] for b in bones]
    c.require(js, "GLB动画没有可关联的骨架")
    relevant = set(js)
    for j in js:
        while j in parents:
            j = parents[j]
            relevant.add(j)

    def pose(channels, time):
        cache = {}

        def world(i):
            if i in cache:
                return cache[i]
            node = dict(nodes[i])
            for path, times, values, mode in channels.get(i, []):
                c.require("matrix" not in node, "带动画通道的节点必须使用TRS")
                node[path] = sampler_value(times, values, mode, path, time)
            local = c.node_matrix(node)
            cache[i] = c.mm(world(parents[i]), local) if i in parents else local
            return cache[i]

        result = []
        for i, j in enumerate(js):
            p = bones[i]["parent"]
            local = c.mm(c.inverse(world(js[p])), world(j)) if p >= 0 else world(j)
            result.append(decompose_trs(local))
        return result

    bind = pose({}, 0.0)
    clips = []
    cursor = 0
    for source in g.doc.get("animations", []):
        source_flags = source.get("extras", {}).get("sky_flags", 0)
        c.require(
            isinstance(source_flags, int) and 0 <= source_flags <= 0xFFFFFFFF,
            "动画sky_flags必须为u32",
        )
        channels = {}
        first = math.inf
        last = -math.inf
        for channel in source["channels"]:
            target = channel["target"]
            path = target["path"]
            j = target["node"]
            c.require(
                path in ("scale", "rotation", "translation"), "暂不支持morph权重动画"
            )
            c.require(0 <= j < len(nodes), "动画节点索引越界")
            c.require(j in relevant, "动画通道与目标骨架/几何无关联，不能静默丢弃")
            sp = source["samplers"][channel["sampler"]]
            times = [r[0] for r in g.accessor(sp["input"])]
            values = g.accessor(sp["output"])
            mode = sp.get("interpolation", "LINEAR")
            c.require(
                times and all(math.isfinite(x) and x >= 0 for x in times),
                "动画时间非法",
            )
            c.require(
                all(math.isfinite(x) for row in values for x in row), "动画通道非有限"
            )
            c.require(
                not any(p == path for p, *rest in channels.get(j, [])), "重复动画通道"
            )
            width = 4 if path == "rotation" else 3
            c.require(all(len(row) == width for row in values), "动画输出分量数量错误")
            # Validate every channel, including channels outside the selected skeleton.
            sampler_value(times, values, mode, path, times[0])
            channels.setdefault(j, []).append((path, times, values, mode))
            first = min(first, times[0])
            last = max(last, times[-1])
        c.require(channels, "空动画片段")
        duration = last - first
        frame_count = max(1, int(math.ceil(duration * fps - 1e-6)) + 1)
        c.require(frame_count * len(bones) <= 20000000, "动画采样数量超限")
        frames = [
            pose(channels, min(last, first + i / fps)) for i in range(frame_count)
        ]
        tracks = {}
        for bi in range(len(bones)):
            for path in ("scale", "rotation", "translation"):
                rows = [p[bi][path] for p in frames]
                if path == "rotation":
                    for k in range(1, len(rows)):
                        if sum(x * y for x, y in zip(rows[k - 1], rows[k])) < 0:
                            rows[k] = tuple(-x for x in rows[k])
                constant = (
                    max(abs(x - y) for row in rows for x, y in zip(row, rows[0])) < 1e-7
                )
                if constant:
                    equal = (
                        max(abs(x - y) for x, y in zip(rows[0], bind[bi][path])) < 1e-7
                    )
                    if path == "rotation":
                        equal = (
                            equal
                            or max(abs(x + y) for x, y in zip(rows[0], bind[bi][path]))
                            < 1e-7
                        )
                    if not equal:
                        tracks[(bi, path)] = [rows[0]]
                else:
                    tracks[(bi, path)] = rows
        clips.append(
            Clip(
                cursor,
                cursor + frame_count - 1,
                source_flags,
                (0.0,) * 6,
                (0.0,) * 3,
                (0.0,) * 3,
                b"",
                tracks,
                source.get("name", f"animation_{len(clips)}"),
            )
        )
        cursor += frame_count
    return Animation(fps, cursor, bind, clips, 2)


def fingerprint(g):
    payload = {
        "animations": g.doc.get("animations", []),
        "access": {},
        "nodes": [
            {
                k: v
                for k, v in node.items()
                if k in ("matrix", "translation", "rotation", "scale", "children")
            }
            for node in g.doc.get("nodes", [])
        ],
    }
    for clip in payload["animations"]:
        for sampler in clip["samplers"]:
            for k in ("input", "output"):
                i = sampler[k]
                payload["access"][str(i)] = g.accessor(i)
    return c.digest(
        c.json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


_sky_animation = SimpleNamespace(
    Clip=Clip,
    Animation=Animation,
    unit_quaternion=unit_quaternion,
    read_trs=read_trs,
    read_animation=read_animation,
    add_glb_animations=add_glb_animations,
    pack_trs=pack_trs,
    encode_animation=encode_animation,
    decompose_trs=decompose_trs,
    slerp=slerp,
    sampler_value=sampler_value,
    from_glb=from_glb,
    fingerprint=fingerprint,
)

if __name__ == "__main__":
    sys.exit(main())
