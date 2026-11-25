"""공간데이터 처리: 전국 시군구 SHP(sig) → 서울 25개 자치구
산출물
  out/spatial/seoul_gu.shp/.shx/.dbf/.prj/.cpg  : 원본 좌표계(EPSG:5179) 그대로, 서울만 추출 (지오메트리 무손실)
  out/spatial/seoul_gu.geojson                  : WGS84(EPSG:4326) 변환 + 경량화(Douglas-Peucker 5m) — 대시보드 지도용
  out/spatial/seoul_gu_area.csv                 : SHP 폴리곤 면적(km²) vs 제공 면적표 비교
외부 GIS 라이브러리 없이 struct/numpy로 직접 읽고 씀.
사용: python process_spatial.py <sig.shp> <sig.shx> <sig.dbf> <seoul_district_area.csv> <out_dir>
"""
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SHP, SHX, DBF, AREA_CSV, OUT = sys.argv[1:6]
OUT = Path(OUT) / "spatial"
OUT.mkdir(parents=True, exist_ok=True)

PRJ_5179 = ('PROJCS["Korea_2000_Korea_Unified_Coordinate_System",GEOGCS["GCS_Korea_2000",'
            'DATUM["D_Korea_2000",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],'
            'UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
            'PARAMETER["False_Easting",1000000.0],PARAMETER["False_Northing",2000000.0],'
            'PARAMETER["Central_Meridian",127.5],PARAMETER["Scale_Factor",0.9996],'
            'PARAMETER["Latitude_Of_Origin",38.0],UNIT["Meter",1.0]]')


# ── DBF 읽기 ─────────────────────────────────────────────
def read_dbf(path, encoding="cp949"):
    b = open(path, "rb").read()
    nrec, hlen, rlen = struct.unpack("<IHH", b[4:12])
    fields, pos = [], 32
    while b[pos] != 0x0D:
        name = b[pos:pos + 11].split(b"\x00")[0].decode("ascii")
        ftype, flen, fdec = chr(b[pos + 11]), b[pos + 16], b[pos + 17]
        fields.append((name, ftype, flen, fdec))
        pos += 32
    rows = []
    for i in range(nrec):
        rec = b[hlen + i * rlen: hlen + (i + 1) * rlen]
        p, row = 1, {}
        for name, _, flen, _ in fields:
            row[name] = rec[p:p + flen].decode(encoding).strip()
            p += flen
        rows.append(row)
    return pd.DataFrame(rows), fields


# ── SHP 읽기 (Polygon, type 5) ───────────────────────────
def read_polygons(shp_path, shx_path):
    shp, shx = open(shp_path, "rb").read(), open(shx_path, "rb").read()
    assert struct.unpack("<i", shp[32:36])[0] == 5, "Polygon SHP만 지원"
    geoms = []
    for i in range((len(shx) - 100) // 8):
        off = struct.unpack(">i", shx[100 + i * 8: 104 + i * 8])[0] * 2
        c = off + 8
        npart, npt = struct.unpack("<ii", shp[c + 36: c + 44])
        parts = list(struct.unpack(f"<{npart}i", shp[c + 44: c + 44 + 4 * npart])) + [npt]
        pts = np.frombuffer(shp, dtype="<f8", count=npt * 2, offset=c + 44 + 4 * npart).reshape(-1, 2)
        geoms.append([pts[parts[k]:parts[k + 1]].copy() for k in range(npart)])
    return geoms


# ── SHP 쓰기 ─────────────────────────────────────────────
def write_shapefile(base, geoms, records, fields):
    def bbox(arrs):
        a = np.vstack(arrs)
        return a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()

    allb = bbox([r for g in geoms for r in g])
    recs, idx, offset = [], [], 50  # 16-bit word 단위
    for i, g in enumerate(geoms):
        npt = sum(len(r) for r in g)
        starts = np.cumsum([0] + [len(r) for r in g[:-1]])
        content = struct.pack("<i4dii", 5, *bbox(g), len(g), npt)
        content += struct.pack(f"<{len(g)}i", *starts) + np.vstack(g).astype("<f8").tobytes()
        recs.append(struct.pack(">ii", i + 1, len(content) // 2) + content)
        idx.append(struct.pack(">ii", offset, len(content) // 2))
        offset += 4 + len(content) // 2

    def header(length_words):
        return (struct.pack(">i", 9994) + b"\x00" * 20 + struct.pack(">i", length_words)
                + struct.pack("<ii", 1000, 5) + struct.pack("<4d", *allb) + struct.pack("<4d", 0, 0, 0, 0))

    body = b"".join(recs)
    open(f"{base}.shp", "wb").write(header(50 + len(body) // 2) + body)
    open(f"{base}.shx", "wb").write(header(50 + 4 * len(idx)) + b"".join(idx))

    # DBF (UTF-8, .cpg로 인코딩 명시)
    enc = [(n, t, max(fl, 40) if t == "C" else fl, d) for n, t, fl, d in fields]
    rlen = 1 + sum(f[2] for f in enc)
    hlen = 32 + 32 * len(enc) + 1
    h = struct.pack("<BBBBIHH20x", 3, 126, 11, 22, len(records), hlen, rlen)
    for n, t, fl, d in enc:
        h += n.encode("ascii").ljust(11, b"\x00") + t.encode() + b"\x00" * 4 + bytes([fl, d]) + b"\x00" * 14
    h += b"\x0D"
    body = b""
    for r in records:
        body += b" "
        for n, t, fl, d in enc:
            v = str(r[n]).encode("utf-8")
            assert len(v) <= fl
            body += v.ljust(fl, b" ")
    open(f"{base}.dbf", "wb").write(h + body + b"\x1A")
    open(f"{base}.prj", "w").write(PRJ_5179)
    open(f"{base}.cpg", "w").write("UTF-8")


# ── 좌표 변환: EPSG:5179(UTM-K, GRS80 TM) → WGS84 ───────────
A, F = 6378137.0, 1 / 298.257222101
E2 = F * (2 - F)
EP2 = E2 / (1 - E2)
K0, LAT0, LON0, FE, FN = 0.9996, math.radians(38.0), math.radians(127.5), 1_000_000.0, 2_000_000.0


def meridian_arc(phi):
    e4, e6 = E2 ** 2, E2 ** 3
    return A * ((1 - E2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi
                - (3 * E2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * np.sin(2 * phi)
                + (15 * e4 / 256 + 45 * e6 / 1024) * np.sin(4 * phi)
                - (35 * e6 / 3072) * np.sin(6 * phi))


M0 = meridian_arc(LAT0)


def utmk_to_wgs84(xy):
    x, y = xy[:, 0] - FE, xy[:, 1] - FN
    M = M0 + y / K0
    mu = M / (A * (1 - E2 / 4 - 3 * E2 ** 2 / 64 - 5 * E2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - E2)) / (1 + math.sqrt(1 - E2))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * np.sin(6 * mu) + (1097 * e1 ** 4 / 512) * np.sin(8 * mu))
    s, c, t = np.sin(phi1), np.cos(phi1), np.tan(phi1)
    C1, T1 = EP2 * c ** 2, t ** 2
    N1 = A / np.sqrt(1 - E2 * s ** 2)
    R1 = A * (1 - E2) / (1 - E2 * s ** 2) ** 1.5
    D = x / (N1 * K0)
    lat = phi1 - (N1 * t / R1) * (D ** 2 / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * EP2) * D ** 4 / 24
                                  + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * EP2 - 3 * C1 ** 2) * D ** 6 / 720)
    lon = LON0 + (D - (1 + 2 * T1 + C1) * D ** 3 / 6
                  + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * EP2 + 24 * T1 ** 2) * D ** 5 / 120) / c
    return np.column_stack([np.degrees(lon), np.degrees(lat)])


# ── 기하 유틸 ─────────────────────────────────────────────
def signed_area(r):  # 양수 = 반시계
    x, y = r[:, 0], r[:, 1]
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


def douglas_peucker(pts, tol):
    keep = np.zeros(len(pts), bool)
    keep[[0, -1]] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        s, e = stack.pop()
        if e <= s + 1:
            continue
        a, b = pts[s], pts[e]
        seg = b - a
        L = np.hypot(*seg)
        sub = pts[s + 1:e]
        d = (np.abs(seg[0] * (sub[:, 1] - a[1]) - seg[1] * (sub[:, 0] - a[0])) / L) if L > 0 \
            else np.hypot(*(sub - a).T)
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[s + 1 + k] = True
            stack += [(s, s + 1 + k), (s + 1 + k, e)]
    return pts[keep]


# ── 실행 ─────────────────────────────────────────────────
attrs, fields = read_dbf(DBF)
geoms = read_polygons(SHP, SHX)
assert len(attrs) == len(geoms)
seoul = attrs.index[attrs["SIG_CD"].str.startswith("11")].tolist()
print(f"전국 {len(attrs)}개 시군구 중 서울 {len(seoul)}개 추출")

sg = [geoms[i] for i in seoul]
sa = attrs.loc[seoul].reset_index(drop=True)
write_shapefile(OUT / "seoul_gu", sg, sa.to_dict("records"), fields)

areas, features, n_before, n_after = [], [], 0, 0
for (_, row), rings in zip(sa.iterrows(), sg):
    # SHP 규약: 시계방향(면적<0) = 외곽, 반시계 = 구멍
    area = -sum(signed_area(r) for r in rings) / 1e6
    areas.append(area)
    polys = []
    for r in rings:
        n_before += len(r)
        rs = douglas_peucker(r, 5.0)
        if len(rs) < 4:
            continue
        n_after += len(rs)
        ll = np.round(utmk_to_wgs84(rs), 6)
        if signed_area(r) < 0:       # 외곽 → GeoJSON은 반시계
            polys.append([ll[::-1].tolist()])
        elif polys:                  # 구멍 → 직전 외곽에 추가(시계)
            polys[-1].append(ll[::-1].tolist())
    geom = {"type": "Polygon", "coordinates": polys[0]} if len(polys) == 1 \
        else {"type": "MultiPolygon", "coordinates": polys}
    features.append({"type": "Feature",
                     "properties": {"SIG_CD": row["SIG_CD"], "SIG_KOR_NM": row["SIG_KOR_NM"],
                                    "SIG_ENG_NM": row["SIG_ENG_NM"], "AREA_KM2": round(area, 2)},
                     "geometry": geom})

with open(OUT / "seoul_gu.geojson", "w", encoding="utf-8") as f:
    json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False)
print(f"GeoJSON 꼭짓점 {n_before:,} → {n_after:,} (Douglas-Peucker 5m)")

ref = pd.read_csv(AREA_CSV, encoding="utf-8-sig").rename(columns={"자치구": "자치구명", "면적 (km²)": "면적_제공"})
cmp_ = pd.DataFrame({"자치구코드": sa["SIG_CD"], "자치구명": sa["SIG_KOR_NM"], "면적_SHP": np.round(areas, 2)})
cmp_ = cmp_.merge(ref[["자치구명", "면적_제공"]], on="자치구명", how="left")
cmp_["차이"] = (cmp_["면적_SHP"] - cmp_["면적_제공"]).round(2)
cmp_["차이율_%"] = (cmp_["차이"] / cmp_["면적_제공"] * 100).round(2)
cmp_.to_csv(OUT / "seoul_gu_area.csv", index=False, encoding="utf-8-sig")
print(cmp_.to_string(index=False))
