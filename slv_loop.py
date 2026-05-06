#!/usr/bin/env python3
"""
SLV Multi-Use Path Loop (v6 — 8 segments)
Proposed active-transportation loop, San Lorenzo Valley, Santa Cruz County, CA
All distances in miles.

8 segments:
  1  Graham Hill Rd (RAIL_END → Hwy 9 / Felton Empire, via Covered Bridge Park)
  2  Hwy 9 west-side path N — Caltrans 05-1M400 (→ SLV High School entrance)
  3  Hwy 9 west-side path continuing N (→ Highlands County Park entrance)
  4  Highlands County Park (park path → river bank)
  5  Proposed pedestrian bridge (2-pt straight line across San Lorenzo River)
  6  Maple Ave (bridge east landing → Glen Arbor Rd)
  7  Quail Hollow Rd (Glen Arbor Rd east → Olympia Station Rd)
  8  Historic SP Olympia Branch rail trail (→ Graham Hill Rd crossing)

Outputs:
  slv_loop_map.html   — interactive folium map
  slv_loop.geojson    — GeoJSON (all segments + destinations)
"""

import json, math, warnings
import osmnx as ox
import networkx as nx
import folium
import pandas as pd
import requests as _req
from folium.plugins import Fullscreen, MeasureControl

warnings.filterwarnings("ignore")
ox.settings.log_console = False
ox.settings.use_cache = True

# ── Bounding box (osmnx 2.x: left, bottom, right, top) ───────────────────────
N, S, E, W = 37.097, 37.037, -122.040, -122.098
BBOX = (W, S, E, N)

# ── Road network ──────────────────────────────────────────────────────────────
print("Downloading road network …")
G = ox.graph_from_bbox(bbox=BBOX, network_type="all", retain_all=True)
Gu = ox.convert.to_undirected(G)
nodes_gdf, edges_gdf = ox.convert.graph_to_gdfs(Gu)
print(f"  {len(Gu.nodes):,} nodes  |  {len(Gu.edges):,} edges")

# ── Helpers ───────────────────────────────────────────────────────────────────

def snap(lat, lon):
    return ox.distance.nearest_nodes(Gu, X=lon, Y=lat)


def snap_hwy9(lat, lon):
    """Nearest node on Highway 9 (ref = 'CA 9')."""
    def is9(v):
        return ("CA 9" in v) if isinstance(v, list) else (v == "CA 9")
    mask = (edges_gdf
            .get("ref", pd.Series(dtype=object, index=edges_gdf.index))
            .map(is9, na_action="ignore").fillna(False))
    nids = (set(edges_gdf[mask].index.get_level_values("u")) |
            set(edges_gdf[mask].index.get_level_values("v")))
    sub = nodes_gdf[nodes_gdf.index.isin(nids)]
    d = (sub["y"] - lat) ** 2 + (sub["x"] - lon) ** 2
    return d.idxmin()


def nd(nid):
    """(lat, lon) tuple for an OSM node id."""
    r = nodes_gdf.loc[nid]
    return (r["y"], r["x"])


def edge_pts(u, v):
    edata = Gu.get_edge_data(u, v)
    if edata is None:
        return [nd(u), nd(v)]
    best = min(edata.values(), key=lambda d: d.get("length", float("inf")))
    if "geometry" in best:
        return [(y, x) for x, y in best["geometry"].coords]
    return [nd(u), nd(v)]


def stitch(pts, seg):
    if not pts:
        pts.extend(seg)
    elif seg:
        if pts[-1] == seg[0]:
            pts.extend(seg[1:])
        elif pts[-1] == seg[-1]:
            pts.extend(list(reversed(seg))[1:])
        else:
            pts.extend(seg)
    return pts


def dedupe(pts):
    """Remove consecutive duplicate coordinates."""
    out = []
    for p in pts:
        if not out or p != out[-1]:
            out.append(p)
    return out


def route_nodes(a, b):
    """Shortest-path coord list between two OSM node ids."""
    try:
        path = nx.shortest_path(Gu, a, b, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        print(f"    ⚠  no path {a}→{b}; straight line")
        return [nd(a), nd(b)]
    pts = []
    for u, v in zip(path[:-1], path[1:]):
        stitch(pts, edge_pts(u, v))
    return dedupe(pts)


def route_via(*latlon_pairs):
    """Route through a sequence of (lat, lon) waypoints snapped to nearest OSM node."""
    node_seq = [snap(la, lo) for la, lo in latlon_pairs]
    pts = []
    for a, b in zip(node_seq[:-1], node_seq[1:]):
        stitch(pts, route_nodes(a, b))
    return dedupe(pts)


def mi(coords):
    total = 0.0
    for (la1, lo1), (la2, lo2) in zip(coords[:-1], coords[1:]):
        r = math.radians
        a = (math.sin(r(la2 - la1) / 2) ** 2
             + math.cos(r(la1)) * math.cos(r(la2)) * math.sin(r(lo2 - lo1) / 2) ** 2)
        total += 6371 * 2 * math.asin(math.sqrt(max(0.0, a)))
    return total * 0.621371


def split_at_coord(route_pts, target):
    """Split route at point closest to target (lat, lon).
    Returns (segment_up_to_split_inclusive, segment_from_split_inclusive)."""
    la0, lo0 = target
    best = min(range(len(route_pts)),
               key=lambda i: (route_pts[i][0]-la0)**2 + (route_pts[i][1]-lo0)**2)
    return route_pts[:best+1], route_pts[best:]


# Pre-fetched OSRM foot route: RAIL_END → Covered Bridge Park area (610 m / 0.38 mi)
# Source: router.project-osrm.org/route/v1/foot engine=fossgis_osrm_foot
# From (37.048109,-122.064355) to (37.051622,-122.069384)
_OSRM_GH_B = [
    (37.048112,-122.064348),(37.048119,-122.064352),(37.048544,-122.064594),
    (37.048871,-122.064801),(37.049059,-122.064921),(37.049372,-122.065274),
    (37.049595,-122.065560),(37.049756,-122.065806),(37.050099,-122.066363),
    (37.050299,-122.066700),(37.050546,-122.067113),(37.050917,-122.067687),
    (37.051099,-122.067968),(37.051226,-122.068194),(37.051437,-122.068620),
    (37.051481,-122.068760),(37.051672,-122.069349),(37.051674,-122.069356),
]

# Pre-fetched OSRM foot route: Covered Bridge Park area → Hwy 9/Graham Hill Rd (401 m / 0.25 mi)
# From (37.051622,-122.069384) to (37.052947,-122.073244)
_OSRM_GH_A = [
    (37.051674,-122.069356),(37.051783,-122.069671),(37.051817,-122.069772),
    (37.051915,-122.070088),(37.051960,-122.070245),(37.052002,-122.070397),
    (37.052037,-122.070509),(37.052076,-122.070637),(37.052087,-122.070671),
    (37.052176,-122.070940),(37.052255,-122.071193),(37.052335,-122.071441),
    (37.052393,-122.071495),(37.052418,-122.071575),(37.052653,-122.072273),
    (37.052670,-122.072314),(37.052806,-122.072650),(37.052859,-122.072736),
    (37.052948,-122.072852),(37.053001,-122.072963),(37.053072,-122.073143),
    (37.053080,-122.073282),(37.052945,-122.073256),
]

# Combined Graham Hill Rd raw points (finalised after H9_TRANSITION is snapped)
_gh_raw = _OSRM_GH_B + _OSRM_GH_A[1:]

# Pre-fetched OSRM foot route: Highlands Park transition → park road end (396 m / 0.25 mi)
# From (37.079357,-122.083458) to (37.080787,-122.080915)
_OSRM_HI4 = [
    (37.079357,-122.083460),(37.079371,-122.083466),(37.079393,-122.083367),
    (37.079459,-122.083173),(37.079550,-122.083136),(37.079651,-122.083217),
    (37.079832,-122.083364),(37.079973,-122.083362),(37.080199,-122.083325),
    (37.080312,-122.083271),(37.080373,-122.083166),(37.080706,-122.082594),
    (37.080831,-122.082383),(37.080722,-122.082185),(37.080661,-122.082129),
    (37.080396,-122.081929),(37.080561,-122.081542),(37.080710,-122.081375),
    (37.080780,-122.081164),(37.080787,-122.080925),
]

# Pre-fetched OSRM car-mode route: BRIDGE_E → Olympia Station Rd area (4082 m / 2.54 mi)
# Source: router.project-osrm.org/route/v1/driving engine=fossgis_osrm_car
# From (37.081908,-122.07952) to (37.073676,-122.053642)
_OSRM69 = [
    (37.081913,-122.079520),(37.081911,-122.079009),(37.081919,-122.078093),
    (37.081464,-122.077975),(37.081237,-122.077904),(37.081125,-122.077869),
    (37.080454,-122.077662),(37.080243,-122.077615),(37.080084,-122.077601),
    (37.079924,-122.077592),(37.079749,-122.077610),(37.079601,-122.077639),
    (37.079382,-122.077706),(37.079196,-122.077799),(37.079055,-122.077643),
    (37.078983,-122.077470),(37.079039,-122.077393),(37.079216,-122.077164),
    (37.079436,-122.077033),(37.080401,-122.076472),(37.080666,-122.076400),
    (37.081299,-122.076395),(37.081704,-122.076392),(37.082126,-122.076357),
    (37.082348,-122.076290),(37.082545,-122.076158),(37.082653,-122.075982),
    (37.082682,-122.075850),(37.082649,-122.075721),(37.081778,-122.073813),
    (37.081601,-122.073478),(37.081439,-122.073258),(37.081348,-122.073070),
    (37.081289,-122.072886),(37.081284,-122.072626),(37.081387,-122.072146),
    (37.081477,-122.071820),(37.081565,-122.070614),(37.081621,-122.070166),
    (37.081694,-122.069927),(37.081817,-122.069760),(37.081935,-122.069660),
    (37.082092,-122.069623),(37.082215,-122.069619),(37.082523,-122.069610),
    (37.082702,-122.069523),(37.082986,-122.069122),(37.083157,-122.068865),
    (37.083349,-122.068628),(37.083505,-122.068435),(37.083893,-122.068073),
    (37.084091,-122.067923),(37.084191,-122.067893),(37.084298,-122.067894),
    (37.084633,-122.068151),(37.084738,-122.068239),(37.084919,-122.068288),
    (37.085069,-122.068312),(37.085340,-122.068141),(37.085569,-122.067870),
    (37.085758,-122.067523),(37.085837,-122.067360),(37.085867,-122.067169),
    (37.085858,-122.066939),(37.085784,-122.066659),(37.085583,-122.066047),
    (37.085220,-122.065198),(37.084951,-122.064831),(37.084764,-122.064645),
    (37.084533,-122.064557),(37.083273,-122.064303),(37.082966,-122.064246),
    (37.082620,-122.064098),(37.082331,-122.063959),(37.082032,-122.063779),
    (37.081670,-122.063532),(37.081347,-122.063137),(37.080782,-122.062825),
    (37.080459,-122.062625),(37.080135,-122.062276),(37.079869,-122.061721),
    (37.079708,-122.061244),(37.079481,-122.060823),(37.078676,-122.059621),
    (37.078393,-122.059147),(37.078199,-122.058899),(37.077947,-122.058659),
    (37.077143,-122.057985),(37.074555,-122.055753),(37.074530,-122.055733),
    (37.074476,-122.055647),(37.074312,-122.055283),(37.074283,-122.055214),
    (37.074077,-122.055422),(37.073925,-122.055563),(37.073757,-122.055066),
    (37.073580,-122.054541),(37.073531,-122.054235),(37.073540,-122.053984),
    (37.073662,-122.053741),(37.073675,-122.053644),(37.073675,-122.053642),
]


# ── Hwy 9 snap points ─────────────────────────────────────────────────────────
print("Snapping Hwy 9 waypoints …")
H9_TRANSITION = snap_hwy9(37.052990, -122.073289)
H9_SLVHS      = snap_hwy9(37.060,    -122.080)
H9_HI         = snap_hwy9(37.079357, -122.083458)  # Highlands Park entrance transition

# Graham Hill Rd seg 1: small southward offset (~8 m) represents south-side path;
# last point snaps exactly to H9_TRANSITION so seg 1 butts perfectly against seg 2.
_GH_S = 0.000075
gh_coords = (
    [_gh_raw[0]]
    + [(la - _GH_S, lo) for la, lo in _gh_raw[1:-1]]
    + [nd(H9_TRANSITION)]
)

for label, nid in [("  Hwy9/Transition", H9_TRANSITION),
                   ("  Hwy9/SLV HS",     H9_SLVHS),
                   ("  Hwy9/Highlands",  H9_HI)]:
    y, x = nd(nid)
    print(f"{label}: ({y:.5f}, {x:.5f})")

# ── Fixed coordinates ─────────────────────────────────────────────────────────
PARK       = (37.051935, -122.070662)   # Covered Bridge Park front entrance
BRIDGE_W   = (37.081497, -122.080086)   # Bridge west (further inland, park side)
BRIDGE_E   = (37.081890, -122.079521)   # Bridge east (Maple Ave, user-specified)
RAIL_START = (37.073669, -122.053605)   # Rail trail start (Olympia Station area)
RAIL_END   = (37.048138, -122.064368)   # Rail crosses Graham Hill Rd

# ── Highlands Park segment ────────────────────────────────────────────────────
# Uses pre-fetched OSRM foot route; append BRIDGE_W as the final river-bank point
print("\nBuilding Highlands Park segment …")
hi_path = _OSRM_HI4 + [BRIDGE_W]
print(f"  Park path: {len(hi_path)} pts, {mi(hi_path):.2f} mi")

# ── Rail trail (Seg 9) ───────────────────────────────────────────────────────
# Pre-fetched OSM geometry for the Olympia Branch (Santa Cruz, Big Trees & Pacific
# Railway). Eight connected OSM ways chained N→S from the Olympia Station area
# to the Graham Hill Rd crossing. Clipped at RAIL_START and RAIL_END.
# OSM way IDs in order: 10550465, 784102825, 592185114, 1426303967,
#                        592185115, 1465667097, 43036475, 43036476
print("\nBuilding rail trail geometry …")
_OLYMPIA_RAIL = [
    (37.075827,-122.051145),(37.075524,-122.051340),(37.075329,-122.051476),
    (37.075038,-122.051716),(37.074819,-122.051927),(37.074594,-122.052182),
    (37.074407,-122.052428),(37.074320,-122.052551),(37.074235,-122.052683),
    (37.074102,-122.052941),(37.074021,-122.053119),(37.073930,-122.053282),
    (37.073855,-122.053392),(37.073675,-122.053644),(37.073482,-122.053844),
    (37.073313,-122.053995),(37.073082,-122.054172),(37.072784,-122.054367),
    (37.072685,-122.054426),(37.072570,-122.054481),(37.072346,-122.054570),
    (37.072233,-122.054606),(37.072120,-122.054643),(37.071891,-122.054694),
    (37.071384,-122.054794),(37.070700,-122.054935),(37.070438,-122.054995),
    (37.070253,-122.055068),(37.070073,-122.055153),(37.069940,-122.055226),
    (37.069809,-122.055316),(37.069571,-122.055515),(37.069123,-122.055898),
    (37.068849,-122.056135),(37.068642,-122.056297),(37.068536,-122.056364),
    (37.068421,-122.056427),(37.068311,-122.056472),(37.068200,-122.056508),
    (37.068094,-122.056528),(37.067997,-122.056541),(37.067838,-122.056544),
    (37.067696,-122.056531),(37.067549,-122.056506),(37.067288,-122.056426),
    (37.066761,-122.056232),(37.066521,-122.056137),(37.066289,-122.056049),
    (37.066137,-122.056006),(37.066034,-122.055984),(37.065929,-122.055969),
    (37.065807,-122.055965),(37.065740,-122.055962),(37.065674,-122.055964),
    (37.065329,-122.055984),(37.065017,-122.055996),(37.064931,-122.055997),
    (37.064839,-122.055990),(37.064707,-122.055972),(37.064540,-122.055945),
    (37.064515,-122.055939),(37.064372,-122.055903),(37.064289,-122.055881),
    (37.064206,-122.055851),(37.064038,-122.055773),(37.063879,-122.055702),
    (37.063564,-122.055544),(37.063361,-122.055457),(37.063264,-122.055427),
    (37.063167,-122.055401),(37.062939,-122.055362),(37.062758,-122.055351),
    (37.062593,-122.055364),(37.062433,-122.055392),(37.062287,-122.055433),
    (37.062072,-122.055512),(37.061873,-122.055600),(37.061737,-122.055679),
    (37.061571,-122.055792),(37.061407,-122.055914),(37.061183,-122.056130),
    (37.060960,-122.056416),(37.060603,-122.056923),(37.060516,-122.057086),
    (37.060411,-122.057251),(37.060289,-122.057403),(37.060167,-122.057537),
    (37.059952,-122.057743),(37.059495,-122.058145),(37.058982,-122.058599),
    (37.058832,-122.058729),(37.058405,-122.059046),(37.057977,-122.059322),
    (37.057546,-122.059597),(37.056852,-122.060051),(37.056297,-122.060381),
    (37.055888,-122.060543),(37.055572,-122.060646),(37.055138,-122.060742),
    (37.054909,-122.060801),(37.054700,-122.060880),(37.054471,-122.061012),
    (37.054360,-122.061092),(37.054253,-122.061178),(37.054082,-122.061349),
    (37.053632,-122.061829),(37.053148,-122.062274),(37.052251,-122.062800),
    (37.051449,-122.063266),(37.051211,-122.063396),(37.050973,-122.063526),
    (37.050139,-122.064002),(37.050043,-122.064062),(37.049950,-122.064111),
    (37.049884,-122.064143),(37.049720,-122.064222),(37.049566,-122.064286),
    (37.049352,-122.064336),(37.049216,-122.064363),(37.049094,-122.064382),
    (37.049007,-122.064383),(37.048119,-122.064352),(37.047923,-122.064345),
    (37.047566,-122.064334),(37.047194,-122.064320),(37.046593,-122.064302),
    (37.045984,-122.064286),(37.045283,-122.064270),(37.044893,-122.064254),
    (37.044796,-122.064241),(37.044700,-122.064222),(37.044643,-122.064211),
    (37.044541,-122.064182),
]
_, _rail_from_start = split_at_coord(_OLYMPIA_RAIL, RAIL_START)
rail_coords, _ = split_at_coord(_rail_from_start, RAIL_END)
print(f"  Rail trail: {len(rail_coords)} pts, {mi(rail_coords):.2f} mi")

# ── Segments 6-9 via OSRM car routing ────────────────────────────────────────
# Uses OSRM car-mode route (same engine as OSM "car directions") to avoid the
# bicycle router picking alternative paths through park trails or side roads.
# The full route is fetched once, then split at natural road-transition coordinates
# derived from analysis of the 102-point OSRM response.
print("\nRouting segments 6-9 via OSRM (car mode) …")
_osrm69 = _OSRM69
print(f"    OSRM: {len(_osrm69)} pts, {mi(_osrm69):.2f} mi")

# Split coordinates (taken from the known OSRM route geometry):
#   Seg 6 → 7: southernmost point of Maple Ave section (where Glen Arbor Rd starts E)
#   Seg 7 → 8: peak latitude = Glen Arbor Rd / Quail Hollow Rd junction (top of loop)
#   Seg 8 → 9: south end of Quail Hollow Rd / Olympia Station Rd transition
_SEG67 = (37.078983, -122.077470)
_SEG78 = (37.085867, -122.067169)
_SEG89 = (37.073531, -122.054235)

seg6_coords, _r69 = split_at_coord(_osrm69, _SEG67)
seg7_coords, _r79 = split_at_coord(_r69,    _SEG78)
seg8_coords, seg9_coords = split_at_coord(_r79, _SEG89)
seg8_9_coords = seg8_coords + seg9_coords[1:]
# Merge Glen Arbor Rd + Quail Hollow Rd into one "Quail Hollow Rd" segment
seg7_8_coords = seg7_coords + seg8_9_coords[1:]

print("\nRouting road segments …")

# ── Compile all 8 segments ────────────────────────────────────────────────────
SEGS = [
    dict(n=1,  label="Graham Hill Rd",
         color="#C0392B", dash=None,
         desc=("Graham Hill Rd from the historic rail crossing northeast "
               "to the Hwy 9 / Felton Empire Rd intersection, "
               "passing Felton Covered Bridge Park"),
         coords=gh_coords),

    dict(n=2,  label="Hwy 9 — Caltrans 05-1M400 (in progress)",
         color="#E67E22", dash=None,
         desc=("Hwy 9 west-side shared-use path northbound — "
               "existing Caltrans project 05-1M400 "
               "(Felton Empire / Graham Hill Rd to SLV High School entrance)"),
         coords=route_nodes(H9_TRANSITION, H9_SLVHS)),

    dict(n=3,  label="Hwy 9 to Highlands Park",
         color="#F1C40F", dash=None,
         desc=("Hwy 9 west-side path continuing north from SLV High School "
               "to the Highlands County Park entrance"),
         coords=route_nodes(H9_SLVHS, H9_HI)),

    dict(n=4,  label="Highlands County Park — park path to river",
         color="#27AE60", dash=None,
         desc=("Highlands County Park: park path from the Hwy 9 entrance "
               "to the San Lorenzo River bank"),
         coords=hi_path),

    dict(n=5,  label="Proposed Pedestrian / Bicycle Bridge",
         color="#2980B9", dash="10 6",
         desc=("Proposed pedestrian and bicycle bridge across the San Lorenzo River "
               "(2-point straight-line placeholder)"),
         coords=[BRIDGE_W, BRIDGE_E]),

    dict(n=6,  label="Maple Ave and Glen Arbor Rd",
         color="#16A085", dash=None,
         desc=("Maple Ave north from the bridge east landing "
               "to the Glen Arbor Rd junction"),
         coords=seg6_coords),

    dict(n=7,  label="Quail Hollow Rd",
         color="#8E44AD", dash=None,
         desc=("Glen Arbor Rd east from Maple Ave, then Quail Hollow Rd south "
               "to the Olympia Station Rd / Olympia Watershed trail entrance"),
         coords=seg7_8_coords),

    dict(n=8,  label="Historic SP Olympia Branch (proposed rail trail)",
         color="#795548", dash="5 8",
         desc=("Historic Southern Pacific / Santa Cruz, Big Trees & Pacific Railway "
               "Olympia Branch alignment — proposed rail trail southwest "
               "to the Graham Hill Rd crossing"),
         coords=rail_coords),
]

# ── Print segment table ───────────────────────────────────────────────────────
total = sum(mi(s["coords"]) for s in SEGS)
print(f"\n{'#':>2}  {'Segment':<52}  mi")
print("─" * 62)
for s in SEGS:
    d = mi(s["coords"])
    print(f"{s['n']:2d}  {s['label']:<52}  {d:.2f}")
print("─" * 62)
print(f"{'Total':>56}  {total:.2f}")

# ── Destinations ──────────────────────────────────────────────────────────────
slvhs_y, slvhs_x = nd(H9_SLVHS)
DESTS = [
    dict(name="Felton Covered Bridge Park",
         lat=37.051935, lon=-122.070662, fcolor="blue",
         desc="Historic 1892 covered bridge over the San Lorenzo River. "
              "Key landmark on the Graham Hill Rd segment."),
    dict(name="SLV High School (Hwy 9 entrance)",
         lat=slvhs_y, lon=slvhs_x, fcolor="blue",
         desc="San Lorenzo Valley High School — Hwy 9 driveway entrance. "
              "End of Caltrans 05-1M400 project segment."),
    dict(name="Highlands County Park",
         lat=37.080500, lon=-122.082839, fcolor="blue",
         desc="Highlands County Park. Park path leads from "
              "the Hwy 9 entrance to the proposed river crossing."),
    dict(name="Quail Hollow Ranch County Park",
         lat=37.082358, lon=-122.063464, fcolor="blue",
         desc="Quail Hollow Ranch County Park — on-loop access from Quail Hollow Rd."),
    dict(name="Olympia Watershed (trail entrance)",
         lat=37.069449, lon=-122.055370, fcolor="blue",
         desc="Olympia Watershed trail entrance / start of the historic "
              "SP Olympia Branch rail trail segment."),
]

# ── GeoJSON ───────────────────────────────────────────────────────────────────
print("\nWriting GeoJSON …")
features = []
for s in SEGS:
    features.append({
        "type": "Feature",
        "geometry": {"type": "LineString",
                     "coordinates": [[lo, la] for la, lo in s["coords"]]},
        "properties": {"segment_number": s["n"], "label": s["label"],
                       "description": s["desc"], "color": s["color"],
                       "dashed": bool(s["dash"]),
                       "length_mi": round(mi(s["coords"]), 3)},
    })
for d in DESTS:
    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [d["lon"], d["lat"]]},
        "properties": {"name": d["name"], "description": d["desc"]},
    })
with open("slv_loop.geojson", "w") as fh:
    json.dump({"type": "FeatureCollection", "features": features}, fh, indent=2)
print("  → slv_loop.geojson")

# ── Folium map ────────────────────────────────────────────────────────────────
print("Building folium map …")
m = folium.Map(location=(37.068, -122.072), zoom_start=14,
               tiles="OpenStreetMap", control_scale=True)
for tile, attr, name in [
    ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
     "Esri", "ESRI Topo"),
    ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
     "Esri", "ESRI Satellite"),
]:
    folium.TileLayer(tile, attr=attr, name=name).add_to(m)
folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

for s in SEGS:
    seg_mi = mi(s["coords"])
    tip = (f"<b>Seg {s['n']}: {s['label']}</b><br>{s['desc']}<br>"
           f"<i>{seg_mi:.2f} mi</i>")
    kw = dict(locations=s["coords"], color=s["color"], weight=7, opacity=0.90,
              tooltip=folium.Tooltip(tip, sticky=True),
              popup=folium.Popup(tip, max_width=360))
    if s["dash"]:
        kw["dash_array"] = s["dash"]
    folium.PolyLine(**kw).add_to(m)
    if s["coords"]:
        mid = s["coords"][len(s["coords"]) // 2]
        folium.Marker(
            mid,
            icon=folium.DivIcon(
                html=(f'<div style="font-size:11px;font-weight:bold;color:#fff;'
                      f'background:{s["color"]};border-radius:50%;width:22px;'
                      f'height:22px;line-height:22px;text-align:center;'
                      f'border:2px solid #fff;box-shadow:1px 1px 3px rgba(0,0,0,.5);">'
                      f'{s["n"]}</div>'),
                icon_size=(22, 22), icon_anchor=(11, 11))).add_to(m)

for d in DESTS:
    folium.Marker(
        (d["lat"], d["lon"]),
        tooltip=f"<b>{d['name']}</b>",
        popup=folium.Popup(f"<b>{d['name']}</b><br>{d['desc']}", max_width=300),
        icon=folium.Icon(color=d["fcolor"], icon="star")).add_to(m)


def fetch_elevations(pts):
    """Batch-fetch elevations (meters) from Open-Elevation API."""
    locations = [{"latitude": la, "longitude": lo} for la, lo in pts]
    elevs = []
    for i in range(0, len(locations), 100):
        chunk = locations[i:i+100]
        try:
            r = _req.post("https://api.open-elevation.com/api/v1/lookup",
                          json={"locations": chunk}, timeout=30)
            for res in r.json()["results"]:
                elevs.append(res["elevation"])
        except Exception as e:
            print(f"  ⚠  elevation fetch error: {e}")
            elevs.extend([0] * len(chunk))
    return elevs


def smooth_list(arr, window=9):
    out = []
    for i in range(len(arr)):
        a = max(0, i - window // 2)
        b = min(len(arr), i + window // 2 + 1)
        out.append(sum(arr[a:b]) / (b - a))
    return out


def leg_row(s):
    bar = (f'width:30px;height:0;display:inline-block;margin-right:7px;'
           f'border-top:5px dashed {s["color"]};') if s["dash"] else (
           f'width:30px;height:5px;display:inline-block;'
           f'margin-right:7px;background:{s["color"]};')
    return (f'<div style="display:flex;align-items:center;margin:2px 0;">'
            f'<span style="{bar}"></span>'
            f'<span style="font-size:11px;">{s["n"]}. {s["label"]}</span></div>')


# ── Elevation profile data ────────────────────────────────────────────────────
print("\nBuilding elevation profile …")
all_pts = []
for s in SEGS:
    if not all_pts:
        all_pts.extend(s["coords"])
    elif s["coords"]:
        if s["coords"][0] == all_pts[-1]:
            all_pts.extend(s["coords"][1:])
        else:
            all_pts.extend(s["coords"])

step = max(1, len(all_pts) // 100)
sampled = all_pts[::step]
if sampled[-1] != all_pts[-1]:
    sampled.append(all_pts[-1])

cum_mi = [0.0]
for (la1, lo1), (la2, lo2) in zip(sampled[:-1], sampled[1:]):
    cum_mi.append(cum_mi[-1] + mi([(la1, lo1), (la2, lo2)]))

print(f"  Fetching elevations for {len(sampled)} sample points …")
elevs_m = fetch_elevations(sampled)
elevs_ft = [e * 3.28084 for e in elevs_m]

elevs_smooth = smooth_list(elevs_ft, window=9)
baseline = elevs_smooth[0]
elevs_rel = [round(e - baseline) for e in elevs_smooth]

elev_gain = round(sum(max(0, b - a) for a, b in zip(elevs_rel[:-1], elevs_rel[1:])))
elev_abs_min = round(min(elevs_ft))
elev_abs_max = round(max(elevs_ft))
print(f"  Elevation: {elev_abs_min}–{elev_abs_max} ft absolute, +{elev_gain} ft gain")

dist_js = "[" + ",".join(f"{d:.3f}" for d in cum_mi) + "]"
elev_js  = "[" + ",".join(str(e) for e in elevs_rel) + "]"

# ── Combined collapsible left sidebar (legend + elevation) ────────────────────
sidebar_html = f"""
<style>
  #slv-sidebar {{
    position:fixed;left:0;bottom:30px;z-index:9999;
    display:flex;align-items:flex-start;
    transition:transform 0.3s ease;
  }}
  #slv-panels {{
    display:flex;flex-direction:column;gap:8px;
    width:min(430px, calc(100vw - 28px));
  }}
  .slv-panel {{
    background:#fff;
    border:2px solid #444;border-left:none;border-radius:0 7px 7px 0;
    font-family:Arial,sans-serif;
    box-shadow:3px 3px 10px rgba(0,0,0,0.35);
  }}
  #slv-legend {{
    padding:12px 14px 10px;
    max-height:min(45vh,280px);overflow-y:auto;
  }}
  #slv-elev {{
    padding:8px 12px 8px;
  }}
  #slv-toggle {{
    flex-shrink:0;margin-top:20px;width:24px;height:48px;
    background:#fff;border:2px solid #444;border-left:none;
    border-radius:0 6px 6px 0;cursor:pointer;
    font-size:13px;line-height:1;padding:0;
    box-shadow:3px 3px 8px rgba(0,0,0,0.3);
  }}
  /* Mobile: hide elevation chart behind its own toggle to save space */
  @media (max-width:600px) {{
    #slv-legend {{ padding:8px 10px 8px; max-height:min(38vh,220px); }}
    #slv-elev   {{ padding:6px 10px 6px; }}
    #elev-title {{ font-size:11px; }}
    #elev-meta  {{ display:none; }}
  }}
</style>

<div id="slv-sidebar">
  <div id="slv-panels">
    <div id="slv-legend" class="slv-panel">
      <b style="font-size:13px;">SLV Multi-Use Path Loop Proposal</b><br>
      <span style="font-size:10px;color:#666;">
        San Lorenzo Valley, Santa Cruz County, CA &nbsp;·&nbsp; ~{total:.1f} mi
      </span>
      <hr style="margin:6px 0;border-color:#ccc;">
      {"".join(leg_row(s) for s in SEGS)}
      <hr style="margin:5px 0;border-color:#ccc;">
      <span style="font-size:10px;color:#888;">
        &#9733; Key destinations &nbsp;|&nbsp; - - proposed bridge &amp; rail trail
      </span>
    </div>
    <div id="slv-elev" class="slv-panel">
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;">
        <b id="elev-title" style="font-size:12px;">Elevation Profile</b>
        <span id="elev-meta" style="font-size:10px;color:#666;">
          {elev_abs_min}–{elev_abs_max} ft &nbsp;·&nbsp; +{elev_gain} ft gain
        </span>
      </div>
      <div style="position:relative;height:80px;">
        <canvas id="elevChart"></canvas>
      </div>
    </div>
  </div>
  <button id="slv-toggle">◀</button>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
(function() {{
  var sb  = document.getElementById('slv-sidebar');
  var pan = document.getElementById('slv-panels');
  var btn = document.getElementById('slv-toggle');

  function collapse() {{
    sb.style.transform = 'translateX(-' + pan.offsetWidth + 'px)';
    sb.dataset.c = '1';
    btn.textContent = '▶';
  }}
  function expand() {{
    sb.style.transform = '';
    sb.dataset.c = '';
    btn.textContent = '◀';
  }}

  // Auto-collapse on small screens
  if (window.innerWidth < 600) {{ collapse(); }}

  btn.addEventListener('click', function() {{
    sb.dataset.c === '1' ? expand() : collapse();
  }});

  var dists = {dist_js};
  var elevs = {elev_js};
  var ctx = document.getElementById('elevChart').getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: dists,
      datasets: [{{
        data: elevs,
        fill: true,
        backgroundColor: 'rgba(76,175,80,0.20)',
        borderColor: 'rgba(46,125,50,0.85)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.5
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          mode: 'index',
          intersect: false,
          callbacks: {{
            title: function(items) {{ return dists[items[0].dataIndex].toFixed(2) + ' mi'; }},
            label: function(item) {{
              var v = item.parsed.y;
              return (v >= 0 ? '+' : '') + v + ' ft';
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          ticks: {{
            maxTicksLimit: 6,
            font: {{ size: 9 }},
            callback: function(val, i) {{
              return typeof dists[i] === 'number' ? dists[i].toFixed(1) + ' mi' : '';
            }}
          }},
          grid: {{ color: 'rgba(0,0,0,0.06)' }}
        }},
        y: {{
          ticks: {{
            font: {{ size: 9 }},
            maxTicksLimit: 4,
            callback: function(v) {{ return (v >= 0 ? '+' : '') + v + ' ft'; }}
          }},
          grid: {{ color: 'rgba(0,0,0,0.06)' }}
        }}
      }}
    }}
  }});
}})();
</script>
"""
m.get_root().html.add_child(folium.Element(sidebar_html))

Fullscreen().add_to(m)
MeasureControl(position="topright", primary_length_unit="miles",
               secondary_length_unit="kilometers").add_to(m)
folium.LayerControl(position="topright").add_to(m)

m.save("slv_loop_map.html")
print("  → slv_loop_map.html")
print(f"\n✓  Done  —  {total:.1f} mi loop, 8 segments")
