#!/usr/bin/env python3
"""
SLV Multi-Use Path Loop (v8 — 7 segments + standalone bridge + 3 proposed branches)
Proposed active-transportation loop, San Lorenzo Valley, Santa Cruz County, CA
All distances in miles.

7 main-loop segments:
  1  Graham Hill Rd (RAIL_END → Hwy 9 / Felton Empire, via Covered Bridge Park)
  2  Hwy 9 west-side path N — Caltrans 05-1M400 (→ SLV High School entrance)
  3  Hwy 9 west-side path continuing N (→ Highlands County Park entrance)
  4  Highlands County Park (park path → river bank)
  5  Maple Ave (bridge east landing → Glen Arbor Rd) — blue
  6  Quail Hollow Rd (Glen Arbor Rd east → Olympia Station Rd) — indigo
  7  Historic SP Olympia Branch rail trail (→ Graham Hill Rd crossing) — purple, solid

Standalone bridge (not part of the numbered segments, brown dashed):
  Proposed pedestrian/bicycle bridge (2-pt straight line across San Lorenzo River)

3 proposed branches (gray, dashed, toggleable "Branch proposals" layer):
  1  Hwy 9 Bike Lanes (Class II) — Graham Hill Rd → Glengarry Rd
  2  Glen Arbor Rd Bike Lanes (Class II) — corrected start point → Hwy 9 (Ben Lomond)
  3  Felton Empire Rd Multi-Use Path (Class I) — Hwy 9 → Fetherston Way

Also renders (toggleable, off by default): 2020 Census population density
choropleth, and 0.25/0.50/1.00 mi road-network catchment overlays. The
catchment overlays are recomputed from the road network below; their
shape/population depends on whether the "Branch proposals" layer is
toggled on or off.

Outputs:
  slv_loop_map.html   — interactive folium map
  slv_loop.geojson    — GeoJSON (all segments, bridge, branches + destinations)
"""

import json, math, os, warnings
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


def path_midpoint(coords):
    """Point at half the path's total length — unlike coords[len//2], this
    lands on the true geometric middle even for a 2-point straight line."""
    if not coords:
        return None
    if len(coords) == 1:
        return coords[0]
    half = mi(coords) / 2
    acc = 0.0
    for (la1, lo1), (la2, lo2) in zip(coords[:-1], coords[1:]):
        leg = mi([(la1, lo1), (la2, lo2)])
        if acc + leg >= half:
            frac = (half - acc) / leg if leg > 0 else 0.0
            return (la1 + (la2 - la1) * frac, lo1 + (lo2 - lo1) * frac)
        acc += leg
    return coords[-1]


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

    dict(n=5,  label="Maple Ave and Glen Arbor Rd",
         color="#2980B9", dash=None,
         desc=("Maple Ave north from the bridge east landing "
               "to the Glen Arbor Rd junction"),
         coords=seg6_coords),

    dict(n=6,  label="Quail Hollow Rd",
         color="#4B0082", dash=None,
         desc=("Glen Arbor Rd east from Maple Ave, then Quail Hollow Rd south "
               "to the Olympia Station Rd / Olympia Watershed trail entrance"),
         coords=seg7_8_coords),

    dict(n=7,  label="Historic SP Olympia Branch (proposed rail trail)",
         color="#8E44AD", dash=None,
         desc=("Historic Southern Pacific / Santa Cruz, Big Trees & Pacific Railway "
               "Olympia Branch alignment — proposed rail trail southwest "
               "to the Graham Hill Rd crossing"),
         coords=rail_coords),
]

# ── Proposed bridge (standalone — not part of the numbered main-loop segments)
BRIDGE = dict(n="B", label="Proposed Pedestrian / Bicycle Bridge",
              color="#795548", dash="10 6",
              desc=("Proposed pedestrian and bicycle bridge across the San Lorenzo River "
                    "(2-point straight-line placeholder)"),
              coords=[BRIDGE_W, BRIDGE_E])

# ── Proposed bikeway branch overlays ──────────────────────────────────────────
# Previously maintained only as hand-written Leaflet JS patched directly into
# slv_loop_map.html; ported here so the generator script is the single source
# of truth and these survive a regeneration.
BRANCH_DASH = "10 5"

# Branch 1: Hwy 9 bike lanes (Class II) — Graham Hill Rd → Glengarry Rd
_b1_raw = [
    (37.052938,-122.073255),(37.052829,-122.073241),(37.052675,-122.073219),(37.052476,-122.073214),(37.052103,-122.073236),(37.052008,-122.073243),(37.051758,-122.073251),(37.051670,-122.073253),(37.051302,-122.073269),(37.051204,-122.073271),(37.051147,-122.073272),(37.050881,-122.073280),(37.050834,-122.073282),(37.050695,-122.073287),(37.050515,-122.073293),(37.050456,-122.073295),(37.050403,-122.073297),(37.050327,-122.073299),(37.050032,-122.073309),(37.049842,-122.073314),(37.049805,-122.073317),(37.049675,-122.073322),(37.049606,-122.073323),(37.049566,-122.073324),(37.049512,-122.073325),(37.049099,-122.073341),(37.049080,-122.073342),(37.048984,-122.073345),(37.048643,-122.073354),(37.048507,-122.073359),(37.048300,-122.073366),(37.048102,-122.073372),(37.047861,-122.073379),(37.047850,-122.073379),(37.047842,-122.073379),(37.047471,-122.073388),(37.047249,-122.073393),(37.047042,-122.073399),(37.046818,-122.073406),(37.046658,-122.073410),(37.046411,-122.073424),(37.046022,-122.073445),(37.045980,-122.073446),(37.045729,-122.073455),(37.045411,-122.073435),(37.045285,-122.073425),(37.045151,-122.073387),(37.044935,-122.073314),(37.044698,-122.073222),(37.044492,-122.073144),(37.044077,-122.072992),(37.043894,-122.072916),(37.043812,-122.072887),(37.043563,-122.072821),(37.043320,-122.072791),(37.042318,-122.072705),(37.042226,-122.072698),(37.042122,-122.072683),(37.041992,-122.072653),(37.041858,-122.072622),(37.041714,-122.072556),(37.041118,-122.072204),(37.040835,-122.072038),(37.040734,-122.071962),(37.040634,-122.071874),(37.040491,-122.071732),(37.040375,-122.071584),(37.040063,-122.071080),(37.039653,-122.070344),(37.039529,-122.070167),(37.039189,-122.069733),(37.038957,-122.069402),(37.038833,-122.069174),(37.038743,-122.068933),(37.038505,-122.068111),(37.038450,-122.067950),(37.038391,-122.067795),(37.038325,-122.067654),(37.038160,-122.067376),(37.038057,-122.067236),(37.037607,-122.066662),(37.037318,-122.066373),(37.037093,-122.066149),(37.037033,-122.066089),(37.036776,-122.065839),(37.036644,-122.065710),(37.036341,-122.065520),(37.035962,-122.065295),(37.035791,-122.065170),(37.035571,-122.065042),(37.035457,-122.064883),(37.035294,-122.064708),(37.034936,-122.064325),(37.034894,-122.064295),(37.034751,-122.064198),(37.034602,-122.064122),(37.034379,-122.064039),(37.034113,-122.063939),(37.033955,-122.063817),(37.033798,-122.063673),(37.033663,-122.063516),(37.033602,-122.063436),(37.033542,-122.063359),(37.033480,-122.063274),(37.033114,-122.062873),(37.033036,-122.062819),(37.032958,-122.062782),(37.032882,-122.062767),(37.032810,-122.062764),(37.032774,-122.062770),(37.032735,-122.062777),(37.032685,-122.062795),(37.032640,-122.062820),(37.032606,-122.062848),(37.032571,-122.062881),(37.032489,-122.062973),(37.032430,-122.063063),(37.032371,-122.063155),(37.032311,-122.063270),(37.032254,-122.063391),(37.032100,-122.063830),(37.031951,-122.064352),(37.031933,-122.064499),(37.031918,-122.064641),(37.031899,-122.064847),(37.031885,-122.064933),(37.031876,-122.064970),(37.031865,-122.065006),(37.031840,-122.065068),(37.031812,-122.065116),(37.031765,-122.065170),(37.031740,-122.065190),(37.031712,-122.065206),
]

# Branch 2: Glen Arbor Rd bike lanes (Class II) — corrected start point:
# was (37.079196,-122.077799); user-specified corrected alignment now runs
# from ~(37.081957,-122.078114) to ~(37.088822,-122.088503) — trimmed from
# the same source geometry via split_at_coord (end point is unchanged).
_b2_raw = [
    (37.079196,-122.077799),(37.079382,-122.077706),(37.079601,-122.077639),(37.079749,-122.077610),(37.079923,-122.077592),(37.080084,-122.077601),(37.080243,-122.077615),(37.080454,-122.077662),(37.081125,-122.077869),(37.081237,-122.077904),(37.081464,-122.077974),(37.081919,-122.078093),(37.082178,-122.078168),(37.082423,-122.078245),(37.082629,-122.078334),(37.082751,-122.078418),(37.082863,-122.078518),(37.083000,-122.078659),(37.083115,-122.078792),(37.083217,-122.078943),(37.083312,-122.079112),(37.083378,-122.079266),(37.083431,-122.079419),(37.083463,-122.079561),(37.083673,-122.080919),(37.083798,-122.081536),(37.083869,-122.081882),(37.083898,-122.082024),(37.083947,-122.082161),(37.084009,-122.082297),(37.084041,-122.082354),(37.084078,-122.082420),(37.084146,-122.082509),(37.084351,-122.082737),(37.084587,-122.082901),(37.084771,-122.082995),(37.085372,-122.083255),(37.085446,-122.083288),(37.085767,-122.083432),(37.086688,-122.083881),(37.086833,-122.083952),(37.086885,-122.083977),(37.087557,-122.084329),(37.088369,-122.084800),(37.088438,-122.084865),(37.088481,-122.084933),(37.088510,-122.085006),(37.088540,-122.085090),(37.088991,-122.086499),(37.089020,-122.086605),(37.089031,-122.086715),(37.089029,-122.086847),(37.089019,-122.086946),(37.089002,-122.087046),(37.088899,-122.087388),(37.088841,-122.087680),(37.088802,-122.087874),(37.088798,-122.087899),(37.088784,-122.088001),(37.088777,-122.088099),(37.088776,-122.088196),(37.088784,-122.088275),(37.088805,-122.088494),
]
_, b2_coords = split_at_coord(_b2_raw, (37.081957, -122.078114))

# Branch 3: Felton Empire Rd multi-use path (Class I) — Hwy 9 → Fetherston Way
_b3_raw = [
    (37.053080,-122.073282),(37.053092,-122.073417),(37.053098,-122.073583),(37.053102,-122.073685),(37.053118,-122.074103),(37.053104,-122.074456),(37.053084,-122.074639),(37.052994,-122.074970),(37.052965,-122.075034),(37.052618,-122.075816),(37.052544,-122.075982),(37.052235,-122.076633),(37.052030,-122.077064),(37.051719,-122.077759),(37.051269,-122.078765),(37.051173,-122.078972),(37.051018,-122.079302),(37.050746,-122.079886),(37.050671,-122.080068),(37.050385,-122.080765),(37.050113,-122.081428),(37.049752,-122.082116),(37.049402,-122.082737),(37.049304,-122.082980),(37.049247,-122.083210),(37.049206,-122.083433),(37.049177,-122.083711),(37.049146,-122.083995),
]

BRANCH_COLOR = "#808080"  # gray — uniform color for all proposed branches

BRANCHES = [
    dict(n=1, label="Hwy 9 Bike Lanes", cls="Class II",
         color=BRANCH_COLOR, dash=BRANCH_DASH,
         desc="Proposed Class II bike lanes along Hwy 9, from Graham Hill Rd "
              "to Glengarry Rd.",
         coords=_b1_raw),
    dict(n=2, label="Glen Arbor Rd Bike Lanes", cls="Class II",
         color=BRANCH_COLOR, dash=BRANCH_DASH,
         desc="Proposed Class II bike lanes along Glen Arbor Rd, corrected "
              "start point, to Hwy 9 (Ben Lomond).",
         coords=b2_coords),
    dict(n=3, label="Felton Empire Rd Multi-Use Path", cls="Class I",
         color=BRANCH_COLOR, dash=BRANCH_DASH,
         desc="Proposed multi-use path along Felton Empire Rd, from Hwy 9 "
              "to Fetherston Way.",
         coords=_b3_raw),
]

# ── Road-network catchment (static, restored from the original hand-built
# overlay) ─────────────────────────────────────────────────────────────────
# This geometry was never computed by this script — it (and the density
# choropleth) was hand-patched directly into slv_loop_map.html across
# several early commits, by a process/tool outside this repo. Several
# attempts to recompute it programmatically (census-block clipping, then a
# building-footprint + network-distance model) all diverged visually from
# that original and introduced their own artifacts. Restored verbatim here
# from git history (commit 58d41b2, the last one where it was hand-edited)
# instead: same shape, same population figures as the live site.
print("\nLoading road-network catchment overlays …")
_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load_geojson(fname):
    with open(os.path.join(_data_dir, fname)) as fh:
        return json.load(fh)


CATCHMENT_DATA = [
    dict(dist=0.25, file="catchment_025mi.geojson", pop=4831),
    dict(dist=0.50, file="catchment_050mi.geojson", pop=6826),
    dict(dist=1.00, file="catchment_100mi.geojson", pop=9190),
]

# ── Print segment table ───────────────────────────────────────────────────────
total = sum(mi(s["coords"]) for s in SEGS)
bridge_mi = mi(BRIDGE["coords"])
total_loop = total + bridge_mi
print(f"\n{'#':>2}  {'Segment':<52}  mi")
print("─" * 62)
for s in SEGS:
    d = mi(s["coords"])
    print(f"{s['n']:2d}  {s['label']:<52}  {d:.2f}")
print(f" {BRIDGE['n']}  {BRIDGE['label']:<52}  {bridge_mi:.2f}  (standalone)")
print("─" * 62)
print(f"{'Total (loop + bridge)':>56}  {total_loop:.2f}")

print(f"\n{'#':>2}  {'Proposed branch':<40}  mi")
print("─" * 50)
for b in BRANCHES:
    print(f"{b['n']:2d}  {b['label']:<40}  {mi(b['coords']):.2f}")

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
for b in BRANCHES:
    features.append({
        "type": "Feature",
        "geometry": {"type": "LineString",
                     "coordinates": [[lo, la] for la, lo in b["coords"]]},
        "properties": {"branch_number": b["n"], "label": b["label"],
                       "description": b["desc"], "color": b["color"],
                       "bikeway_class": b["cls"], "dashed": bool(b["dash"]),
                       "length_mi": round(mi(b["coords"]), 3)},
    })
features.append({
    "type": "Feature",
    "geometry": {"type": "LineString",
                 "coordinates": [[lo, la] for la, lo in BRIDGE["coords"]]},
    "properties": {"kind": "bridge", "label": BRIDGE["label"],
                   "description": BRIDGE["desc"], "color": BRIDGE["color"],
                   "dashed": bool(BRIDGE["dash"]),
                   "length_mi": round(mi(BRIDGE["coords"]), 3)},
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

def _render_seg(s, label_prefix="Seg"):
    seg_mi = mi(s["coords"])
    tip = (f"<b>{label_prefix} {s['n']}: {s['label']}</b><br>{s['desc']}<br>"
           f"<i>{seg_mi:.2f} mi</i>")
    kw = dict(locations=s["coords"], color=s["color"], weight=7, opacity=0.90,
              tooltip=folium.Tooltip(tip, sticky=True),
              popup=folium.Popup(tip, max_width=360))
    if s["dash"]:
        kw["dash_array"] = s["dash"]
    folium.PolyLine(**kw).add_to(m)
    if s["coords"]:
        mid = path_midpoint(s["coords"])
        folium.Marker(
            mid,
            icon=folium.DivIcon(
                html=(f'<div style="font-size:11px;font-weight:bold;color:#fff;'
                      f'background:{s["color"]};border-radius:50%;width:22px;'
                      f'height:22px;line-height:22px;text-align:center;'
                      f'border:2px solid #fff;box-shadow:1px 1px 3px rgba(0,0,0,.5);">'
                      f'{s["n"]}</div>'),
                icon_size=(22, 22), icon_anchor=(11, 11))).add_to(m)


for s in SEGS:
    _render_seg(s)
_render_seg(BRIDGE, label_prefix="Bridge")

for d in DESTS:
    folium.Marker(
        (d["lat"], d["lon"]),
        tooltip=f"<b>{d['name']}</b>",
        popup=folium.Popup(f"<b>{d['name']}</b><br>{d['desc']}", max_width=300),
        icon=folium.Icon(color=d["fcolor"], icon="star")).add_to(m)

# ── Proposed branch overlays (toggleable) ─────────────────────────────────────
branch_layer = folium.FeatureGroup(name="Branch proposals", show=True)
for b in BRANCHES:
    b_mi = mi(b["coords"])
    tip = (f"<b>Branch {b['n']} — {b['label']}</b><br>{b['desc']}<br>"
           f"<i>{b_mi:.2f} mi · {b['cls']}</i>")
    folium.PolyLine(
        locations=b["coords"], color=b["color"], weight=5, opacity=0.85,
        dash_array=b["dash"], line_cap="round",
        tooltip=folium.Tooltip(tip, sticky=True),
        popup=folium.Popup(tip, max_width=360)).add_to(branch_layer)
branch_layer.add_to(m)


def fetch_elevations(pts):
    """Fetch elevations (feet) from the USGS Elevation Point Query Service.

    Switched from the Open-Elevation public API, which proved unreliable
    (repeated connection timeouts / 504s) — USGS EPQS is government-run,
    needs no API key, and covers the US (fine for this project). It only
    takes one point per request, so this fetches sequentially.
    """
    elevs = []
    for la, lo in pts:
        value = None
        for attempt in range(3):
            try:
                r = _req.get("https://epqs.nationalmap.gov/v1/json",
                            params={"x": lo, "y": la, "units": "Feet", "wkid": 4326},
                            timeout=15)
                value = float(r.json()["value"])
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  ⚠  elevation fetch error at ({la:.4f},{lo:.4f}): {e}")
        elevs.append(value if value is not None and value > -1000 else 0)
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
elevs_ft = fetch_elevations(sampled)

elevs_smooth = smooth_list(elevs_ft, window=9)
baseline = elevs_smooth[0]
elevs_rel = [round(e - baseline) for e in elevs_smooth]

elev_gain = round(sum(max(0, b - a) for a, b in zip(elevs_rel[:-1], elevs_rel[1:])))
elev_abs_min = round(min(elevs_ft))
elev_abs_max = round(max(elevs_ft))
print(f"  Elevation: {elev_abs_min}–{elev_abs_max} ft absolute, +{elev_gain} ft gain")

dist_js   = "[" + ",".join(f"{d:.3f}" for d in cum_mi) + "]"
elev_js   = "[" + ",".join(str(e) for e in elevs_rel) + "]"
coords_js = "[" + ",".join(f"[{la:.6f},{lo:.6f}]" for la, lo in sampled) + "]"
map_var   = m.get_name()

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
        San Lorenzo Valley, Santa Cruz County, CA &nbsp;·&nbsp; ~{total_loop:.1f} mi
      </span>
      <hr style="margin:6px 0;border-color:#ccc;">
      {"".join(leg_row(s) for s in SEGS)}
      {leg_row(BRIDGE)}
      <hr style="margin:5px 0;border-color:#ccc;">
      <span style="font-size:10px;color:#666;">Proposed branches (toggle layer)</span>
      {"".join(leg_row(b) for b in BRANCHES)}
      <hr style="margin:5px 0;border-color:#ccc;">
      <span style="font-size:10px;color:#888;">
        &#9733; Key destinations &nbsp;|&nbsp; - - proposed bridge &amp; branches
      </span>
    </div>
    <div id="slv-elev" class="slv-panel">
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;">
        <b id="elev-title" style="font-size:12px;">Elevation Profile</b>
        <span id="elev-meta" style="font-size:10px;color:#666;">
          {elev_abs_min}–{elev_abs_max} ft &nbsp;·&nbsp; +{elev_gain} ft gain
        </span>
      </div>
      <canvas id="elevChart" height="90" style="width:100%;"></canvas>
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

  if (window.innerWidth < 600) {{ collapse(); }}
  btn.addEventListener('click', function() {{ sb.dataset.c === '1' ? expand() : collapse(); }});

  // Map dot that tracks elevation profile hover. Deferred to window 'load':
  // this <script> tag is emitted before folium's own map-building script
  // (which defines map_XXX as a global), so window['{map_var}'] would still
  // be undefined at parse time — the try/catch below was silently eating
  // that failure every time, so the dot never actually appeared.
  var coords = {coords_js};
  var hoverDot = null;
  window.addEventListener('load', function() {{
    try {{
      var leafletMap = window['{map_var}'];
      hoverDot = L.circleMarker([0, 0], {{
        radius: 8, color: '#fff', weight: 2.5,
        fillColor: '#2980B9', fillOpacity: 0, opacity: 0,
        interactive: false
      }}).addTo(leafletMap);
    }} catch(e) {{ console.warn('Hover dot init failed:', e); }}
  }});

  function showDot(idx) {{
    if (!hoverDot) return;
    hoverDot.setLatLng(coords[idx]);
    hoverDot.setStyle({{ opacity: 1, fillOpacity: 0.9 }});
  }}
  function hideDot() {{
    if (hoverDot) hoverDot.setStyle({{ opacity: 0, fillOpacity: 0 }});
  }}

  document.getElementById('elevChart').addEventListener('mouseleave', hideDot);

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
      animation: false,
      onHover: function(event, activeEls) {{
        if (activeEls.length) {{ showDot(activeEls[0].index); }}
        else {{ hideDot(); }}
      }},
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
            maxTicksLimit: 6, font: {{ size: 9 }},
            callback: function(val, i) {{
              return typeof dists[i] === 'number' ? dists[i].toFixed(1) + ' mi' : '';
            }}
          }},
          grid: {{ color: 'rgba(0,0,0,0.06)' }}
        }},
        y: {{
          ticks: {{
            font: {{ size: 9 }}, maxTicksLimit: 4,
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

# ── Population density & road-catchment overlays (toggleable, off by default) ─
# Previously maintained only as hand-written Leaflet JS patched directly into
# slv_loop_map.html (2020 Census block data + pre-computed road-network buffer
# polygons); ported here as static sidecar GeoJSON so the generator script is
# the single source of truth and these survive a regeneration.
_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load_geojson(fname):
    with open(os.path.join(_data_dir, fname)) as fh:
        return json.load(fh)


_YL_OR_RD = [(255, 255, 178), (254, 204, 92), (253, 141, 60),
             (240, 59, 32), (189, 0, 38)]


def _density_color(d):
    if not d or d <= 0:
        return "rgba(200,200,200,0)"
    t = min(1.0, max(0.0, math.log10(max(d, 1)) / math.log10(10000)))
    idx = t * (len(_YL_OR_RD) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(_YL_OR_RD) - 1)
    f = idx - lo
    r = round(_YL_OR_RD[lo][0] + f * (_YL_OR_RD[hi][0] - _YL_OR_RD[lo][0]))
    g = round(_YL_OR_RD[lo][1] + f * (_YL_OR_RD[hi][1] - _YL_OR_RD[lo][1]))
    b = round(_YL_OR_RD[lo][2] + f * (_YL_OR_RD[hi][2] - _YL_OR_RD[lo][2]))
    return f"rgba({r},{g},{b},0.68)"


density_layer = folium.FeatureGroup(name="Population Density", show=False)
folium.GeoJson(
    _load_geojson("density_blocks.geojson"),
    style_function=lambda f: {
        "fillColor": _density_color(f["properties"]["density"]),
        "fillOpacity": 1, "color": "transparent", "weight": 0,
    },
    tooltip=folium.GeoJsonTooltip(fields=["density", "POP100"],
                                   aliases=["Persons / sq mi", "Residents"]),
).add_to(density_layer)
density_layer.add_to(m)

# Road-network catchment rings — static, one FeatureGroup/checkbox per
# distance, population baked into both the checkbox label and a small
# legend box shown while that ring is checked (matching the live site).
CATCHMENT_COLORS = {0.25: "#3182BD", 0.50: "#E6550D", 1.00: "#6A3D9A"}
_catch_specs = []
for cd in CATCHMENT_DATA:
    dist, pop, color = cd["dist"], cd["pop"], CATCHMENT_COLORS[cd["dist"]]
    label = f"{dist:.2f} mi road catchment (~{pop:,} residents)"
    fg = folium.FeatureGroup(name=label, show=False)
    folium.GeoJson(
        _load_geojson(cd["file"]),
        style_function=lambda f, col=color: {
            "fillColor": col, "fillOpacity": 0.25, "color": col,
            "weight": 2, "dashArray": "6 4", "opacity": 0.85,
        },
    ).add_to(fg)
    fg.add_to(m)
    _catch_specs.append(dict(fg_var=fg.get_name(), dist=dist, pop=pop, color=color))

_catch_legend_js = "\n".join(f"""
  (function() {{
    var legend = L.control({{position: 'bottomright'}});
    legend.onAdd = function() {{
      var div = L.DomUtil.create('div', 'density-legend');
      div.innerHTML =
        '<b style="font-size:11px;">{c['dist']:.2f} mi road catchment</b>' +
        '<div style="margin-top:4px;font-size:11px;">' +
        '  <span style="display:inline-block;width:14px;height:14px;background:{c['color']}33;' +
        '  border:1.5px dashed {c['color']};vertical-align:middle;margin-right:5px;"></span>' +
        '  Area within {c['dist']:.2f} mi by road' +
        '</div>' +
        '<div style="margin-top:3px;font-size:12px;font-weight:bold;color:{c['color']};">' +
        '  ~{c['pop']:,} residents' +
        '</div>' +
        '<div style="font-size:9px;color:#888;margin-top:2px;">2020 Census, area-weighted</div>';
      return div;
    }};
    var fg = window['{c['fg_var']}'];
    leafletMap.on('overlayadd', function(e) {{ if (e.layer === fg) legend.addTo(leafletMap); }});
    leafletMap.on('overlayremove', function(e) {{ if (e.layer === fg) legend.remove(); }});
  }})();""" for c in _catch_specs)

catch_legend_js = f"""
<style>
.density-legend {{
  background: white; padding: 8px 10px; border-radius: 6px;
  border: 1px solid #aaa; font-family: Arial, sans-serif; font-size: 11px;
  box-shadow: 2px 2px 6px rgba(0,0,0,0.25); line-height: 1.4; min-width: 150px;
}}
</style>
<script>
window.addEventListener('load', function() {{
  // Deferred to window 'load': this <script> tag is emitted before folium's
  // own map-building script (which defines map_XXX as a global), so
  // referencing it any earlier would hit undefined and throw.
  var leafletMap = window['{m.get_name()}'];
  {_catch_legend_js}
}});
</script>
"""
m.get_root().html.add_child(folium.Element(catch_legend_js))

Fullscreen().add_to(m)
MeasureControl(position="topright", primary_length_unit="miles",
               secondary_length_unit="kilometers").add_to(m)
folium.LayerControl(position="topright").add_to(m)

m.save("slv_loop_map.html")
print("  → slv_loop_map.html")
print(f"\n✓  Done  —  {total_loop:.1f} mi loop, {len(SEGS)} segments + bridge, "
      f"{len(BRANCHES)} proposed branches")
