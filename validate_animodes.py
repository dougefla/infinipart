#!/usr/bin/env python3
"""Validate animode joint selection for all factories across all seeds."""
import xml.etree.ElementTree as ET
import os, json
from collections import defaultdict

BASE_DIR = "/mnt/data/yurh/Infinite-Mobility"

# Copy FACTORY_RULES and selection logic from render_articulation.py
import math
FACTORY_RULES = {
    "DishwasherFactory": {"animode_joints": {0: [("revolute",)], 1: [("prismatic",)], 2: [("revolute",), ("prismatic",)]}},
    "BeverageFridgeFactory": {"animode_joints": {0: [("revolute",)], 1: [("prismatic",)], 2: [("revolute",), ("prismatic",)]}},
    "MicrowaveFactory": {"animode_joints": {0: [("revolute",)], 1: [("continuous",)], 2: [("revolute",), ("continuous",)]}},
    "OvenFactory": {"animode_joints": {0: [("revolute",)], 1: [("prismatic",)], 2: [("revolute",), ("prismatic",)]}},
    "KitchenCabinetFactory": {"animode_joints": {0: [("revolute",)], 1: [("prismatic",)], 2: [("revolute",), ("prismatic",)]}},
    "ToiletFactory": {"animode_joints": {0: [("revolute", -1)], 1: [("revolute", 0)], 2: [("prismatic",)], 3: [("revolute",), ("prismatic",)]}},
    "WindowFactory": {"animode_joints": {0: [("revolute", 0)], 1: [("revolute", -1)], 2: [("prismatic",)], 3: [("revolute",)], 4: [("revolute",), ("prismatic",)]}},
    "OfficeChairFactory": {"animode_joints": {0: [("prismatic",)], 1: [("revolute",)], 2: [("prismatic",), ("revolute",)]}},
    "TapFactory": {"animode_joints": {0: [("revolute",)], 1: [("continuous",)], 2: [("revolute",), ("continuous",)]}},
    "LampFactory": {"animode_joints": {0: [("prismatic", 0)], 1: [("prismatic", -1)], 2: [("revolute", 0)], 3: [("prismatic",), ("revolute",)]}},
    "PotFactory": {"animode_joints": {0: [("prismatic",)], 1: [("continuous",)], 2: [("prismatic",), ("continuous",)]}},
    "BottleFactory": {"animode_joints": {0: [("prismatic",)], 1: [("continuous",)], 2: [("prismatic",), ("continuous",)]}},
}

MIN_PRISMATIC = 0.005
MIN_ROTARY = 0.05

def parse_urdf(path):
    tree = ET.parse(path)
    root = tree.getroot()
    joints = []
    parent_map = {}
    for j in root.findall("joint"):
        jtype = j.get("type")
        name = j.get("name")
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        limit = j.find("limit")
        lower = float(limit.get("lower", 0)) if limit is not None else 0
        upper = float(limit.get("upper", 0)) if limit is not None else 0
        axis_el = j.find("axis")
        axis = [float(x) for x in axis_el.get("xyz").split()] if axis_el is not None else [0, 0, 0]
        joints.append({"name": name, "jtype": jtype, "parent": parent, "child": child,
                       "lower": lower, "upper": upper, "axis": axis})
        parent_map[child] = (parent, joints[-1])
    return joints, parent_map

def is_significant(j):
    rng = abs(j["upper"] - j["lower"])
    if j["jtype"] == "prismatic": return rng >= MIN_PRISMATIC
    if j["jtype"] in ("revolute", "continuous"): return rng >= MIN_ROTARY
    return False

def compute_depth(child_link, parent_map):
    depth = 0
    curr = child_link
    while curr in parent_map:
        depth += 1
        curr = parent_map[curr][0]
    return depth

def resolve_selectors(selectors, non_fixed, parent_map):
    sig = [j for j in non_fixed if is_significant(j)]
    selected = set()
    for sel in selectors:
        if len(sel) == 1:
            sel_type = sel[0]
            selected |= {j["name"] for j in sig if j["jtype"] == sel_type}
        elif len(sel) == 3 and sel[1] == "axis":
            sel_type, _, axis_name = sel
            axis_idx = {"x": 0, "y": 1, "z": 2}[axis_name.lower()]
            selected |= {j["name"] for j in sig if j["jtype"] == sel_type
                         and abs(j["axis"][axis_idx]) > 0.5}
        elif len(sel) == 3 and sel[1] == "sign":
            sel_type, _, sign = sel
            for j in sig:
                if j["jtype"] != sel_type:
                    continue
                target = j["upper"] if abs(j["upper"]) >= abs(j["lower"]) else j["lower"]
                if sign == "+" and target > 0:
                    selected.add(j["name"])
                elif sign == "-" and target < 0:
                    selected.add(j["name"])
        elif len(sel) == 2:
            sel_type, sel_ord = sel
            candidates = [(compute_depth(j["child"], parent_map), j["name"])
                          for j in sig if j["jtype"] == sel_type]
            candidates.sort()
            if candidates:
                selected.add(candidates[sel_ord][1])
    return selected

# Validate
for factory in sorted(FACTORY_RULES.keys()):
    cfg = FACTORY_RULES[factory]["animode_joints"]
    max_animode = max(cfg.keys())

    print(f"\n{'='*60}")
    print(f"{factory}: {max_animode+1} animodes")

    issues = []
    for seed in range(8):
        urdf = os.path.join(BASE_DIR, f"outputs/{factory}/{seed}/scene.urdf")
        if not os.path.exists(urdf):
            continue
        joints, parent_map = parse_urdf(urdf)
        non_fixed = [j for j in joints if j["jtype"] != "fixed"]

        for animode in range(max_animode + 1):
            if animode not in cfg:
                continue
            selected = resolve_selectors(cfg[animode], non_fixed, parent_map)
            if not selected:
                issues.append(f"  seed={seed} animode={animode}: EMPTY (no joints matched)")
            else:
                types = defaultdict(int)
                for jname in selected:
                    j = next(j for j in non_fixed if j["name"] == jname)
                    types[j["jtype"]] += 1
                type_str = ", ".join(f"{t}:{c}" for t, c in sorted(types.items()))
                if seed == 0:  # Only print detail for seed 0
                    print(f"  animode {animode}: {len(selected)} joints ({type_str})")
                    for jname in sorted(selected):
                        j = next(j for j in non_fixed if j["name"] == jname)
                        print(f"    {jname}: {j['jtype']} [{j['lower']:+.4f}, {j['upper']:+.4f}]")

    if issues:
        print(f"  ISSUES:")
        for i in issues:
            print(i)
    else:
        print(f"  All seeds OK")
