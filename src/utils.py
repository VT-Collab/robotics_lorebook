from .env import PandaEnv, RealEnv
from .objects import CollabObject, YCBObject, RoboCasaObject
import json
import re
from math import fmod, pi
from typing import Union
from .llm import SimpleRAG, DoubleSimRAG

def extract_json(content: str):
    """Helper to extract JSON from markdown code blocks"""
    try:
        # Try finding standard json block
        match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        # Fallback to plain text parsing
        return json.loads(content)
    except Exception:
        return None


def get_total_aabb(env: PandaEnv, body_id):
    cp_min, cp_max = env.p.getAABB(body_id, linkIndex=-1)
    for i in range(env.p.getNumJoints(body_id)):
        link_min, link_max = env.p.getAABB(body_id, linkIndex=i)
        cp_min = [min(cp_min[j], link_min[j]) for j in range(3)]
        cp_max = [max(cp_max[j], link_max[j]) for j in range(3)]
    return tuple(cp_min), tuple(cp_max)


def generate_relative_objects_table(env: PandaEnv) -> str:
    info = []
    for obj_entry in env.objects:
        body_id = obj_entry["id"]
        t = obj_entry["type"]
        if t == "plane":
            continue
        pos, quat = env.p.getBasePositionAndOrientation(body_id)
        euler = [round(x, 2) for x in env.p.getEulerFromQuaternion(quat)]
        pos = [round(x, 2) for x in pos]
        aabb_min, aabb_max = get_total_aabb(env, body_id)
        dims = [round(aabb_max[i] - aabb_min[i], 3) for i in range(3)]
        info.append({"type": t, "pos": pos, "orn": euler, "dims": dims})

        # Handle handles/sub-parts
        if isinstance(obj_entry["ref"], CollabObject):
            state = obj_entry["ref"].get_state()
            handle_pos = [round(x, 2) for x in state["handle_position"]]
            handle_orn = [round(fmod(x, pi), 2) for x in state["handle_euler"]]
            h_min, h_max = env.p.getAABB(body_id, linkIndex=1)
            h_dims = [round(h_max[i] - h_min[i], 3) for i in range(3)]
            info.append(
                {
                    "type": t + " handle",
                    "pos": handle_pos,
                    "orn": handle_orn,
                    "dims": h_dims,
                }
            )
    table_info = "Position refers to the center of the object. Dimensions are relative to the object center and object's orientation. WxLxH refers to XxYxZ.\n\n"
    table = "| Object | Position | Orientation | Dimensions (WxLxH) |\n| ------ | -------- | ----------- | ------------------ |"
    for obj in info:
        table += f'\n| {obj["type"]} | {obj["pos"]} | {obj["orn"]} | {obj["dims"]} |'
    return table_info + table


def generate_aabb_objects_table(env: PandaEnv) -> str:
    info = []
    for obj_entry in env.objects:
        body_id = obj_entry["id"]
        t = obj_entry["type"]
        if t == "plane":
            continue
        pos, quat = env.p.getBasePositionAndOrientation(body_id)
        euler = [round(x, 2) for x in env.p.getEulerFromQuaternion(quat)]
        aabb_min, aabb_max = get_total_aabb(env, body_id)
        aabb_min = [round(x, 3) for x in aabb_min]
        aabb_max = [round(x, 3) for x in aabb_max]
        info.append(
            {
                "type": t,
                "pos": [round(x, 2) for x in pos],
                "min": aabb_min,
                "max": aabb_max,
                "yaw": euler[2],
            }
        )
        if isinstance(obj_entry["ref"], CollabObject):
            state = obj_entry["ref"].get_state()
            handle_pos = [round(x, 2) for x in state["handle_position"]]
            handle_orn = [round(min(pi - fmod(x, pi), fmod(x, pi)), 2) for x in state["handle_euler"]]
            h_min, h_max = env.p.getAABB(body_id, linkIndex=1)
            info.append(
                {
                    "type": t + " handle",
                    "pos": handle_pos,
                    "min": [round(x, 3) for x in h_min],
                    "max": [round(x, 3) for x in h_max],
                    "yaw": handle_orn[2],
                }
            )
    table_header = (
        "### Scene Objects (World Coordinates)\n"
        "The table below lists objects with their Axis-Aligned Bounding Box (AABB) in [X, Y, Z].\n"
        "- **AABB Min**: The [min_x, min_y, min_z] corner.\n"
        "- **AABB Max**: The [max_x, max_y, max_z] corner.\n"
        "- To avoid collision, your path must stay outside these ranges.\n\n"
    )

    table = "| Object | Center (X,Y,Z) | AABB Min | AABB Max | Yaw |\n"
    table += "| :--- | :--- | :--- | :--- | :--- |\n"

    for obj in info:
        table += f"| {obj['type']} | {obj['pos']} | {obj['min']} | {obj['max']} | {obj['yaw']} |\n"

    return table_header + table


def generate_objects_table(env: Union[PandaEnv, RealEnv], new_detection: bool = False) -> str:
    if isinstance(env, PandaEnv):
        info = []
        for obj_entry in env.objects:
            body_id = obj_entry["id"]
            t = obj_entry["type"]
            if t == "plane":
                continue

            # 1. Position and Full Orientation
            pos, quat = env.p.getBasePositionAndOrientation(body_id)
            euler = [round(x, 2) for x in env.p.getEulerFromQuaternion(quat)]
            if isinstance(obj_entry["ref"], YCBObject):
                # Normalize (with 2*pi - x flip) into [-pi/2, pi/2)
                euler = [round((((2 * pi - x) + pi / 2) % pi) - pi / 2, 2) for x in euler]

            # 2. Bounding Box (AABB) Calculations
            aabb_min, aabb_max = get_total_aabb(env, body_id)

            # 3. Calculate Dimensions from AABB
            dims = [round(aabb_max[i] - aabb_min[i], 3) for i in range(3)]

            obj_data = {
                "type": t,
                "pos": [round(x, 2) for x in pos],
                "orn": euler,
                "min": [round(x, 3) for x in aabb_min],
                "max": [round(x, 3) for x in aabb_max],
                "dims": dims,
                "yaw": euler[2],
            }
            info.append(obj_data)

            # Handle handles/sub-parts
            if isinstance(obj_entry["ref"], CollabObject) or isinstance(obj_entry["ref"], RoboCasaObject):
                state = obj_entry["ref"].get_state()
                if state.get("handle_position") is None:
                    continue
                h_min, h_max = env.p.getAABB(body_id, linkIndex=1)
                h_dims = [round(h_max[i] - h_min[i], 3) for i in range(3)]
                h_euler = [round(-fmod(2 * pi - x, pi), 2) for x in state["handle_euler"]]

                info.append(
                    {
                        "type": t + " handle",
                        "pos": [round(x, 2) for x in state["handle_position"]],
                        "orn": h_euler,
                        "min": [round(x, 3) for x in h_min],
                        "max": [round(x, 3) for x in h_max],
                        "dims": h_dims,
                        "yaw": h_euler[2],
                    }
                )
    else: # If using real world
        if new_detection:
            env.load_objects()
        info = []

        for obj in env.objects:
            name = obj["type"] 
            bbox = obj["bbox"]
            pos = obj["pos"]
            min_corners = min(bbox['corners'], key=lambda c: sum(c))
            max_corners = max(bbox['corners'], key=lambda c: sum(c))
            obj_info = {
                "type": name,
                "pos": [round(x, 2) for x in pos] if pos else None,
                "orn": None,  # Placeholder
                "min": [round(float(x), 3) for x in min_corners] if bbox else None,
                "max": [round(float(x), 3) for x in max_corners] if bbox else None,
                "dims": [round(bbox['width'], 3), round(bbox['length'], 3), round(bbox['height'], 3)] if bbox else None,  
                "yaw": float(f"{bbox['angle']:.3f}") if bbox else None,
            }
            info.append(obj_info)
        
        if env.previous_marked_pos: 
            object = env.previous_marked_pos[1][0] # take the first marked object
            info.append({
                "type": env.previous_marked_pos[0],
                "pos": [round(object["x"], 2), round(object["y"], 2), round(object["z"], 2)],
                "orn": [round(i, 2) for i in object["robot_rvec"]],
                "min": None,
                "max": None,
                "dims": None,
                "yaw": round(object["robot_rvec"][2], 2)
            })


    # Header designed to prime the LLM's spatial awareness
    header = (
        "## Scene Spatial Data\n"
        "All coordinates are in World Space [X, Y, Z].\n"
        "- **AABB Min/Max**: Absolute boundaries. Any point P is inside if Min <= P <= Max.\n"
        "- **Dimensions**: Total width, length, and height (X, Y, Z spread).\n"
        "- **Yaw**: Rotation around the Z-axis in radians.\n\n"
    )

    table = (
        "| Object | Center Pos | Orientation (R,P,Y) | AABB Min | AABB Max | Dimensions | Yaw |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    )

    for obj in info:
        table += (
            f"| {obj['type']} | {obj['pos']} | {obj['orn']} | {obj['min']} | "
            f"{obj['max']} | {obj['dims']} | {obj['yaw']} |\n"
        )

    return header + table



def convert_from_simple_to_ds_rag(old_path, new_path, model_name="all-MiniLM-L6-v2"):
    import os 
    import pickle
    if not os.path.exists(old_path):
        print(f"Error: {old_path} not found.")
        return

    print(f"Loading old lorebook from {old_path}...")
    with open(old_path, "rb") as fh:
        old_data = pickle.load(fh)

    # Initialize the model and the new RAG
    # We use the DoubleSimRag class to ensure the parsing logic matches perfectly
    new_rag = DoubleSimRAG(filename=None, model_name=model_name)
    
    old_metadata = old_data.get("metadata", [])
    total = len(old_metadata)
    
    print(f"Migrating {total} entries. This will re-encode vectors...")

    for i, entry in enumerate(old_metadata):
        # In SimpleRAG, metadata is a list of dicts: {'key': '...', 'value': '...'}
        key = entry['key']
        value = entry['value']
        
        # We use the internal split logic from our new class
        branch, act_part, obj_env_part = new_rag._split_components(key)
        
        # Generate the new dual-vector representation
        vec_act = new_rag.model.encode(act_part)
        vec_obj = new_rag.model.encode(obj_env_part if obj_env_part else "none")
        
        # Manually append to the new_rag structures
        new_rag.vectors.append((vec_act, vec_obj))
        new_rag.metadata.append({
            "key": key,
            "value": value,
            "branch": branch,
            "act_str": act_part,
            "obj_str": obj_env_part
        })
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{total}")

    # Save the migrated data
    print(f"Saving migrated lorebook to {new_path}...")
    new_rag.save_to_file(new_path)
    print("Migration complete!")