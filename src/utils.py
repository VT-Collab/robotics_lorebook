from .env import PandaEnv
from .objects import CollabObject
import json
import re

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


def generate_objects_table(env: PandaEnv) -> str:
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
            handle_orn = [round(x, 2) for x in state["handle_euler"]]
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

    table = "| Object | Position | Orientation | Dimensions (WxLxH) |\n| ------ | -------- | ----------- | ------------------ |"
    for obj in info:
        table += f'\n| {obj["type"]} | {obj["pos"]} | {obj["orn"]} | {obj["dims"]} |'
    return table
