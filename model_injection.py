import xml.etree.ElementTree as ET

import mujoco


GROUND_FRAGMENT = (
    '<geom name="__mtool_local_ground" type="plane" pos="0 0 0" '
    'size="0 0 0.1" rgba="0.25 0.28 0.32 1"/>'
)


def append_static_xml(xml_text, fragment):
    try:
        root = ET.fromstring(xml_text)
        element = ET.fromstring(fragment)
    except ET.ParseError as error:
        raise ValueError(f"注入 XML 解析失败：{error}") from error
    if element.tag not in {"geom", "body"}:
        raise ValueError("只能向 worldbody 注入 geom 或 body")
    if any(node.tag in {"joint", "freejoint"} for node in element.iter()):
        raise ValueError("注入元素不能包含 joint 或 freejoint")

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    worldbody.append(element)
    return ET.tostring(root, encoding="unicode")


def inject_ground(xml_text):
    return append_static_xml(xml_text, GROUND_FRAGMENT)


def joint_layout(model):
    return (
        int(model.nq),
        int(model.nv),
        int(model.nu),
        int(model.njnt),
        tuple(int(value) for value in model.jnt_type),
        tuple(int(value) for value in model.jnt_qposadr),
        tuple(int(value) for value in model.jnt_dofadr),
        tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in range(model.njnt)
        ),
    )
