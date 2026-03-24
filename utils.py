import math
import uuid
import json
import bmesh
import mathutils.geometry
import bpy
import gpu
import mathutils
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector

from .constants import *

def create_style_id():
    return f"style_{uuid.uuid4().hex[:8]}"


def sanitize_name(value):
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(value))
    return safe.strip("_") or "Style"


def iter_dim_instances():
    for obj in bpy.data.objects:
        if obj.get("is_dim_instance"):
            yield obj


def get_scene_scale_length(scene):
    return scene.unit_settings.scale_length or 1.0


def copy_style_settings(source, target):
    for field in STYLE_COPY_FIELDS:
        if field == "name":
            continue
        setattr(target, field, getattr(source, field))


def unique_style_name(scene, base_name):
    used_names = {style.name for style in scene.dim_styles}
    if base_name not in used_names:
        return base_name

    index = 2
    while True:
        candidate = f"{base_name} {index}"
        if candidate not in used_names:
            return candidate
        index += 1


def ensure_default_style(scene):
    styles = scene.dim_styles
    if not styles:
        style = styles.add()
        style.name = "Default"
        style.style_id = DEFAULT_STYLE_ID

    for idx, style in enumerate(styles):
        if not style.style_id:
            style.style_id = DEFAULT_STYLE_ID if idx == 0 else create_style_id()

    if scene.dim_active_style_index < 0:
        scene.dim_active_style_index = 0
    if scene.dim_active_style_index >= len(styles):
        scene.dim_active_style_index = len(styles) - 1

    return styles[scene.dim_active_style_index]


def maybe_get_active_style(scene):
    styles = scene.dim_styles
    if not styles:
        return None

    if scene.dim_active_style_index < 0:
        return styles[0]
    if scene.dim_active_style_index >= len(styles):
        return styles[-1]
    return styles[scene.dim_active_style_index]


def get_active_style(scene):
    return ensure_default_style(scene)


def get_style_by_id(scene, style_id):
    ensure_default_style(scene)
    for style in scene.dim_styles:
        if style.style_id == style_id:
            return style
    return get_active_style(scene)


def get_style_for_dim(scene, obj):
    style_id = obj.get("style_id", DEFAULT_STYLE_ID)
    return get_style_by_id(scene, style_id)


def get_or_load_font(filepath):
    if not filepath or str(filepath).strip() == "":
        return bpy.data.fonts.get("Bfont")

    abs_path = bpy.path.abspath(filepath)
    for font in bpy.data.fonts:
        if font.filepath == abs_path or font.filepath == filepath:
            return font
    try:
        return bpy.data.fonts.load(abs_path)
    except Exception:
        return bpy.data.fonts.get("Bfont")


def get_formatted_text(dist_bu, scene, style):
    dist_m = dist_bu * get_scene_scale_length(scene)
    unit_choice = style.dim_unit
    show_suffix = style.dim_show_suffix
    prec = style.dim_precision

    if unit_choice == 'AUTO':
        unit_system = scene.unit_settings.system
        length_unit = scene.unit_settings.length_unit
        if unit_system == 'IMPERIAL':
            unit_choice = 'in' if length_unit == 'INCHES' else 'ft'
        else:
            if length_unit == 'MILLIMETERS':
                unit_choice = 'mm'
            elif length_unit == 'CENTIMETERS':
                unit_choice = 'cm'
            else:
                unit_choice = 'm'

    value = dist_m
    suffix = ""
    if unit_choice == 'cm':
        value *= 100.0
        suffix = "cm"
    elif unit_choice == 'mm':
        value *= 1000.0
        suffix = "mm"
    elif unit_choice == 'ft':
        value *= 3.2808399
        suffix = "ft"
    elif unit_choice == 'in':
        value *= 39.3700787
        suffix = "in"
    else:
        suffix = "m"

    result = f"{value:.{prec}f}"
    if show_suffix:
        result += suffix
    return result


def setup_shadeless_mat(mat_name, color):
    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    mat.diffuse_color = color

    has_alpha = len(color) == 4 and color[3] < 1.0
    if hasattr(mat, "blend_method"):
        mat.blend_method = 'BLEND' if has_alpha else 'OPAQUE'
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = 'NONE'

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node = nodes.new('ShaderNodeOutputMaterial')
    out_node.location = (300, 0)

    if has_alpha:
        mix_node = nodes.new('ShaderNodeMixShader')
        mix_node.inputs[0].default_value = color[3]

        transp_node = nodes.new('ShaderNodeBsdfTransparent')
        bsdf_node = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf_node.inputs[0].default_value = (color[0], color[1], color[2], 1.0)
        if len(bsdf_node.inputs) > 4:
            bsdf_node.inputs[4].default_value = 1.0

        links.new(transp_node.outputs[0], mix_node.inputs[1])
        links.new(bsdf_node.outputs[0], mix_node.inputs[2])
        links.new(mix_node.outputs[0], out_node.inputs[0])
    else:
        bsdf_node = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf_node.inputs[0].default_value = color
        if len(bsdf_node.inputs) > 4:
            bsdf_node.inputs[4].default_value = 1.0
        links.new(bsdf_node.outputs[0], out_node.inputs[0])

    return mat


def get_style_materials(style):
    style_key = sanitize_name(style.style_id)
    text_color = tuple(style.dim_text_color)
    line_color = text_color
    return {
        "ext": setup_shadeless_mat(
            f"Mat_Ext_{style_key}",
            (line_color[0], line_color[1], line_color[2], 0.8),
        ),
        "dim": setup_shadeless_mat(f"Mat_Dim_{style_key}", line_color),
        "text": setup_shadeless_mat(f"Mat_Dim_Text_{style_key}", text_color),
        "preview": setup_shadeless_mat(f"Mat_Dim_Text_Preview_{style_key}", text_color),
    }


def set_viewport_display_color(obj, color):
    if obj is None or not hasattr(obj, "color"):
        return
    rgba = tuple(color)
    if len(rgba) == 3:
        rgba = (rgba[0], rgba[1], rgba[2], 1.0)
    obj.color = rgba


def set_object_material(obj, material):
    if obj.data is None:
        return
    materials = obj.data.materials
    if materials:
        materials[0] = material
    else:
        materials.append(material)

@bpy.app.handlers.persistent
def auto_cleanup_dim_data_handler(dummy1=None, dummy2=None):
    auto_cleanup_dim_data()

def auto_cleanup_dim_data():
    active_cols = {obj.instance_collection for obj in bpy.data.objects if obj.instance_collection}

    for col in list(bpy.data.collections):
        if col.name.startswith("DimData_") and col not in active_cols:
            for obj in list(col.objects):
                data = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                if not data:
                    continue
                if data.__class__.__name__ == 'Mesh':
                    bpy.data.meshes.remove(data)
                elif data.__class__.__name__ == 'TextCurve':
                    bpy.data.curves.remove(data)
            bpy.data.collections.remove(col)


def remove_preview_text():
    preview_text = bpy.data.objects.get("Preview_Dim_Text")
    if not preview_text:
        return

    font_data = preview_text.data
    bpy.data.objects.remove(preview_text, do_unlink=True)
    if font_data and font_data.users == 0:
        bpy.data.curves.remove(font_data)


def remove_dimension_instance(instance_obj):
    if instance_obj is None:
        return

    col_data = instance_obj.instance_collection
    bpy.data.objects.remove(instance_obj, do_unlink=True)

    if col_data and col_data.users == 0:
        for obj in list(col_data.objects):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if not data:
                continue
            if data.__class__.__name__ == 'Mesh':
                bpy.data.meshes.remove(data)
            elif data.__class__.__name__ == 'TextCurve':
                bpy.data.curves.remove(data)
        bpy.data.collections.remove(col_data)


def add_arrow_to_bmesh(bm, arrow_style, d1_loc, d2_loc, x_axis, y_axis, arrow_size):
    if arrow_style == 'TICK':
        v_tick = (x_axis + y_axis).normalized() * (arrow_size / 2)
        v1 = bm.verts.new(d1_loc - v_tick)
        v2 = bm.verts.new(d1_loc + v_tick)
        bm.edges.new((v1, v2))
        v3 = bm.verts.new(d2_loc - v_tick)
        v4 = bm.verts.new(d2_loc + v_tick)
        bm.edges.new((v3, v4))
    else:
        line_dir = (d2_loc - d1_loc)
        if line_dir.length > 0.0001:
            line_dir.normalize()
        else:
            line_dir = Vector((1.0, 0.0, 0.0))
            
        a_width = arrow_size * 0.25
        base1 = d1_loc + line_dir * arrow_size
        v1 = bm.verts.new(d1_loc)
        v2 = bm.verts.new(base1 + y_axis * a_width)
        v3 = bm.verts.new(base1 - y_axis * a_width)
        bm.edges.new((v1, v2))
        bm.edges.new((v1, v3))

        base2 = d2_loc - line_dir * arrow_size
        v4 = bm.verts.new(d2_loc)
        v5 = bm.verts.new(base2 + y_axis * a_width)
        v6 = bm.verts.new(base2 - y_axis * a_width)
        bm.edges.new((v4, v5))
        bm.edges.new((v4, v6))

def build_arrow_mesh(mesh, arrow_style, d1_loc, d2_loc, x_axis, y_axis, arrow_size):
    if hasattr(mesh, "clear_geometry"):
        mesh.clear_geometry()
    bm = bmesh.new()
    add_arrow_to_bmesh(bm, arrow_style, d1_loc, d2_loc, x_axis, y_axis, arrow_size)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def apply_dimension_style(instance_obj, scene):
    import json
    if "points_json" in instance_obj:
        points = [Vector(p) for p in json.loads(instance_obj["points_json"])]
    else:
        points = [Vector(instance_obj["p1"]), Vector(instance_obj["p2"])]
        
    data = {
        'points': points,
        'offset_dir': Vector(instance_obj["offset_dir"]),
        'offset_dist': instance_obj["offset_dist"],
        'style_id': instance_obj["style_id"],
        'linear_axis': instance_obj.get("linear_axis"),
    }
    
    class DummyContext:
        def __init__(self, scene):
            self.scene = scene
            self.space_data = None
            
    create_real_dimension(data, DummyContext(scene), existing_instance=instance_obj)

def iter_dim_instances():
    col = bpy.data.collections.get(COLLECTION_NAME)
    if col:
        for obj in list(col.objects):
            if obj.get("is_dim_instance"):
                yield obj

def update_all_dimensions(self, context):
    if not context:
        return
    scene = context.scene
    ensure_default_style(scene)
    for obj in iter_dim_instances():
        apply_dimension_style(obj, scene)

def get_or_create_main_collection(scene):
    if COLLECTION_NAME not in bpy.data.collections:
        new_col = bpy.data.collections.new(COLLECTION_NAME)
        scene.collection.children.link(new_col)
    return bpy.data.collections[COLLECTION_NAME]

def create_real_dimension(data, context, existing_instance=None):
    scene = context.scene
    style = get_style_by_id(scene, data.get('style_id'))

    offset_dir = data['offset_dir'].normalized()
    offset_dist = data.get('offset_dist', 0.0)
    
    linear_axis_name = data.get('linear_axis')
    linear_axis = None
    if linear_axis_name:
        axes_vecs = {'X': Vector((1,0,0)), 'Y': Vector((0,1,0)), 'Z': Vector((0,0,1))}
        linear_axis = axes_vecs.get(linear_axis_name)
    
    if 'points' in data:
        points = [Vector(p) for p in data['points']]
    else:
        points = [Vector(data['p1']), Vector(data['p2'])]

    if len(points) < 2:
        return existing_instance

    p0 = points[0]
    total_raw_dist = 0.0
    for i in range(len(points)-1):
        total_raw_dist += (points[i+1] - points[i]).length
        
    scale_x = style.dim_scale_x
    t_size = (style.dim_text_size_mm / 1000.0) * scale_x
    t_gap = (style.dim_text_gap_mm / 1000.0) * scale_x
    overshoot = (style.dim_ext_overshoot_mm / 1000.0) * scale_x
    fixed_len = (style.dim_ext_fixed_len_mm / 1000.0) * scale_x
    arrow_size = (style.dim_arrow_size_mm / 1000.0) * scale_x
    custom_font = get_or_load_font(style.dim_font_path)
    materials = get_style_materials(style)

    viewport_color = tuple(style.dim_text_color)
    
    if linear_axis:
        first_x_line = linear_axis
    else:
        first_x_line = (points[1] - points[0]).normalized() if (points[1] - points[0]).length>0.0001 else Vector((1,0,0))
    v_normal = first_x_line.cross(offset_dir).normalized()
    if v_normal.length < 0.0001:
        v_normal = Vector((0,0,1))
        
    rv3d = None
    if hasattr(context, "space_data") and getattr(context, "space_data", None) and getattr(context.space_data, "type", "") == 'VIEW_3D':
        rv3d = getattr(context.space_data, "region_3d", None)
    elif hasattr(context, "area") and getattr(context, "area", None) and getattr(context.area, "type", "") == 'VIEW_3D':
        rv3d = getattr(context.area.spaces.active, "region_3d", None)

    if rv3d:
        view_rot = rv3d.view_rotation
        view_fwd = view_rot @ Vector((0, 0, -1))
        view_right = view_rot @ Vector((1, 0, 0))
    else:
        view_fwd = Vector((0,0,-1))
        view_right = Vector((1,0,0))

    z_axis = v_normal if v_normal.dot(view_fwd) < 0 else -v_normal
    x_axis = first_x_line if first_x_line.dot(view_right) > 0 else -first_x_line
    y_axis = z_axis.cross(x_axis).normalized()

    rot_matrix = Matrix((x_axis, y_axis, z_axis)).transposed()
    
    import uuid
    if existing_instance:
        col_data = existing_instance.instance_collection
        instance_obj = existing_instance
        if col_data:
            for obj in list(col_data.objects):
                data_block = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                if data_block:
                    if data_block.__class__.__name__ == 'Mesh':
                        bpy.data.meshes.remove(data_block)
                    elif data_block.__class__.__name__ == 'TextCurve':
                        bpy.data.curves.remove(data_block)
        else:
            col_data = bpy.data.collections.new(f"DimData_{style.style_id}_{uuid.uuid4().hex[:6]}")
            instance_obj.instance_collection = col_data
    else:
        col_data = bpy.data.collections.new(f"DimData_{style.style_id}_{uuid.uuid4().hex[:6]}")
        instance_obj = bpy.data.objects.new(f"DimChain_{uuid.uuid4().hex[:4]}", None)
        instance_obj.instance_type = 'COLLECTION'
        instance_obj.instance_collection = col_data
        instance_obj.location = p0
        instance_obj.empty_display_size = 0.0
        get_or_create_main_collection(scene).objects.link(instance_obj)

    ext_verts = []
    ext_edges = []
    dim_verts = []
    dim_edges = []

    arrow_bm = bmesh.new()

    for i in range(len(points)):
        pt_loc = points[i] - p0
        
        t = pt_loc.dot(first_x_line)
        d_loc = offset_dir * offset_dist + t * first_x_line
        
        if style.dim_ext_use_fixed:
            start_loc = d_loc - offset_dir * fixed_len
        else:
            start_loc = pt_loc
        end_loc = d_loc + offset_dir * overshoot
        
        idx = len(ext_verts)
        ext_verts.extend([start_loc, end_loc])
        ext_edges.append((idx, idx+1))

        dim_verts.append(d_loc)
        
        if i > 0:
            dim_edges.append((i-1, i))
            
            prev_d_loc = dim_verts[i-1]
            dist_raw = abs((points[i] - points[i-1]).dot(first_x_line))
            text_str_seg = get_formatted_text(dist_raw, scene, style)
            
            font_data = bpy.data.curves.new(name="DimText_Data", type='FONT')
            font_data.body = text_str_seg
            font_data.size = t_size
            font_data.align_x = 'CENTER'
            font_data.align_y = 'BOTTOM'
            if custom_font:
                font_data.font = custom_font
                
            obj_text = bpy.data.objects.new(f"Dimension_Text_{i}", font_data)
            col_data.objects.link(obj_text)
            set_object_material(obj_text, materials["text"])
            set_viewport_display_color(obj_text, viewport_color)
            
            mid_point_loc = (prev_d_loc + d_loc) / 2
            obj_text.location = mid_point_loc + y_axis * t_gap + z_axis * max((points[i]-points[i-1]).length * 0.002, 0.001)
            obj_text.rotation_euler = rot_matrix.to_euler('XYZ')
            
            add_arrow_to_bmesh(arrow_bm, style.dim_arrow_style, prev_d_loc, d_loc, x_axis, y_axis, arrow_size)

    mesh_ext = bpy.data.meshes.new("Extension_Mesh")
    mesh_ext.from_pydata(ext_verts, ext_edges, [])
    obj_ext = bpy.data.objects.new("Extension_Lines", mesh_ext)
    col_data.objects.link(obj_ext)
    set_object_material(obj_ext, materials["ext"])
    set_viewport_display_color(obj_ext, viewport_color)

    mesh_dim = bpy.data.meshes.new("Dimension_Mesh")
    mesh_dim.from_pydata(dim_verts, dim_edges, [])
    obj_dim = bpy.data.objects.new("Dimension_Line", mesh_dim)
    col_data.objects.link(obj_dim)
    set_object_material(obj_dim, materials["dim"])
    set_viewport_display_color(obj_dim, viewport_color)

    mesh_arrows = bpy.data.meshes.new("Dimension_Arrows")
    arrow_bm.to_mesh(mesh_arrows)
    arrow_bm.free()
    obj_arrows = bpy.data.objects.new("Dimension_Arrows", mesh_arrows)
    col_data.objects.link(obj_arrows)
    set_object_material(obj_arrows, materials["dim"])
    set_viewport_display_color(obj_arrows, viewport_color)

    set_viewport_display_color(instance_obj, viewport_color)
    import json
    instance_obj["is_dim_instance"] = True
    if linear_axis_name:
        instance_obj["linear_axis"] = linear_axis_name
    elif "linear_axis" in instance_obj:
        del instance_obj["linear_axis"]
    instance_obj["points_json"] = json.dumps([[v.x, v.y, v.z] for v in points])
    instance_obj["offset_dir"] = offset_dir
    instance_obj["offset_dist"] = offset_dist
    instance_obj["Y_axis"] = y_axis
    instance_obj["Z_axis"] = z_axis
    instance_obj["style_id"] = style.style_id
    
    instance_obj["p1"] = points[0]
    instance_obj["p2"] = points[-1]
    
    return instance_obj


def get_preview_material(scene):
    style = get_active_style(scene)
    return get_style_materials(style)["preview"]


