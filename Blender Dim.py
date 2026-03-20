bl_info = {
    "name": "Pro 3D Dimension Tool",
    "author": "Pro CAD User",
    "version": (1, 2),
    "blender": (3, 0, 0),
    "location": "View3D > UI (N Panel) > Pro Dim",
    "description": "Cong cu do kich thuoc 3D chuyen nghiep chuan CAD/Sketchup",
    "warning": "",
    "doc_url": "",
    "category": "3D View",
}

import math
import uuid

import bmesh
import mathutils.geometry
import bpy
import gpu
import mathutils
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector

try:
    SHADER = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
except ValueError:
    SHADER = gpu.shader.from_builtin('UNIFORM_COLOR')

COLLECTION_NAME = "Dimensions"
DEFAULT_STYLE_ID = "default"

STYLE_COPY_FIELDS = (
    "name",
    "dim_unit",
    "dim_show_suffix",
    "dim_precision",
    "dim_font_path",
    "dim_text_color",
    "dim_scale_x",
    "dim_text_size_mm",
    "dim_text_gap_mm",
    "dim_ext_overshoot_mm",
    "dim_arrow_style",
    "dim_arrow_size_mm",
    "dim_ext_use_fixed",
    "dim_ext_fixed_len_mm",
)


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


def set_object_material(obj, material):
    if obj.data is None:
        return
    materials = obj.data.materials
    if materials:
        materials[0] = material
    else:
        materials.append(material)

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
    return 2.0


def remove_preview_text():
    preview_text = bpy.data.objects.get("Preview_Dim_Text")
    if not preview_text:
        return

    font_data = preview_text.data
    bpy.data.objects.remove(preview_text, do_unlink=True)
    if font_data and font_data.users == 0:
        bpy.data.curves.remove(font_data)


def build_arrow_mesh(mesh, arrow_style, d1_loc, d2_loc, x_axis, y_axis, arrow_size):
    if hasattr(mesh, "clear_geometry"):
        mesh.clear_geometry()

    bm = bmesh.new()
    if arrow_style == 'TICK':
        v_tick = (x_axis + y_axis).normalized() * (arrow_size / 2)
        v1 = bm.verts.new(d1_loc - v_tick)
        v2 = bm.verts.new(d1_loc + v_tick)
        bm.edges.new((v1, v2))
        v3 = bm.verts.new(d2_loc - v_tick)
        v4 = bm.verts.new(d2_loc + v_tick)
        bm.edges.new((v3, v4))
    else:
        a_width = arrow_size * 0.25
        base1 = d1_loc + x_axis * arrow_size
        v1 = bm.verts.new(d1_loc)
        v2 = bm.verts.new(base1 + y_axis * a_width)
        v3 = bm.verts.new(base1 - y_axis * a_width)
        bm.edges.new((v1, v2))
        bm.edges.new((v1, v3))

        base2 = d2_loc - x_axis * arrow_size
        v4 = bm.verts.new(d2_loc)
        v5 = bm.verts.new(base2 + y_axis * a_width)
        v6 = bm.verts.new(base2 - y_axis * a_width)
        bm.edges.new((v4, v5))
        bm.edges.new((v4, v6))

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def apply_dimension_style(instance_obj, scene):
    style = get_style_for_dim(scene, instance_obj)
    materials = get_style_materials(style)
    custom_font = get_or_load_font(style.dim_font_path)

    p1 = Vector(instance_obj["p1"])
    p2 = Vector(instance_obj["p2"])
    offset_dir = Vector(instance_obj["offset_dir"])
    offset_dist = instance_obj["offset_dist"]
    y_axis = Vector(instance_obj["Y_axis"])
    z_axis = Vector(instance_obj["Z_axis"])
    x_axis = y_axis.cross(z_axis).normalized()

    dist_raw = (p1 - p2).length
    text_str = get_formatted_text(dist_raw, scene, style)

    d1 = p1 + offset_dir * offset_dist
    d2 = p2 + offset_dir * offset_dist

    p1_loc = Vector((0, 0, 0))
    p2_loc = p2 - p1
    d1_loc = d1 - p1
    d2_loc = d2 - p1

    scale_x = style.dim_scale_x
    t_size = (style.dim_text_size_mm / 1000.0) * scale_x
    t_gap = (style.dim_text_gap_mm / 1000.0) * scale_x
    overshoot = (style.dim_ext_overshoot_mm / 1000.0) * scale_x
    fixed_len = (style.dim_ext_fixed_len_mm / 1000.0) * scale_x
    arrow_size = (style.dim_arrow_size_mm / 1000.0) * scale_x

    if style.dim_ext_use_fixed:
        start1_loc = d1_loc - offset_dir * fixed_len
        start2_loc = d2_loc - offset_dir * fixed_len
    else:
        start1_loc = p1_loc
        start2_loc = p2_loc

    d1_ext_loc = d1_loc + offset_dir * overshoot
    d2_ext_loc = d2_loc + offset_dir * overshoot

    col_data = instance_obj.instance_collection
    if not col_data:
        return

    for child in col_data.objects:
        if "Extension_Lines" in child.name and child.type == 'MESH':
            child.data.vertices[0].co = start1_loc
            child.data.vertices[1].co = d1_ext_loc
            child.data.vertices[2].co = start2_loc
            child.data.vertices[3].co = d2_ext_loc
            child.data.update()
            set_object_material(child, materials["ext"])
        elif "Dimension_Line" in child.name and child.type == 'MESH':
            child.data.vertices[0].co = d1_loc
            child.data.vertices[1].co = d2_loc
            child.data.update()
            set_object_material(child, materials["dim"])
        elif "Dimension_Arrows" in child.name and child.type == 'MESH':
            build_arrow_mesh(child.data, style.dim_arrow_style, d1_loc, d2_loc, x_axis, y_axis, arrow_size)
            set_object_material(child, materials["dim"])
        elif "Dimension_Text" in child.name and child.type == 'FONT':
            child.data.body = text_str
            child.data.size = t_size
            child.data.align_x = 'CENTER'
            child.data.align_y = 'BOTTOM'
            if custom_font:
                child.data.font = custom_font
            child.location = ((d1_loc + d2_loc) / 2) + y_axis * t_gap + z_axis * max(dist_raw * 0.002, 0.001)
            set_object_material(child, materials["text"])

    instance_obj.name = f"Dim_{text_str}"


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


def create_real_dimension(data, context):
    scene = context.scene
    style = get_active_style(scene)

    p1, p2, d1, d2 = data['p1'], data['p2'], data['d1'], data['d2']
    offset_dir = data['offset_dir'].normalized()
    offset_dist = (d1 - p1).dot(offset_dir)

    scale_x = style.dim_scale_x
    t_size = (style.dim_text_size_mm / 1000.0) * scale_x
    t_gap = (style.dim_text_gap_mm / 1000.0) * scale_x
    overshoot = (style.dim_ext_overshoot_mm / 1000.0) * scale_x
    fixed_len = (style.dim_ext_fixed_len_mm / 1000.0) * scale_x
    arrow_size = (style.dim_arrow_size_mm / 1000.0) * scale_x
    custom_font = get_or_load_font(style.dim_font_path)
    materials = get_style_materials(style)

    dist_raw = (p1 - p2).length
    text_str = get_formatted_text(dist_raw, scene, style)

    col_data = bpy.data.collections.new(f"DimData_{style.style_id}_{scene.frame_current}_{uuid.uuid4().hex[:6]}")

    p1_loc = Vector((0, 0, 0))
    p2_loc = p2 - p1
    d1_loc = d1 - p1
    d2_loc = d2 - p1

    if style.dim_ext_use_fixed:
        start1_loc = d1_loc - offset_dir * fixed_len
        start2_loc = d2_loc - offset_dir * fixed_len
    else:
        start1_loc = p1_loc
        start2_loc = p2_loc

    d1_ext_loc = d1_loc + offset_dir * overshoot
    d2_ext_loc = d2_loc + offset_dir * overshoot

    mesh_ext = bpy.data.meshes.new("Extension_Mesh")
    mesh_ext.from_pydata([start1_loc, d1_ext_loc, start2_loc, d2_ext_loc], [(0, 1), (2, 3)], [])
    obj_ext = bpy.data.objects.new("Extension_Lines", mesh_ext)
    col_data.objects.link(obj_ext)
    set_object_material(obj_ext, materials["ext"])

    mesh_dim = bpy.data.meshes.new("Dimension_Mesh")
    mesh_dim.from_pydata([d1_loc, d2_loc], [(0, 1)], [])
    obj_dim = bpy.data.objects.new("Dimension_Line", mesh_dim)
    col_data.objects.link(obj_dim)
    set_object_material(obj_dim, materials["dim"])
    x_line = (p2 - p1).normalized()
    v_normal = x_line.cross(offset_dir).normalized()

    rv3d = context.space_data.region_3d
    view_rot = rv3d.view_rotation
    view_fwd = view_rot @ Vector((0, 0, -1))
    view_right = view_rot @ Vector((1, 0, 0))

    z_axis = v_normal if v_normal.dot(view_fwd) < 0 else -v_normal
    x_axis = x_line if x_line.dot(view_right) > 0 else -x_line
    y_axis = z_axis.cross(x_axis).normalized()

    rot_matrix = Matrix((x_axis, y_axis, z_axis)).transposed()

    mesh_arrows = bpy.data.meshes.new("Dimension_Arrows")
    build_arrow_mesh(mesh_arrows, style.dim_arrow_style, d1_loc, d2_loc, x_axis, y_axis, arrow_size)
    obj_arrows = bpy.data.objects.new("Dimension_Arrows", mesh_arrows)
    col_data.objects.link(obj_arrows)
    set_object_material(obj_arrows, materials["dim"])

    font_data = bpy.data.curves.new(name="DimText_Data", type='FONT')
    font_data.body = text_str
    font_data.size = t_size
    font_data.align_x = 'CENTER'
    font_data.align_y = 'BOTTOM'
    if custom_font:
        font_data.font = custom_font

    obj_text = bpy.data.objects.new("Dimension_Text", font_data)
    col_data.objects.link(obj_text)
    set_object_material(obj_text, materials["text"])

    mid_point_loc = (d1_loc + d2_loc) / 2
    obj_text.location = mid_point_loc + y_axis * t_gap + z_axis * max(dist_raw * 0.002, 0.001)
    obj_text.rotation_euler = rot_matrix.to_euler('XYZ')

    main_col = get_or_create_main_collection(scene)
    instance_obj = bpy.data.objects.new(f"Dim_{text_str}", None)
    instance_obj.instance_type = 'COLLECTION'
    instance_obj.instance_collection = col_data
    instance_obj.location = p1
    instance_obj.empty_display_size = 0.0

    instance_obj["is_dim_instance"] = True
    instance_obj["p1"] = p1
    instance_obj["p2"] = p2
    instance_obj["offset_dir"] = offset_dir
    instance_obj["offset_dist"] = offset_dist
    instance_obj["Y_axis"] = y_axis
    instance_obj["Z_axis"] = z_axis
    instance_obj["style_id"] = style.style_id

    main_col.objects.link(instance_obj)


def get_preview_material(scene):
    style = get_active_style(scene)
    return get_style_materials(style)["preview"]


class DimStyleItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Style Name", default="Style", update=update_all_dimensions)
    style_id: bpy.props.StringProperty(name="Style ID", default="")

    dim_unit: bpy.props.EnumProperty(
        name="Unit",
        items=[
            ('AUTO', "Scene Unit", ""),
            ('m', "Meters (m)", ""),
            ('cm', "Centimeters (cm)", ""),
            ('mm', "Millimeters (mm)", ""),
            ('ft', "Feet (ft)", ""),
            ('in', "Inches (in)", ""),
        ],
        default='AUTO',
        update=update_all_dimensions,
    )
    dim_show_suffix: bpy.props.BoolProperty(name="Show Suffix", default=False, update=update_all_dimensions)
    dim_precision: bpy.props.IntProperty(name="Decimals", default=0, min=0, max=5, update=update_all_dimensions)
    dim_font_path: bpy.props.StringProperty(
        name="Font Path",
        description="Duong dan den file font (.ttf, .otf)",
        subtype='FILE_PATH',
        update=update_all_dimensions,
    )

    dim_text_color: bpy.props.FloatVectorProperty(
        name="Text Color",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0),
        update=update_all_dimensions,
    )
    dim_scale_x: bpy.props.FloatProperty(name="Scale", default=100.0, min=1.0, update=update_all_dimensions)
    dim_text_size_mm: bpy.props.FloatProperty(name="Text Size", default=2.0, min=0.1, update=update_all_dimensions)
    dim_text_gap_mm: bpy.props.FloatProperty(name="Text Gap", default=0.5, min=0.0, update=update_all_dimensions)
    dim_ext_overshoot_mm: bpy.props.FloatProperty(name="Overshoot", default=0.0, min=0.0, update=update_all_dimensions)
    dim_arrow_style: bpy.props.EnumProperty(
        name="Arrow Style",
        items=[('ARROW', "Arrow", ""), ('TICK', "Tick", "")],
        default='ARROW',
        update=update_all_dimensions,
    )
    dim_arrow_size_mm: bpy.props.FloatProperty(name="Arrow Size", default=3.0, min=0.1, update=update_all_dimensions)
    dim_ext_use_fixed: bpy.props.BoolProperty(name="Fixed Ext Lines", default=True, update=update_all_dimensions)
    dim_ext_fixed_len_mm: bpy.props.FloatProperty(name="Fixed Length", default=4.0, min=1.0, update=update_all_dimensions)


class VIEW3D_UL_DimStyles(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.prop(item, "name", text="", emboss=False, icon='MATERIAL')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon='MATERIAL')


class OT_DimStyleAdd(bpy.types.Operator):
    bl_idname = "view3d.dim_style_add"
    bl_label = "Add Style"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        source_style = get_active_style(scene)
        new_style = scene.dim_styles.add()
        copy_style_settings(source_style, new_style)
        new_style.name = unique_style_name(scene, f"{source_style.name} Copy")
        new_style.style_id = create_style_id()
        scene.dim_active_style_index = len(scene.dim_styles) - 1
        update_all_dimensions(self, context)
        return {'FINISHED'}


class OT_DimStyleRemove(bpy.types.Operator):
    bl_idname = "view3d.dim_style_remove"
    bl_label = "Remove Style"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        ensure_default_style(scene)

        if len(scene.dim_styles) <= 1:
            self.report({'WARNING'}, "At least one style must remain.")
            return {'CANCELLED'}

        remove_index = scene.dim_active_style_index
        remove_style = scene.dim_styles[remove_index]

        fallback_index = 0 if remove_index != 0 else 1
        fallback_style_id = scene.dim_styles[fallback_index].style_id

        for obj in iter_dim_instances():
            if obj.get("style_id") == remove_style.style_id:
                obj["style_id"] = fallback_style_id

        scene.dim_styles.remove(remove_index)
        scene.dim_active_style_index = min(remove_index, len(scene.dim_styles) - 1)
        update_all_dimensions(self, context)
        return {'FINISHED'}

class OT_DimAssignActiveStyle(bpy.types.Operator):
    bl_idname = "view3d.dim_assign_active_style"
    bl_label = "Assign Active Style"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        active_style = get_active_style(scene)
        assigned = 0

        for obj in context.selected_objects:
            if obj.get("is_dim_instance"):
                obj["style_id"] = active_style.style_id
                assigned += 1

        if assigned == 0:
            self.report({'WARNING'}, "Select one or more dimension instances first.")
            return {'CANCELLED'}

        update_all_dimensions(self, context)
        self.report({'INFO'}, f"Assigned style to {assigned} dimension(s).")
        return {'FINISHED'}


class OT_SketchupProDim(bpy.types.Operator):
    bl_idname = "view3d.sketchup_pro_dim"
    bl_label = "Pro 3D Dim Tool"
    bl_options = {'REGISTER', 'UNDO'}

    _is_running = False
    _handle = None
    data = {
        'step': 0,
        'snap_loc': None,
        'snap_loc_raw': None,
        'p1': None,
        'p2': None,
        'd1': None,
        'd2': None,
        'offset_dir': None,
        'snap_color': (1.0, 0.0, 0.0, 1.0),
        'p2_constraint': None,
    }

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == 'VIEW_3D'

    def get_constraint_axes(self):
        return {
            'X': Vector((1, 0, 0)),
            'Y': Vector((0, 1, 0)),
            'Z': Vector((0, 0, 1)),
        }

    def get_constraint_label(self):
        constraint = self.__class__.data.get('p2_constraint')
        if not constraint:
            return 'Free'
        mode, axis = constraint
        return f"Axis {axis}" if mode == 'AXIS' else f"Plane !{axis}"

    def update_step_header(self, context):
        step = self.__class__.data['step']
        if step == 0:
            context.area.header_text_set("Step 1: Pick Start Point")
        elif step == 1:
            label = self.get_constraint_label()
            context.area.header_text_set(f"Step 2: Pick End Point | Constraint: {label} | X/Y/Z = Axis, Shift+X/Y/Z = Plane")
        elif step == 2:
            context.area.header_text_set("Step 3: Move to set Direction & Distance. Click to finish.")

    def apply_p2_constraint(self, candidate_loc):
        cls_data = self.__class__.data
        p1 = cls_data.get('p1')
        constraint = cls_data.get('p2_constraint')
        if candidate_loc is None or p1 is None or not constraint:
            return candidate_loc

        mode, axis_name = constraint
        axis_vec = self.get_constraint_axes()[axis_name]
        delta = candidate_loc - p1

        if mode == 'AXIS':
            return p1 + axis_vec * delta.dot(axis_vec)

        return candidate_loc - axis_vec * delta.dot(axis_vec)

    def get_snap_settings(self, context):
        tool_settings = context.scene.tool_settings
        return tool_settings.use_snap, set(tool_settings.snap_elements)

    def closest_point_on_segment_2d(self, point, a, b):
        ab = b - a
        length_sq = ab.length_squared
        if length_sq == 0.0:
            return a.copy(), 0.0
        t = max(0.0, min(1.0, (point - a).dot(ab) / length_sq))
        return a + ab * t, t

    def get_vertex_snap_candidate(self, region, rv3d, poly, mat, mesh, mouse_2d, threshold):
        best = None
        best_dist = threshold
        for v_idx in poly.vertices:
            v_world = mat @ mesh.vertices[v_idx].co
            v_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, v_world)
            if not v_2d:
                continue
            dist_2d = (v_2d - mouse_2d).length
            if dist_2d < best_dist:
                best = v_world
                best_dist = dist_2d
        return best, best_dist

    def get_edge_snap_candidate(self, region, rv3d, poly, mat, mesh, mouse_2d, threshold):
        best = None
        best_dist = threshold
        for edge_key in poly.edge_keys:
            v1_world = mat @ mesh.vertices[edge_key[0]].co
            v2_world = mat @ mesh.vertices[edge_key[1]].co
            v1_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, v1_world)
            v2_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, v2_world)
            if not v1_2d or not v2_2d:
                continue
            closest_2d, factor = self.closest_point_on_segment_2d(mouse_2d, v1_2d, v2_2d)
            dist_2d = (closest_2d - mouse_2d).length
            if dist_2d < best_dist:
                best = v1_world.lerp(v2_world, factor)
                best_dist = dist_2d
        return best, best_dist

    def get_raw_snap_location(self, context, event):
        scene = context.scene
        region = context.region
        rv3d = context.space_data.region_3d
        coord = (event.mouse_region_x, event.mouse_region_y)
        mouse_2d = Vector(coord)
        view_vec = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        result, location, _norm, idx, obj, mat = scene.ray_cast(context.view_layer.depsgraph, ray_origin, view_vec)

        if not result:
            return None

        use_snap, snap_elements = self.get_snap_settings(context)
        if not obj or obj.type != 'MESH':
            return location

        mesh = obj.data
        poly = mesh.polygons[idx]
        threshold = 30.0

        if not use_snap or not snap_elements:
            vertex_loc, vertex_dist = self.get_vertex_snap_candidate(region, rv3d, poly, mat, mesh, mouse_2d, threshold)
            return vertex_loc if vertex_loc is not None and vertex_dist < threshold else location

        best_loc = None
        best_dist = threshold

        if 'VERTEX' in snap_elements:
            vertex_loc, vertex_dist = self.get_vertex_snap_candidate(region, rv3d, poly, mat, mesh, mouse_2d, threshold)
            if vertex_loc is not None and vertex_dist < best_dist:
                best_loc = vertex_loc
                best_dist = vertex_dist

        if 'EDGE' in snap_elements or 'EDGE_MIDPOINT' in snap_elements:
            edge_loc, edge_dist = self.get_edge_snap_candidate(region, rv3d, poly, mat, mesh, mouse_2d, threshold)
            if edge_loc is not None and edge_dist < best_dist:
                best_loc = edge_loc
                best_dist = edge_dist

        if best_loc is not None:
            return best_loc

        if 'FACE' in snap_elements or 'FACE_NEAREST' in snap_elements:
            return location

        return location

    def update_proxy_text(self, context):
        preview_text = bpy.data.objects.get("Preview_Dim_Text")
        if not preview_text:
            return

        cls_data = self.__class__.data
        if cls_data['step'] != 2 or not cls_data['d1']:
            preview_text.hide_viewport = True
            return

        scene = context.scene
        style = get_active_style(scene)
        p1, p2 = cls_data['p1'], cls_data['p2']
        dist = (p1 - p2).length
        scale_x = style.dim_scale_x
        t_size = (style.dim_text_size_mm / 1000.0) * scale_x
        t_gap = (style.dim_text_gap_mm / 1000.0) * scale_x
        custom_font = get_or_load_font(style.dim_font_path)

        preview_text.hide_viewport = False
        preview_text.data.body = get_formatted_text(dist, scene, style)
        preview_text.data.size = t_size
        preview_text.data.align_y = 'BOTTOM'
        if custom_font:
            preview_text.data.font = custom_font
        set_object_material(preview_text, get_preview_material(scene))

        x_line = (p2 - p1).normalized()
        offset_dir = cls_data['offset_dir'].normalized()
        v_normal = x_line.cross(offset_dir).normalized()

        rv3d = context.space_data.region_3d
        view_rot = rv3d.view_rotation
        view_fwd = view_rot @ Vector((0, 0, -1))
        view_right = view_rot @ Vector((1, 0, 0))

        z_axis = v_normal if v_normal.dot(view_fwd) < 0 else -v_normal
        x_axis = x_line if x_line.dot(view_right) > 0 else -x_line
        y_axis = z_axis.cross(x_axis).normalized()

        rot_matrix = Matrix((x_axis, y_axis, z_axis)).transposed()
        mid_point = (cls_data['d1'] + cls_data['d2']) / 2
        preview_text.rotation_euler = rot_matrix.to_euler('XYZ')
        preview_text.location = mid_point + y_axis * t_gap + z_axis * max(dist * 0.002, 0.001)

    def modal(self, context, event):
        nav_events = {
            'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
            'NUMPAD_1', 'NUMPAD_2', 'NUMPAD_3', 'NUMPAD_4', 'NUMPAD_5',
            'NUMPAD_6', 'NUMPAD_7', 'NUMPAD_8', 'NUMPAD_9', 'NUMPAD_0',
            'TRACKPADPAN', 'TRACKPADZOOM',
        }
        if event.type in nav_events:
            return {'PASS_THROUGH'}

        cls_data = self.__class__.data

        if cls_data['step'] == 1 and event.type in {'X', 'Y', 'Z'} and event.value == 'PRESS':
            new_constraint = ('PLANE', event.type) if event.shift else ('AXIS', event.type)
            cls_data['p2_constraint'] = None if cls_data.get('p2_constraint') == new_constraint else new_constraint
            if cls_data.get('snap_loc_raw') is not None:
                cls_data['snap_loc'] = self.apply_p2_constraint(cls_data['snap_loc_raw'])
            self.update_step_header(context)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE':
            self.mouse_pos = Vector((event.mouse_region_x, event.mouse_region_y))
            if cls_data['step'] < 2:
                raw_loc = self.get_raw_snap_location(context, event)
                cls_data['snap_loc_raw'] = raw_loc
                cls_data['snap_loc'] = self.apply_p2_constraint(raw_loc) if cls_data['step'] == 1 else raw_loc
            elif cls_data['step'] == 2:
                self.calculate_combined_offset(context)
                self.update_proxy_text(context)
            context.area.tag_redraw()

        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if cls_data['step'] == 0 and cls_data['snap_loc']:
                cls_data['p1'] = cls_data['snap_loc'].copy()
                cls_data['step'] = 1
                cls_data['snap_loc_raw'] = cls_data['snap_loc']
                self.update_step_header(context)

            elif cls_data['step'] == 1 and cls_data['snap_loc']:
                cls_data['p2'] = cls_data['snap_loc'].copy()
                cls_data['step'] = 2
                self.update_step_header(context)
                self.calculate_combined_offset(context)
                self.update_proxy_text(context)

            elif cls_data['step'] == 2:
                create_real_dimension(cls_data, context)
                self.stop_ui(context)
                return {'FINISHED'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self.stop_ui(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def calculate_combined_offset(self, context):
        cls_data = self.__class__.data
        p1, p2 = cls_data['p1'], cls_data['p2']
        if not p1 or not p2:
            return
        if (p2 - p1).length < 0.0001:
            return

        v_line = (p2 - p1).normalized()
        region = context.region
        rv3d = context.space_data.region_3d
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, self.mouse_pos)
        ray_dir = view3d_utils.region_2d_to_vector_3d(region, rv3d, self.mouse_pos)

        pt_mouse = view3d_utils.region_2d_to_location_3d(region, rv3d, self.mouse_pos, p1)
        if not pt_mouse:
            return

        v_raw = pt_mouse - p1
        v_perp = v_raw - v_raw.dot(v_line) * v_line
        if v_perp.length < 0.001:
            return

        style = get_active_style(context.scene)
        free_dir = v_perp.normalized()
        best_dir = free_dir
        snap_color = tuple(style.dim_text_color)
        final_dist = v_perp.length

        axes = {
            'X': (Vector((1, 0, 0)), (1.0, 0.2, 0.2, 1.0)),
            'Y': (Vector((0, 1, 0)), (0.2, 1.0, 0.2, 1.0)),
            'Z': (Vector((0, 0, 1)), (0.2, 0.6, 1.0, 1.0)),
        }

        min_angle = math.radians(15)
        for axis_vec, color in axes.values():
            a_perp = axis_vec - axis_vec.dot(v_line) * v_line
            if a_perp.length <= 0.001:
                continue

            snap_dir = a_perp.normalized()
            if free_dir.dot(snap_dir) < 0:
                snap_dir = -snap_dir

            angle = free_dir.angle(snap_dir)
            if angle < min_angle:
                min_angle = angle
                best_dir = snap_dir
                snap_color = color

                plane_normal = v_line.cross(best_dir).normalized()
                pt_intersect = mathutils.geometry.intersect_line_plane(
                    ray_origin,
                    ray_origin + ray_dir,
                    p1,
                    plane_normal,
                )
                if pt_intersect:
                    final_dist = (pt_intersect - p1).dot(best_dir)
                break

        cls_data['offset_dir'] = best_dir
        cls_data['snap_color'] = snap_color
        cls_data['d1'] = p1 + best_dir * final_dist
        cls_data['d2'] = p2 + best_dir * final_dist

    def draw_callback_px(self, context):
        if context.area is None:
            return

        cls_data = self.__class__.data
        step = cls_data['step']
        region = context.region
        rv3d = context.space_data.region_3d

        if step < 2 and cls_data['snap_loc']:
            snap_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, cls_data['snap_loc'])
            if snap_2d:
                size = 6
                batch = batch_for_shader(
                    SHADER,
                    'LINES',
                    {"pos": [(snap_2d.x - size, snap_2d.y), (snap_2d.x + size, snap_2d.y), (snap_2d.x, snap_2d.y - size), (snap_2d.x, snap_2d.y + size)]},
                )
                SHADER.bind()
                SHADER.uniform_float("color", (1, 0, 0, 1))
                batch.draw(SHADER)

        if step == 1 and cls_data['p1'] and cls_data['snap_loc']:
            p1_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, cls_data['p1'])
            p2_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, cls_data['snap_loc'])
            if p1_2d and p2_2d:
                guide = batch_for_shader(SHADER, 'LINES', {"pos": [p1_2d, p2_2d]})
                constraint = cls_data.get('p2_constraint')
                color_map = {
                    ('AXIS', 'X'): (1.0, 0.2, 0.2, 1.0),
                    ('AXIS', 'Y'): (0.2, 1.0, 0.2, 1.0),
                    ('AXIS', 'Z'): (0.2, 0.6, 1.0, 1.0),
                    ('PLANE', 'X'): (1.0, 0.6, 0.6, 1.0),
                    ('PLANE', 'Y'): (0.6, 1.0, 0.6, 1.0),
                    ('PLANE', 'Z'): (0.6, 0.8, 1.0, 1.0),
                }
                SHADER.bind()
                SHADER.uniform_float("color", color_map.get(constraint, (1.0, 1.0, 0.0, 1.0)))
                guide.draw(SHADER)

        if step != 2 or not cls_data['d1'] or not cls_data['d2']:
            return

        style = get_active_style(context.scene)
        scale_x = style.dim_scale_x
        overshoot = (style.dim_ext_overshoot_mm / 1000.0) * scale_x
        fixed_len = (style.dim_ext_fixed_len_mm / 1000.0) * scale_x
        arrow_size = (style.dim_arrow_size_mm / 1000.0) * scale_x

        offset_dir = cls_data['offset_dir']
        d1 = cls_data['d1']
        d2 = cls_data['d2']

        if style.dim_ext_use_fixed:
            start1 = d1 - offset_dir * fixed_len
            start2 = d2 - offset_dir * fixed_len
        else:
            start1 = cls_data['p1']
            start2 = cls_data['p2']

        d1_ext = d1 + offset_dir * overshoot
        d2_ext = d2 + offset_dir * overshoot

        start1_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, start1)
        start2_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, start2)
        d1_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, d1)
        d2_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, d2)
        d1_ext_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, d1_ext)
        d2_ext_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, d2_ext)

        if not all((start1_2d, start2_2d, d1_2d, d2_2d, d1_ext_2d, d2_ext_2d)):
            return

        preview_color = cls_data['snap_color']
        batch = batch_for_shader(SHADER, 'LINES', {"pos": [start1_2d, d1_ext_2d, start2_2d, d2_ext_2d, d1_2d, d2_2d]})
        SHADER.bind()
        SHADER.uniform_float("color", preview_color)
        batch.draw(SHADER)

        v_dir = (d2 - d1).normalized()
        if v_dir.length <= 0.0001:
            return

        x_line = v_dir
        offset_dir_n = offset_dir.normalized()
        v_normal = x_line.cross(offset_dir_n).normalized()
        view_rot = rv3d.view_rotation
        view_fwd = view_rot @ Vector((0, 0, -1))
        view_right = view_rot @ Vector((1, 0, 0))
        z_axis = v_normal if v_normal.dot(view_fwd) < 0 else -v_normal
        x_axis = x_line if x_line.dot(view_right) > 0 else -x_line
        y_axis = z_axis.cross(x_axis).normalized()

        arrow_pos_2d = []
        arrow_batch = None
        if style.dim_arrow_style == 'TICK':
            v_tick = (x_axis + y_axis).normalized() * (arrow_size / 2)
            t1_1 = view3d_utils.location_3d_to_region_2d(region, rv3d, d1 - v_tick)
            t1_2 = view3d_utils.location_3d_to_region_2d(region, rv3d, d1 + v_tick)
            t2_1 = view3d_utils.location_3d_to_region_2d(region, rv3d, d2 - v_tick)
            t2_2 = view3d_utils.location_3d_to_region_2d(region, rv3d, d2 + v_tick)
            if all((t1_1, t1_2, t2_1, t2_2)):
                arrow_pos_2d.extend([t1_1, t1_2, t2_1, t2_2])
                arrow_batch = batch_for_shader(SHADER, 'LINES', {"pos": arrow_pos_2d})
        else:
            a_width = arrow_size * 0.25
            base1 = d1 + v_dir * arrow_size
            base2 = d2 - v_dir * arrow_size
            p1_1 = view3d_utils.location_3d_to_region_2d(region, rv3d, d1)
            p1_2 = view3d_utils.location_3d_to_region_2d(region, rv3d, base1 + y_axis * a_width)
            p1_3 = view3d_utils.location_3d_to_region_2d(region, rv3d, base1 - y_axis * a_width)
            p2_1 = view3d_utils.location_3d_to_region_2d(region, rv3d, d2)
            p2_2 = view3d_utils.location_3d_to_region_2d(region, rv3d, base2 + y_axis * a_width)
            p2_3 = view3d_utils.location_3d_to_region_2d(region, rv3d, base2 - y_axis * a_width)
            if all((p1_1, p1_2, p1_3, p2_1, p2_2, p2_3)):
                arrow_pos_2d.extend([p1_1, p1_2, p1_1, p1_3, p2_1, p2_2, p2_1, p2_3])
                arrow_batch = batch_for_shader(SHADER, 'LINES', {"pos": arrow_pos_2d})

        if arrow_batch:
            SHADER.bind()
            SHADER.uniform_float("color", preview_color)
            arrow_batch.draw(SHADER)

    def invoke(self, context, event):
        if context.area.type != 'VIEW_3D':
            return {'CANCELLED'}
        if self.__class__._is_running:
            return {'CANCELLED'}

        active_style = get_active_style(context.scene)
        self.__class__._is_running = True
        self.__class__.data = {
            'step': 0,
            'snap_loc': None,
            'snap_loc_raw': None,
            'p1': None,
            'p2': None,
            'd1': None,
            'd2': None,
            'offset_dir': None,
            'snap_color': tuple(active_style.dim_text_color),
            'p2_constraint': None,
        }

        remove_preview_text()
        preview_font = bpy.data.curves.new(name="Preview_Dim_Font", type='FONT')
        preview_font.align_x = 'CENTER'
        preview_font.align_y = 'BOTTOM'
        preview_text = bpy.data.objects.new("Preview_Dim_Text", preview_font)
        context.collection.objects.link(preview_text)
        set_object_material(preview_text, get_preview_material(context.scene))
        preview_text.hide_viewport = True

        self.mouse_pos = Vector((event.mouse_region_x, event.mouse_region_y))
        self.update_step_header(context)
        self.__class__._handle = bpy.types.SpaceView3D.draw_handler_add(self.draw_callback_px, (context,), 'WINDOW', 'POST_PIXEL')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def stop_ui(self, context):
        self.__class__._is_running = False
        if context.area:
            context.area.header_text_set(None)

        remove_preview_text()

        if getattr(self.__class__, "_handle", None):
            bpy.types.SpaceView3D.draw_handler_remove(self.__class__._handle, 'WINDOW')
            self.__class__._handle = None
        if context.area:
            context.area.tag_redraw()

class VIEW3D_PT_ProDim(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Pro Dim'
    bl_label = 'Pro Dim'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        style = maybe_get_active_style(scene)

        if OT_SketchupProDim._is_running:
            layout.label(text="Follow Header. ESC to exit", icon='INFO')
        else:
            layout.operator("view3d.sketchup_pro_dim", text="Start Dimming", icon='DRIVER_DISTANCE')

        layout.separator()

        if style is None:
            layout.label(text="Style data is not initialized yet.", icon='INFO')
            layout.operator("view3d.dim_style_add", text="Create Default Style", icon='ADD')
            return

        style_box = layout.box()
        style_box.label(text="Style Library:")
        row = style_box.row()
        row.template_list("VIEW3D_UL_DimStyles", "", scene, "dim_styles", scene, "dim_active_style_index", rows=4)
        col = row.column(align=True)
        col.operator("view3d.dim_style_add", text="", icon='ADD')
        col.operator("view3d.dim_style_remove", text="", icon='REMOVE')

        style_box.prop(style, "name", text="Active Style")
        style_box.operator("view3d.dim_assign_active_style", text="Assign Active Style To Selected", icon='STYLUS_PRESSURE')

        settings = layout.box()
        settings.label(text="Unit Settings:")
        settings.prop(style, "dim_unit", text="Unit")
        row = settings.row()
        row.prop(style, "dim_show_suffix", text="Show Suffix")
        row.prop(style, "dim_precision", text="Decimals")

        scale_box = layout.box()
        scale_box.label(text="Annotative Scale:")
        row = scale_box.row()
        row.label(text="Scale 1 :")
        row.prop(style, "dim_scale_x", text="")

        geom_box = layout.box()
        geom_box.label(text="Geometry:")
        geom_box.prop(style, "dim_text_size_mm", text="Text Size")
        geom_box.prop(style, "dim_text_gap_mm", text="Text Gap")
        geom_box.prop(style, "dim_ext_overshoot_mm", text="Ext Overshoot")
        geom_box.prop(style, "dim_arrow_style", text="Arrow Style")
        geom_box.prop(style, "dim_arrow_size_mm", text="Arrow Size")
        geom_box.prop(style, "dim_ext_use_fixed", text="Fixed Ext Lines")
        if style.dim_ext_use_fixed:
            geom_box.prop(style, "dim_ext_fixed_len_mm", text="Fixed Length")

        visual_box = layout.box()
        visual_box.label(text="Visual Style:")
        visual_box.prop(style, "dim_font_path", text="Font File")
        visual_box.prop(style, "dim_text_color", text="Text Color")




classes = (
    DimStyleItem,
    VIEW3D_UL_DimStyles,
    OT_DimStyleAdd,
    OT_DimStyleRemove,
    OT_DimAssignActiveStyle,
    OT_SketchupProDim,
    VIEW3D_PT_ProDim,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.dim_styles = bpy.props.CollectionProperty(type=DimStyleItem)
    bpy.types.Scene.dim_active_style_index = bpy.props.IntProperty(name="Active Style Index", default=0)

    for scene in bpy.data.scenes:
        ensure_default_style(scene)

    if not bpy.app.timers.is_registered(auto_cleanup_dim_data):
        bpy.app.timers.register(auto_cleanup_dim_data)


def unregister():
    if bpy.app.timers.is_registered(auto_cleanup_dim_data):
        bpy.app.timers.unregister(auto_cleanup_dim_data)

    if getattr(OT_SketchupProDim, "_handle", None):
        bpy.types.SpaceView3D.draw_handler_remove(OT_SketchupProDim._handle, 'WINDOW')
        OT_SketchupProDim._handle = None
    OT_SketchupProDim._is_running = False
    remove_preview_text()

    del bpy.types.Scene.dim_styles
    del bpy.types.Scene.dim_active_style_index

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()







