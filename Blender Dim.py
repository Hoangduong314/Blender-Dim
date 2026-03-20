bl_info = {
    "name": "Pro 3D Dimension Tool",
    "author": "Pro CAD User",
    "version": (1, 1),
    "blender": (3, 0, 0),
    "location": "View3D > UI (N Panel) > Tool > Pro Dim",
    "description": "Công cụ đo kích thước 3D chuyên nghiệp chuẩn CAD/Sketchup",
    "warning": "",
    "doc_url": "",
    "category": "3D View",
}

import bpy
import gpu
import blf
import math
import mathutils
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
from mathutils import Vector, Euler, Matrix

try:
    SHADER = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
except ValueError:
    SHADER = gpu.shader.from_builtin('UNIFORM_COLOR')

COLLECTION_NAME = "Dimensions"

# ===============================================================
# TIẾN TRÌNH DỌN RÁC BỘ NHỚ (AUTO CLEANUP)
# ===============================================================
def auto_cleanup_dim_data():
    active_cols = set(obj.instance_collection for obj in bpy.data.objects if obj.instance_collection)
    
    for col in bpy.data.collections:
        if col.name.startswith("DimData_") and col not in active_cols:
            for obj in col.objects:
                if obj.data:
                    data = obj.data
                    bpy.data.objects.remove(obj, do_unlink=True)
                    if data.__class__.__name__ == 'Mesh': bpy.data.meshes.remove(data)
                    elif data.__class__.__name__ == 'TextCurve': bpy.data.curves.remove(data)
            bpy.data.collections.remove(col)
    return 2.0 

# ===============================================================
# QUẢN LÝ FONT CHỮ VÀ ĐƠN VỊ
# ===============================================================
def get_or_load_font(filepath):
    if not filepath or str(filepath).strip() == "":
        return bpy.data.fonts.get("Bfont") 
    
    abs_path = bpy.path.abspath(filepath)
    for f in bpy.data.fonts:
        if f.filepath == abs_path or f.filepath == filepath:
            return f
    try:
        return bpy.data.fonts.load(abs_path)
    except:
        return bpy.data.fonts.get("Bfont")

def get_formatted_text(dist_m, context):
    unit_choice = context.scene.dim_unit
    show_suffix = context.scene.dim_show_suffix
    prec = context.scene.dim_precision
    
    if unit_choice == 'AUTO':
        sys = context.scene.unit_settings.system
        l_unit = context.scene.unit_settings.length_unit
        if sys == 'IMPERIAL':
            unit_choice = 'in' if l_unit == 'INCHES' else 'ft'
        else:
            if l_unit == 'MILLIMETERS': unit_choice = 'mm'
            elif l_unit == 'CENTIMETERS': unit_choice = 'cm'
            else: unit_choice = 'm'
            
    val = dist_m
    s_str = ""
    if unit_choice == 'cm': val *= 100.0; s_str = "cm"
    elif unit_choice == 'mm': val *= 1000.0; s_str = "mm"
    elif unit_choice == 'ft': val *= 3.2808399; s_str = "ft"
    elif unit_choice == 'in': val *= 39.3700787; s_str = "in"
    else: s_str = "m"
    
    format_str = f"{{:.{prec}f}}"
    result = format_str.format(val)
    if show_suffix: result += s_str
    return result

# ===============================================================
# TẠO VẬT LIỆU SHADELESS ĐẢM BẢO HIỂN THỊ TRONG SHADING
# ===============================================================
def setup_shadeless_mat(mat_name, color, strength):
    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    mat.diffuse_color = color 
    
    has_alpha = len(color) == 4 and color[3] < 1.0
    
    if has_alpha:
        if hasattr(mat, "blend_method"): mat.blend_method = 'BLEND'
        if hasattr(mat, "shadow_method"): mat.shadow_method = 'NONE'
    else:
        if hasattr(mat, "blend_method"): mat.blend_method = 'OPAQUE'
        if hasattr(mat, "shadow_method"): mat.shadow_method = 'NONE'
        
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear() 
    
    out_node = nodes.new('ShaderNodeOutputMaterial')
    out_node.location = (300, 0)
    
    if has_alpha:
        mix_node = nodes.new('ShaderNodeMixShader')
        mix_node.inputs[0].default_value = color[3] 
        
        transp_node = nodes.new('ShaderNodeBsdfTransparent')
        emiss_node = nodes.new('ShaderNodeEmission')
        emiss_node.inputs[0].default_value = (color[0], color[1], color[2], 1.0)
        emiss_node.inputs[1].default_value = strength 
        
        links.new(transp_node.outputs[0], mix_node.inputs[1])
        links.new(emiss_node.outputs[0], mix_node.inputs[2])
        links.new(mix_node.outputs[0], out_node.inputs[0])
    else:
        emiss_node = nodes.new('ShaderNodeEmission')
        emiss_node.inputs[0].default_value = color
        emiss_node.inputs[1].default_value = strength 
        links.new(emiss_node.outputs[0], out_node.inputs[0])
        
    return mat

# ===============================================================
# CẬP NHẬT TỰ ĐỘNG GIAO DIỆN
# ===============================================================
def update_all_dimensions(self, context):
    scale_x = context.scene.dim_scale_x
    t_size = (context.scene.dim_text_size_mm / 1000.0) * scale_x
    t_gap = (context.scene.dim_text_gap_mm / 1000.0) * scale_x
    overshoot = (context.scene.dim_ext_overshoot_mm / 1000.0) * scale_x
    use_fixed = context.scene.dim_ext_use_fixed
    fixed_len = (context.scene.dim_ext_fixed_len_mm / 1000.0) * scale_x
    custom_font = get_or_load_font(context.scene.dim_font_path)

    for obj in bpy.data.objects:
        if obj.get("is_dim_instance"):
            p1 = Vector(obj["p1"])
            p2 = Vector(obj["p2"])
            offset_dir = Vector(obj["offset_dir"])
            offset_dist = obj["offset_dist"]
            Y = Vector(obj["Y_axis"])
            Z = Vector(obj["Z_axis"])

            dist_raw = (p1 - p2).length
            text_str = get_formatted_text(dist_raw, context)

            d1 = p1 + offset_dir * offset_dist
            d2 = p2 + offset_dir * offset_dist

            p1_loc = Vector((0, 0, 0))
            p2_loc = p2 - p1
            d1_loc = d1 - p1
            d2_loc = d2 - p1

            if use_fixed:
                start1_loc = d1_loc - offset_dir * fixed_len
                start2_loc = d2_loc - offset_dir * fixed_len
            else:
                start1_loc = p1_loc
                start2_loc = p2_loc

            d1_ext_loc = d1_loc + offset_dir * overshoot
            d2_ext_loc = d2_loc + offset_dir * overshoot

            col_data = obj.instance_collection
            if not col_data: continue

            for child in col_data.objects:
                if child.type == 'MESH':
                    if "Extension_Lines" in child.name:
                        child.data.vertices[0].co = start1_loc
                        child.data.vertices[1].co = d1_ext_loc
                        child.data.vertices[2].co = start2_loc
                        child.data.vertices[3].co = d2_ext_loc
                    elif "Dimension_Line" in child.name:
                        child.data.vertices[0].co = d1_loc
                        child.data.vertices[1].co = d2_loc
                elif child.type == 'FONT' and "Dimension_Text" in child.name:
                    child.data.body = text_str
                    child.data.size = t_size
                    if custom_font: child.data.font = custom_font
                    child.data.align_y = 'BOTTOM'
                    z_lift = max(dist_raw * 0.002, 0.001)
                    child.location = ((d1_loc + d2_loc) / 2) + Y * t_gap + Z * z_lift

def update_dim_colors(self, context):
    c_line = context.scene.dim_color
    c_text = context.scene.dim_text_color
    strength = context.scene.dim_brightness
    
    setup_shadeless_mat("Mat_Dim", c_line, strength)
    setup_shadeless_mat("Mat_Dim_Text", c_text, strength)
    setup_shadeless_mat("Mat_Ext", (c_line[0], c_line[1], c_line[2], 0.8), strength)

# ===============================================================
# TẠO VẬT THỂ THẬT
# ===============================================================
def get_or_create_main_collection():
    if COLLECTION_NAME not in bpy.data.collections:
        new_col = bpy.data.collections.new(COLLECTION_NAME)
        bpy.context.scene.collection.children.link(new_col)
    return bpy.data.collections[COLLECTION_NAME]

def create_real_dimension(data, context):
    p1, p2, d1, d2 = data['p1'], data['p2'], data['d1'], data['d2']
    offset_dir = data['offset_dir'].normalized()
    offset_dist = (d1 - p1).dot(offset_dir)
    
    scale_x = context.scene.dim_scale_x
    t_size = (context.scene.dim_text_size_mm / 1000.0) * scale_x
    t_gap = (context.scene.dim_text_gap_mm / 1000.0) * scale_x
    overshoot = (context.scene.dim_ext_overshoot_mm / 1000.0) * scale_x
    use_fixed = context.scene.dim_ext_use_fixed
    fixed_len = (context.scene.dim_ext_fixed_len_mm / 1000.0) * scale_x
    custom_font = get_or_load_font(context.scene.dim_font_path)
    
    dist_raw = (p1 - p2).length
    text_str = get_formatted_text(dist_raw, context)
    
    col_data = bpy.data.collections.new(f"DimData_{dist_raw:.3f}_{bpy.context.scene.frame_current}")
    
    c_line = context.scene.dim_color
    c_text = context.scene.dim_text_color
    strength = context.scene.dim_brightness
    
    mat_ext = setup_shadeless_mat("Mat_Ext", (c_line[0], c_line[1], c_line[2], 0.8), strength)
    mat_dim = setup_shadeless_mat("Mat_Dim", c_line, strength)
    mat_text = setup_shadeless_mat("Mat_Dim_Text", c_text, strength)
    
    p1_loc = Vector((0, 0, 0))
    p2_loc = p2 - p1
    d1_loc = d1 - p1
    d2_loc = d2 - p1
    
    if use_fixed:
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
    col_data.objects.link(obj_ext); obj_ext.data.materials.append(mat_ext)

    mesh_dim = bpy.data.meshes.new("Dimension_Mesh")
    mesh_dim.from_pydata([d1_loc, d2_loc], [(0, 1)], [])
    obj_dim = bpy.data.objects.new("Dimension_Line", mesh_dim)
    col_data.objects.link(obj_dim); obj_dim.data.materials.append(mat_dim)

    x_line = (p2 - p1).normalized()          
    v_normal = x_line.cross(offset_dir).normalized() 

    rv3d = context.space_data.region_3d
    view_rot = rv3d.view_rotation
    view_fwd = view_rot @ Vector((0, 0, -1)) 
    view_right = view_rot @ Vector((1, 0, 0)) 

    if v_normal.dot(view_fwd) < 0: Z = v_normal
    else: Z = -v_normal

    if x_line.dot(view_right) > 0: X = x_line
    else: X = -x_line

    Y = Z.cross(X).normalized()

    align_y = 'BOTTOM'

    rot_matrix = Matrix((X, Y, Z)).transposed()
    plane_rotation = rot_matrix.to_euler('XYZ')

    mid_point_loc = (d1_loc + d2_loc) / 2
    z_lift = max(dist_raw * 0.002, 0.001)

    font_data = bpy.data.curves.new(name="DimText_Data", type='FONT')
    font_data.body = text_str
    font_data.size = t_size
    if custom_font: font_data.font = custom_font
    font_data.align_x = 'CENTER'
    font_data.align_y = align_y
    
    obj_text = bpy.data.objects.new("Dimension_Text", font_data)
    col_data.objects.link(obj_text); obj_text.data.materials.append(mat_text)

    obj_text.location = mid_point_loc + Y * t_gap + Z * z_lift
    obj_text.rotation_euler = plane_rotation

    main_col = get_or_create_main_collection()
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
    instance_obj["Y_axis"] = Y
    instance_obj["Z_axis"] = Z
    
    main_col.objects.link(instance_obj)

# ===============================================================
# OPERATOR CHÍNH
# ===============================================================
class OT_SketchupProDim(bpy.types.Operator):
    bl_idname = "view3d.sketchup_pro_dim"
    bl_label = "Pro 3D Dim Tool"
    bl_options = {'REGISTER', 'UNDO'}

    _is_running = False 
    _handle = None
    data = {'step': 0, 'snap_loc': None, 'p1': None, 'p2': None, 'd1': None, 'd2': None, 'offset_dir': None, 'snap_color': (1.0, 0.0, 0.0, 1.0)}

    @classmethod
    def poll(cls, context):
        return context.area.type == 'VIEW_3D'

    def update_proxy_text(self, context):
        preview_text = bpy.data.objects.get("Preview_Dim_Text")
        if not preview_text: return
        
        cls_data = self.__class__.data
        if cls_data['step'] == 2 and cls_data['d1']:
            preview_text.hide_viewport = False
            
            p1, p2 = cls_data['p1'], cls_data['p2']
            dist = (p1 - p2).length
            
            scale_x = context.scene.dim_scale_x
            t_size = (context.scene.dim_text_size_mm / 1000.0) * scale_x
            t_gap = (context.scene.dim_text_gap_mm / 1000.0) * scale_x
            custom_font = get_or_load_font(context.scene.dim_font_path)
            
            preview_text.data.body = get_formatted_text(dist, context)
            preview_text.data.size = t_size
            if custom_font: preview_text.data.font = custom_font
            
            x_line = (p2 - p1).normalized()          
            offset_dir = cls_data['offset_dir'].normalized()         
            v_normal = x_line.cross(offset_dir).normalized() 

            rv3d = context.space_data.region_3d
            view_rot = rv3d.view_rotation
            view_fwd = view_rot @ Vector((0, 0, -1))
            view_right = view_rot @ Vector((1, 0, 0))

            if v_normal.dot(view_fwd) < 0: Z = v_normal
            else: Z = -v_normal

            if x_line.dot(view_right) > 0: X = x_line
            else: X = -x_line

            Y = Z.cross(X).normalized()

            align_y = 'BOTTOM'

            preview_text.data.align_y = align_y
            rot_matrix = Matrix((X, Y, Z)).transposed()
            
            mid_point = (cls_data['d1'] + cls_data['d2']) / 2
            z_lift = max(dist * 0.002, 0.001)
            
            preview_text.rotation_euler = rot_matrix.to_euler('XYZ')
            preview_text.location = mid_point + Y * t_gap + Z * z_lift
        else:
            preview_text.hide_viewport = True

    def modal(self, context, event):
        nav_events = {
            'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 
            'NUMPAD_1', 'NUMPAD_2', 'NUMPAD_3', 'NUMPAD_4', 'NUMPAD_5', 
            'NUMPAD_6', 'NUMPAD_7', 'NUMPAD_8', 'NUMPAD_9', 'NUMPAD_0',
            'TRACKPADPAN', 'TRACKPADZOOM'
        }
        if event.type in nav_events:
            return {'PASS_THROUGH'}

        cls_data = self.__class__.data

        if event.type == 'MOUSEMOVE':
            self.mouse_pos = Vector((event.mouse_region_x, event.mouse_region_y))
            if cls_data['step'] < 2:
                cls_data['snap_loc'] = self.get_snap_location(context, event)
            elif cls_data['step'] == 2:
                self.calculate_combined_offset(context)
                self.update_proxy_text(context)
            context.area.tag_redraw()

        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if cls_data['step'] == 0 and cls_data['snap_loc']:
                cls_data['p1'] = cls_data['snap_loc'].copy()
                cls_data['step'] = 1
                context.area.header_text_set("Step 2: Pick End Point")
            
            elif cls_data['step'] == 1 and cls_data['snap_loc']:
                cls_data['p2'] = cls_data['snap_loc'].copy()
                cls_data['step'] = 2
                context.area.header_text_set("Step 3: Move to set Direction & Distance. Click to finish.")
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

    def get_snap_location(self, context, event):
        scene = context.scene; region = context.region; rv3d = context.space_data.region_3d
        coord = (event.mouse_region_x, event.mouse_region_y)
        view_vec = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        result, location, norm, idx, obj, mat = scene.ray_cast(context.view_layer.depsgraph, ray_origin, view_vec)
        
        if not result: return None
        if obj and obj.type == 'MESH':
            mesh = obj.data; poly = mesh.polygons[idx]
            min_dist_2d = 9999; closest_v_world = location; mouse_2d = Vector(coord)
            for v_idx in poly.vertices:
                v_world = mat @ mesh.vertices[v_idx].co
                v_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, v_world)
                if v_2d:
                    dist_2d = (v_2d - mouse_2d).length
                    if dist_2d < 30 and dist_2d < min_dist_2d:
                        min_dist_2d = dist_2d; closest_v_world = v_world
            if min_dist_2d < 30: return closest_v_world
        return location

    def calculate_combined_offset(self, context):
        cls_data = self.__class__.data
        p1, p2 = cls_data['p1'], cls_data['p2']
        if not p1 or not p2: return
        
        v_line = (p2 - p1).normalized()
        if (p2 - p1).length < 0.0001: return
        
        region = context.region; rv3d = context.space_data.region_3d
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, self.mouse_pos)
        ray_dir = view3d_utils.region_2d_to_vector_3d(region, rv3d, self.mouse_pos)
        
        pt_mouse = view3d_utils.region_2d_to_location_3d(region, rv3d, self.mouse_pos, p1)
        if not pt_mouse: return
        v_raw = pt_mouse - p1
        v_perp = v_raw - v_raw.dot(v_line) * v_line
        if v_perp.length < 0.001: return
        
        free_dir = v_perp.normalized()
        best_dir = free_dir
        snap_color = context.scene.dim_color
        final_dist = v_perp.length
        
        axes = {
            'X': (Vector((1,0,0)), (1.0, 0.2, 0.2, 1.0)),
            'Y': (Vector((0,1,0)), (0.2, 1.0, 0.2, 1.0)),
            'Z': (Vector((0,0,1)), (0.2, 0.6, 1.0, 1.0))
        }
        
        min_angle = math.radians(15) 
        
        for axis_name, (axis_vec, color) in axes.items():
            a_perp = axis_vec - axis_vec.dot(v_line) * v_line
            if a_perp.length > 0.001:
                snap_dir = a_perp.normalized()
                if free_dir.dot(snap_dir) < 0:
                    snap_dir = -snap_dir
                
                angle = free_dir.angle(snap_dir)
                if angle < min_angle:
                    min_angle = angle
                    best_dir = snap_dir
                    snap_color = color
                    
                    plane_normal = v_line.cross(best_dir).normalized()
                    pt_intersect = mathutils.geometry.intersect_line_plane(ray_origin, ray_origin + ray_dir, p1, plane_normal)
                    if pt_intersect:
                        v_snap_raw = pt_intersect - p1
                        final_dist = v_snap_raw.dot(best_dir)
                    break
        
        cls_data['offset_dir'] = best_dir
        cls_data['snap_color'] = snap_color
        cls_data['d1'] = p1 + best_dir * final_dist
        cls_data['d2'] = p2 + best_dir * final_dist

    def draw_callback_px(self, context):
        if context.area is None: return
        cls_data = self.__class__.data; step = cls_data['step']
        region = context.region; rv3d = context.space_data.region_3d

        if step < 2 and cls_data['snap_loc']:
            snap_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, cls_data['snap_loc'])
            if snap_2d:
                s = 6
                b = batch_for_shader(SHADER, 'LINES', {"pos": [(snap_2d.x-s, snap_2d.y), (snap_2d.x+s, snap_2d.y), (snap_2d.x, snap_2d.y-s), (snap_2d.x, snap_2d.y+s)]})
                SHADER.bind(); SHADER.uniform_float("color", (1,0,0,1)); b.draw(SHADER)

        if step == 2 and cls_data['d1'] and cls_data['d2']:
            scale_x = context.scene.dim_scale_x
            overshoot = (context.scene.dim_ext_overshoot_mm / 1000.0) * scale_x
            fixed_len = (context.scene.dim_ext_fixed_len_mm / 1000.0) * scale_x
            use_fixed = context.scene.dim_ext_use_fixed
            
            offset_dir = cls_data['offset_dir']
            d1 = cls_data['d1']
            d2 = cls_data['d2']
            
            if use_fixed:
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
            
            if start1_2d and start2_2d and d1_2d and d2_2d and d1_ext_2d and d2_ext_2d:
                preview_color = cls_data['snap_color']
                b = batch_for_shader(SHADER, 'LINES', {"pos": [start1_2d, d1_ext_2d, start2_2d, d2_ext_2d, d1_2d, d2_2d]})
                SHADER.bind(); SHADER.uniform_float("color", preview_color); b.draw(SHADER)

    def invoke(self, context, event):
        if context.area.type == 'VIEW_3D':
            if self.__class__._is_running: return {'CANCELLED'}
            self.__class__._is_running = True
            self.__class__.data = {'step': 0, 'snap_loc': None, 'p1': None, 'p2': None, 'd1': None, 'd2': None, 'offset_dir': None, 'snap_color': context.scene.dim_color}
            
            old_preview = bpy.data.objects.get("Preview_Dim_Text")
            if old_preview: bpy.data.objects.remove(old_preview, do_unlink=True)
            
            preview_font = bpy.data.curves.new(name="Preview_Dim_Font", type='FONT')
            preview_font.align_x = 'CENTER'
            preview_font.align_y = 'BOTTOM' 
            preview_text = bpy.data.objects.new("Preview_Dim_Text", preview_font)
            context.collection.objects.link(preview_text)
            
            mat_text = setup_shadeless_mat("Mat_Dim_Text_Preview", context.scene.dim_text_color, context.scene.dim_brightness)
            preview_text.data.materials.append(mat_text)
            
            preview_text.hide_viewport = True
            
            context.area.header_text_set("Step 1: Pick Start Point")
            self.__class__._handle = bpy.types.SpaceView3D.draw_handler_add(self.draw_callback_px, (context,), 'WINDOW', 'POST_PIXEL')
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        return {'CANCELLED'}

    def stop_ui(self, context):
        self.__class__._is_running = False
        context.area.header_text_set(None) 
        
        preview_text = bpy.data.objects.get("Preview_Dim_Text")
        if preview_text: bpy.data.objects.remove(preview_text, do_unlink=True)

        if getattr(self.__class__, "_handle", None):
            bpy.types.SpaceView3D.draw_handler_remove(self.__class__._handle, 'WINDOW')
            self.__class__._handle = None
        context.area.tag_redraw()

# --- PANEL UI ---
class VIEW3D_PT_ProDim(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'; bl_region_type = 'UI'; bl_category = 'Tool'; bl_label = 'Pro Dim'

    def draw(self, context):
        layout = self.layout
        
        box = layout.box()
        box.label(text="Unit Settings:")
        box.prop(context.scene, "dim_unit", text="Unit")
        row = box.row()
        row.prop(context.scene, "dim_show_suffix", text="Show Suffix")
        row.prop(context.scene, "dim_precision", text="Decimals")
        
        layout.separator()
        
        box = layout.box()
        box.label(text="Annotative Scale:")
        row = box.row()
        row.label(text="Scale 1 :")
        row.prop(context.scene, "dim_scale_x", text="")

        box = layout.box()
        box.label(text="Paper Sizes (mm):")
        box.prop(context.scene, "dim_text_size_mm", text="Text Size")
        box.prop(context.scene, "dim_text_gap_mm", text="Text Gap")
        box.prop(context.scene, "dim_ext_overshoot_mm", text="Ext Overshoot")
        
        box.prop(context.scene, "dim_ext_use_fixed", text="Fixed Ext Lines")
        if context.scene.dim_ext_use_fixed:
            box.prop(context.scene, "dim_ext_fixed_len_mm", text="Fixed Length")
        
        layout.separator()
        
        box = layout.box()
        box.label(text="Style Settings:")
        box.prop(context.scene, "dim_font_path", text="Font File")
        box.prop(context.scene, "dim_color", text="Line Color")
        box.prop(context.scene, "dim_text_color", text="Text Color")
        box.prop(context.scene, "dim_brightness", text="Emission Strength")
        
        layout.separator()
        
        if OT_SketchupProDim._is_running:
            layout.label(text="Follow Header. ESC to exit", icon='INFO')
        else:
            layout.operator("view3d.sketchup_pro_dim", text="Start Dimming", icon='DRIVER_DISTANCE')

classes = (OT_SketchupProDim, VIEW3D_PT_ProDim)

def register():
    bpy.types.Scene.dim_unit = bpy.props.EnumProperty(
        name="Unit",
        items=[('AUTO', "Scene Unit", ""), ('m', "Meters (m)", ""), ('cm', "Centimeters (cm)", ""), ('mm', "Millimeters (mm)", ""), ('ft', "Feet (ft)", ""), ('in', "Inches (in)", "")],
        default='AUTO', update=update_all_dimensions
    )
    bpy.types.Scene.dim_show_suffix = bpy.props.BoolProperty(name="Show Suffix", default=False, update=update_all_dimensions)
    bpy.types.Scene.dim_precision = bpy.props.IntProperty(name="Decimals", default=0, min=0, max=5, update=update_all_dimensions)
    
    bpy.types.Scene.dim_font_path = bpy.props.StringProperty(
        name="Font Path",
        description="Đường dẫn đến file Font (.ttf, .otf)",
        subtype='FILE_PATH',
        update=update_all_dimensions
    )
    
    bpy.types.Scene.dim_color = bpy.props.FloatVectorProperty(name="Line Color", subtype='COLOR', size=4, min=0.0, max=1.0, default=(1.0, 0.0, 0.0, 1.0), update=update_dim_colors)
    bpy.types.Scene.dim_text_color = bpy.props.FloatVectorProperty(name="Text Color", subtype='COLOR', size=4, min=0.0, max=1.0, default=(0.0, 0.0, 0.0, 1.0), update=update_dim_colors)
    bpy.types.Scene.dim_brightness = bpy.props.FloatProperty(name="Emission Strength", default=3.0, min=1.0, max=50.0, update=update_dim_colors)
    
    bpy.types.Scene.dim_scale_x = bpy.props.FloatProperty(name="Scale", default=100.0, min=1.0, update=update_all_dimensions)
    bpy.types.Scene.dim_text_size_mm = bpy.props.FloatProperty(name="Text Size", default=2.0, min=0.1, update=update_all_dimensions)
    bpy.types.Scene.dim_text_gap_mm = bpy.props.FloatProperty(name="Text Gap", default=0.5, min=0.0, update=update_all_dimensions)
    bpy.types.Scene.dim_ext_overshoot_mm = bpy.props.FloatProperty(name="Overshoot", default=0.0, min=0.0, update=update_all_dimensions)
    
    bpy.types.Scene.dim_ext_use_fixed = bpy.props.BoolProperty(name="Fixed Ext Lines", default=True, update=update_all_dimensions)
    bpy.types.Scene.dim_ext_fixed_len_mm = bpy.props.FloatProperty(name="Fixed Length", default=4.0, min=1.0, update=update_all_dimensions)
    
    if not bpy.app.timers.is_registered(auto_cleanup_dim_data):
        bpy.app.timers.register(auto_cleanup_dim_data)

    for cls in classes: bpy.utils.register_class(cls)

def unregister():
    if bpy.app.timers.is_registered(auto_cleanup_dim_data):
        bpy.app.timers.unregister(auto_cleanup_dim_data)
        
    del bpy.types.Scene.dim_unit
    del bpy.types.Scene.dim_show_suffix
    del bpy.types.Scene.dim_precision
    del bpy.types.Scene.dim_font_path
    del bpy.types.Scene.dim_color
    del bpy.types.Scene.dim_text_color
    del bpy.types.Scene.dim_brightness
    del bpy.types.Scene.dim_scale_x
    del bpy.types.Scene.dim_text_size_mm
    del bpy.types.Scene.dim_text_gap_mm
    del bpy.types.Scene.dim_ext_overshoot_mm
    del bpy.types.Scene.dim_ext_use_fixed
    del bpy.types.Scene.dim_ext_fixed_len_mm
    for cls in reversed(classes): bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()