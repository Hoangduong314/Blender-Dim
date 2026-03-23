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
from .properties import *
from .utils import *

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

    edit_mode: bpy.props.BoolProperty(default=False, options={'HIDDEN'})
    edit_dim_line_mode: bpy.props.BoolProperty(default=False, options={'HIDDEN'})

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
        'offset_dist': None,
        'snap_color': (1.0, 0.0, 0.0, 1.0),
        'p2_constraint': None,
        'snap_cache': None,
        'dim_line_snap_cache': None,
        'offset_snap_point': None,
        'chain_mode': False,
        'chain_style_id': None,
        'chain_offset_dir': None,
        'chain_offset_dist': None,
        'chain_line_dir': None,
        'editing_dim_line': False,
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
        cls_data = self.__class__.data
        step = cls_data['step']
        if step == 0:
            context.area.header_text_set("Step 1: Pick Start Point")
        elif step == 1:
            label = self.get_constraint_label()
            if cls_data.get('chain_mode'):
                context.area.header_text_set(f"Chain: Pick Next Point | Constraint: {label} | X/Y/Z = Axis, Shift+X/Y/Z = Plane | ESC = Exit")
            else:
                context.area.header_text_set(f"Step 2: Pick End Point | Constraint: {label} | X/Y/Z = Axis, Shift+X/Y/Z = Plane")
        elif step == 2:
            context.area.header_text_set("Step 3: Move to set Direction & Distance. Click to finish.")

    def reset_preview_geometry(self):
        cls_data = self.__class__.data
        cls_data['p2'] = None
        cls_data['d1'] = None
        cls_data['d2'] = None
        cls_data['offset_dist'] = None
        cls_data['offset_snap_point'] = None

    def get_chain_projected_point(self, base_point, candidate_point):
        cls_data = self.__class__.data
        line_dir = cls_data.get('chain_line_dir')
        if base_point is None or candidate_point is None or line_dir is None:
            return candidate_point

        line_dir = line_dir.normalized()
        return base_point + line_dir * (candidate_point - base_point).dot(line_dir)

    def begin_chain_mode(self, context, anchor_point, style_id, offset_dir, offset_dist, line_dir=None):
        cls_data = self.__class__.data
        cls_data['chain_mode'] = True
        cls_data['chain_style_id'] = style_id
        cls_data['chain_offset_dir'] = offset_dir.normalized().copy()
        cls_data['chain_offset_dist'] = offset_dist
        if line_dir is not None and line_dir.length > 0.0001:
            cls_data['chain_line_dir'] = line_dir.normalized().copy()
        cls_data['step'] = 1
        cls_data['p1'] = anchor_point.copy()
        cls_data['snap_loc'] = None
        cls_data['snap_loc_raw'] = None
        cls_data['p2_constraint'] = None
        self.reset_preview_geometry()
        self.update_step_header(context)

    def refresh_chain_preview(self, context):
        cls_data = self.__class__.data
        p1 = cls_data.get('p1')
        p2 = self.get_chain_projected_point(p1, cls_data.get('snap_loc'))
        offset_dir = cls_data.get('chain_offset_dir')
        offset_dist = cls_data.get('chain_offset_dist')

        self.reset_preview_geometry()
        if p1 is None or p2 is None or offset_dir is None or offset_dist is None:
            return
        if (p2 - p1).length <= 0.0001:
            return

        style = get_style_by_id(context.scene, cls_data.get('chain_style_id'))
        cls_data['p2'] = p2.copy()
        cls_data['offset_dir'] = offset_dir.copy()
        cls_data['snap_color'] = tuple(style.dim_text_color)
        cls_data['d1'] = p1 + offset_dir * offset_dist
        cls_data['d2'] = p2 + offset_dir * offset_dist

    def build_dimension_payload(self, p1, p2, offset_dir, offset_dist, style_id):
        return {
            'p1': p1.copy(),
            'p2': p2.copy(),
            'd1': p1 + offset_dir * offset_dist,
            'd2': p2 + offset_dir * offset_dist,
            'offset_dir': offset_dir.copy(),
            'style_id': style_id,
        }

    def get_dimension_split_candidate(self, context, point):
        cls_data = self.__class__.data
        chain_offset_dir = cls_data.get('chain_offset_dir')
        chain_offset_dist = cls_data.get('chain_offset_dist')
        chain_style_id = cls_data.get('chain_style_id')
        if point is None or chain_offset_dir is None or chain_offset_dist is None:
            return None, -1

        best_obj = None
        best_dist = None
        best_idx = -1
        tolerance = 0.001
        chain_offset_dir = chain_offset_dir.normalized()
        import json

        for obj in context.visible_objects:
            if obj.hide_get() or not obj.get("is_dim_instance") or obj.get("style_id") != chain_style_id:
                continue

            if "points_json" in obj:
                points = [Vector(p) for p in json.loads(obj["points_json"])]
            else:
                if not obj.get("p1") or not obj.get("p2"): continue
                points = [Vector(obj["p1"]), Vector(obj["p2"])]
                
            offset_dir = Vector(obj.get("offset_dir"))
            offset_dist = obj.get("offset_dist")
            
            if abs(offset_dir.dot(chain_offset_dir)) < 0.9999 or abs(offset_dist - chain_offset_dist) > tolerance:
                continue

            for i in range(len(points)-1):
                p1, p2 = points[i], points[i+1]
                segment = p2 - p1
                if segment.length_squared < 0.0001: continue
                factor = (point - p1).dot(segment) / segment.length_squared
                if factor <= 0.0 or factor >= 1.0: continue
                closest = p1.lerp(p2, factor)
                dist = (closest - point).length
                if dist > tolerance: continue
                margin = min(0.001, segment.length * 0.25)
                if (point - p1).length <= margin or (point - p2).length <= margin: continue
                
                if best_dist is None or dist < best_dist:
                    best_obj = obj
                    best_dist = dist
                    best_idx = i + 1
                    
        return best_obj, best_idx

    def get_dimension_merge_candidates(self, context, point):
        cls_data = self.__class__.data
        chain_offset_dir = cls_data.get('chain_offset_dir')
        chain_offset_dist = cls_data.get('chain_offset_dist')
        chain_style_id = cls_data.get('chain_style_id')
        chain_line_dir = cls_data.get('chain_line_dir')
        if point is None or chain_offset_dir is None or chain_offset_dist is None or chain_line_dir is None:
            return None, -1

        tolerance = 0.001
        chain_offset_dir = chain_offset_dir.normalized()
        chain_line_dir = chain_line_dir.normalized()
        import json

        for obj in context.visible_objects:
            if obj.hide_get() or not obj.get("is_dim_instance") or obj.get("style_id") != chain_style_id:
                continue

            if "points_json" in obj:
                points = [Vector(p) for p in json.loads(obj["points_json"])]
            else:
                if not obj.get("p1") or not obj.get("p2"): continue
                points = [Vector(obj["p1"]), Vector(obj["p2"])]
                
            offset_dir = Vector(obj.get("offset_dir"))
            offset_dist = obj.get("offset_dist")
            
            if abs(offset_dir.dot(chain_offset_dir)) < 0.9999 or abs(offset_dist - chain_offset_dist) > tolerance:
                continue

            for i, p in enumerate(points):
                if abs((point - p).dot(chain_line_dir)) < tolerance:
                    return obj, i
                    
        return None, -1

    def try_merge_dimension(self, context, point):
        target_obj, target_idx = self.get_dimension_merge_candidates(context, point)
        if not target_obj: return False, None
        
        import json
        points = [Vector(p) for p in json.loads(target_obj["points_json"])]
        points.pop(target_idx)
        
        if len(points) < 2:
            remove_dimension_instance(target_obj)
            self.clear_snap_cache()
            return True, None
            
        data = {
            'points': points,
            'offset_dir': Vector(target_obj["offset_dir"]),
            'offset_dist': target_obj["offset_dist"],
            'style_id': target_obj["style_id"],
        }
        create_real_dimension(data, context, existing_instance=target_obj)
        self.clear_snap_cache()
        return True, target_obj

    def try_split_dimension(self, context, point):
        split_obj, insert_idx = self.get_dimension_split_candidate(context, point)
        if split_obj is None: return False, []

        import json
        if "points_json" in split_obj:
            points = [Vector(p) for p in json.loads(split_obj["points_json"])]
        else:
            points = [Vector(split_obj["p1"]), Vector(split_obj["p2"])]
            
        points.insert(insert_idx, point)
        
        data = {
            'points': points,
            'offset_dir': Vector(split_obj["offset_dir"]),
            'offset_dist': split_obj["offset_dist"],
            'style_id': split_obj["style_id"],
        }
        create_real_dimension(data, context, existing_instance=split_obj)
        self.clear_snap_cache()
        return True, [split_obj]

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
        snap_elements = set(tool_settings.snap_elements) if tool_settings.use_snap else set()
        snap_target = getattr(tool_settings, "snap_target", 'CLOSEST')
        return tool_settings.use_snap, snap_elements, snap_target

    def ensure_snap_cache(self, context):
        cls_data = self.__class__.data
        if cls_data.get('snap_cache') is not None:
            return cls_data['snap_cache']

        depsgraph = context.evaluated_depsgraph_get()
        snap_cache = []
        for obj in context.visible_objects:
            if obj.type != 'MESH' or obj.hide_get():
                continue

            obj_eval = obj.evaluated_get(depsgraph)
            try:
                mesh = obj_eval.to_mesh()
            except RuntimeError:
                continue

            if mesh is None or len(mesh.vertices) == 0:
                obj_eval.to_mesh_clear()
                continue

            world_matrix = obj_eval.matrix_world.copy()
            verts = [world_matrix @ vert.co for vert in mesh.vertices]
            edges = [tuple(edge.vertices) for edge in mesh.edges]
            snap_cache.append({
                'verts': verts,
                'edges': edges,
            })
            obj_eval.to_mesh_clear()

        cls_data['snap_cache'] = snap_cache
        return snap_cache

    def clear_snap_cache(self):
        self.__class__.data['snap_cache'] = None
        self.__class__.data['dim_line_snap_cache'] = None

    def ensure_dim_line_snap_cache(self, context):
        cls_data = self.__class__.data
        if cls_data.get('dim_line_snap_cache') is not None:
            return cls_data['dim_line_snap_cache']

        dim_line_cache = []
        import json
        for obj in context.visible_objects:
            if obj.hide_get() or not obj.get("is_dim_instance"):
                continue

            if "points_json" in obj:
                points = [Vector(p) for p in json.loads(obj["points_json"])]
            else:
                p1 = obj.get("p1")
                p2 = obj.get("p2")
                if p1 is None or p2 is None: continue
                points = [Vector(p1), Vector(p2)]

            offset_dir = obj.get("offset_dir")
            offset_dist = obj.get("offset_dist")
            if offset_dir is None or offset_dist is None:
                continue

            offset_dir = Vector(offset_dir)
            if offset_dir.length <= 0.0001:
                continue

            offset_dir.normalize()
            
            for i in range(len(points) - 1):
                p1_seg = points[i]
                p2_seg = points[i+1]
                d1 = p1_seg + offset_dir * offset_dist
                d2 = p2_seg + offset_dir * offset_dist
                if (d2 - d1).length <= 0.0001:
                    continue

                dim_line_cache.append({
                    'd1': d1,
                    'd2': d2,
                })

        cls_data['dim_line_snap_cache'] = dim_line_cache
        return dim_line_cache

    def closest_point_on_segment_2d(self, point, a, b):
        ab = b - a
        length_sq = ab.length_squared
        if length_sq == 0.0:
            return a.copy(), 0.0
        t = max(0.0, min(1.0, (point - a).dot(ab) / length_sq))
        return a + ab * t, t

    def get_vertex_snap_candidate(self, region, rv3d, snap_cache, mouse_2d, threshold):
        best_loc = None
        best_dist = threshold
        for entry in snap_cache:
            for world_loc in entry['verts']:
                screen_loc = view3d_utils.location_3d_to_region_2d(region, rv3d, world_loc)
                if screen_loc is None:
                    continue
                dist_2d = (screen_loc - mouse_2d).length
                if dist_2d < best_dist:
                    best_loc = world_loc.copy()
                    best_dist = dist_2d
        return best_loc, best_dist

    def get_edge_snap_candidate(self, region, rv3d, snap_cache, mouse_2d, threshold, midpoint_only=False):
        best_loc = None
        best_dist = threshold
        for entry in snap_cache:
            verts = entry['verts']
            for v1_idx, v2_idx in entry['edges']:
                v1_world = verts[v1_idx]
                v2_world = verts[v2_idx]
                v1_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, v1_world)
                v2_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, v2_world)
                if v1_2d is None or v2_2d is None:
                    continue

                if midpoint_only:
                    midpoint_world = (v1_world + v2_world) * 0.5
                    midpoint_2d = (v1_2d + v2_2d) * 0.5
                    dist_2d = (midpoint_2d - mouse_2d).length
                    if dist_2d < best_dist:
                        best_loc = midpoint_world
                        best_dist = dist_2d
                    continue

                closest_2d, factor = self.closest_point_on_segment_2d(mouse_2d, v1_2d, v2_2d)
                dist_2d = (closest_2d - mouse_2d).length
                if dist_2d < best_dist:
                    best_loc = v1_world.lerp(v2_world, factor)
                    best_dist = dist_2d
        return best_loc, best_dist

    def get_face_snap_candidate(self, context, coord):
        scene = context.scene
        region = context.region
        rv3d = context.space_data.region_3d
        view_vec = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        result, location, _norm, _idx, _obj, _mat = scene.ray_cast(context.view_layer.depsgraph, ray_origin, view_vec)
        if not result:
            return None
        return location

    def get_dim_line_snap_candidate(self, region, rv3d, dim_line_cache, mouse_2d, threshold):
        best_loc = None
        best_dist = threshold
        for entry in dim_line_cache:
            d1_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, entry['d1'])
            d2_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, entry['d2'])
            if d1_2d is None or d2_2d is None:
                continue
            closest_2d, factor = self.closest_point_on_segment_2d(mouse_2d, d1_2d, d2_2d)
            dist_2d = (closest_2d - mouse_2d).length
            if dist_2d < best_dist:
                best_loc = entry['d1'].lerp(entry['d2'], factor)
                best_dist = dist_2d
        return best_loc, best_dist
    def get_raw_snap_location(self, context, event):
        region = context.region
        rv3d = context.space_data.region_3d
        coord = (event.mouse_region_x, event.mouse_region_y)
        mouse_2d = Vector(coord)
        use_snap, snap_elements, _snap_target = self.get_snap_settings(context)

        face_loc = self.get_face_snap_candidate(context, coord)
        if not use_snap or not snap_elements:
            return face_loc

        snap_cache = self.ensure_snap_cache(context)
        threshold = 18.0
        best_loc = None
        best_dist = threshold

        if 'VERTEX' in snap_elements:
            vertex_loc, vertex_dist = self.get_vertex_snap_candidate(region, rv3d, snap_cache, mouse_2d, threshold)
            if vertex_loc is not None and vertex_dist < best_dist:
                best_loc = vertex_loc
                best_dist = vertex_dist

        if 'EDGE_MIDPOINT' in snap_elements:
            midpoint_loc, midpoint_dist = self.get_edge_snap_candidate(region, rv3d, snap_cache, mouse_2d, threshold, midpoint_only=True)
            if midpoint_loc is not None and midpoint_dist < best_dist:
                best_loc = midpoint_loc
                best_dist = midpoint_dist

        if 'EDGE' in snap_elements:
            edge_loc, edge_dist = self.get_edge_snap_candidate(region, rv3d, snap_cache, mouse_2d, threshold)
            if edge_loc is not None and edge_dist < best_dist:
                best_loc = edge_loc
                best_dist = edge_dist

        if best_loc is not None:
            return best_loc

        if 'FACE' in snap_elements or 'FACE_NEAREST' in snap_elements:
            return face_loc

        return face_loc

    def update_proxy_text(self, context):
        preview_text = bpy.data.objects.get("Preview_Dim_Text")
        if not preview_text:
            return

        cls_data = self.__class__.data
        if (cls_data['step'] != 2 and not cls_data.get('chain_mode')) or not cls_data['d1']:
            preview_text.hide_viewport = True
            return

        scene = context.scene
        style = get_style_by_id(scene, cls_data.get('chain_style_id')) if cls_data.get('chain_mode') else get_active_style(scene)
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
        set_object_material(preview_text, get_style_materials(style)["preview"])

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
                if cls_data.get('chain_mode'):
                    self.refresh_chain_preview(context)
                    self.update_proxy_text(context)
            self.update_step_header(context)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE':
            self.mouse_pos = Vector((event.mouse_region_x, event.mouse_region_y))
            if cls_data['step'] < 2:
                raw_loc = self.get_raw_snap_location(context, event)
                cls_data['snap_loc_raw'] = raw_loc
                cls_data['snap_loc'] = self.apply_p2_constraint(raw_loc) if cls_data['step'] == 1 else raw_loc
                if cls_data.get('chain_mode') and cls_data['step'] == 1:
                    self.refresh_chain_preview(context)
                    self.update_proxy_text(context)
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
                if cls_data.get('chain_mode'):
                    next_point = self.get_chain_projected_point(cls_data['p1'], cls_data['snap_loc'])
                    if (next_point - cls_data['p1']).length <= 0.0001:
                        chain_inst = cls_data.get('chain_instance')
                        if chain_inst and "points_json" in chain_inst:
                            import json
                            pts = [Vector(p) for p in json.loads(chain_inst["points_json"])]
                            if len(pts) > 2:
                                pts.pop()
                                cls_data['p1'] = pts[-1].copy()
                                data = {
                                    'points': pts,
                                    'offset_dir': cls_data['chain_offset_dir'],
                                    'offset_dist': cls_data['chain_offset_dist'],
                                    'style_id': cls_data['chain_style_id'],
                                }
                                create_real_dimension(data, context, existing_instance=chain_inst)
                            else:
                                remove_dimension_instance(chain_inst)
                                cls_data['chain_instance'] = None
                                cls_data['chain_mode'] = False
                                cls_data['step'] = 0
                                cls_data['p1'] = None
                                cls_data['p2'] = None
                                cls_data['snap_loc_raw'] = None
                                cls_data['p2_constraint'] = None
                        self.reset_preview_geometry()
                        self.clear_snap_cache()
                        self.update_step_header(context)
                        self.update_proxy_text(context)
                        context.area.tag_redraw()
                        return {'RUNNING_MODAL'}
                        
                    is_merged, merged_dim = self.try_merge_dimension(context, next_point)
                    if is_merged:
                        if hasattr(bpy.ops.ed, 'undo_push'):
                            bpy.ops.ed.undo_push(message="Merge Chain Dimension")
                        if (next_point - cls_data['p1']).length <= 0.0001:
                            chain_inst = cls_data.get('chain_instance')
                            if chain_inst and "points_json" in chain_inst:
                                import json
                                pts = [Vector(p) for p in json.loads(chain_inst["points_json"])]
                                if len(pts) > 0:
                                    cls_data['p1'] = pts[-1].copy()
                            else:
                                cls_data['chain_instance'] = None
                                cls_data['chain_mode'] = False
                                cls_data['step'] = 0
                                cls_data['p1'] = None
                                cls_data['p2'] = None
                                cls_data['snap_loc_raw'] = None
                                cls_data['p2_constraint'] = None
                        self.reset_preview_geometry()
                        self.clear_snap_cache()
                        self.update_step_header(context)
                        self.update_proxy_text(context)
                        context.area.tag_redraw()
                        return {'RUNNING_MODAL'}
                    is_split, split_dim = self.try_split_dimension(context, next_point)
                    if is_split:
                        if hasattr(bpy.ops.ed, 'undo_push'):
                            bpy.ops.ed.undo_push(message="Split Chain Dimension")
                        self.reset_preview_geometry()
                        self.clear_snap_cache()
                        self.update_step_header(context)
                        self.update_proxy_text(context)
                        context.area.tag_redraw()
                        return {'RUNNING_MODAL'}
                        
                    chain_inst = cls_data.get('chain_instance')
                    if chain_inst and "points_json" in chain_inst:
                        import json
                        pts = [Vector(p) for p in json.loads(chain_inst["points_json"])]
                        pts.append(next_point)
                        data = {
                            'points': pts,
                            'offset_dir': cls_data['chain_offset_dir'],
                            'offset_dist': cls_data['chain_offset_dist'],
                            'style_id': cls_data['chain_style_id'],
                        }
                        create_real_dimension(data, context, existing_instance=chain_inst)
                    else:
                        new_dim = create_real_dimension({
                            'points': [cls_data['p1'], next_point],
                            'offset_dir': cls_data['chain_offset_dir'],
                            'offset_dist': cls_data['chain_offset_dist'],
                            'style_id': cls_data['chain_style_id']
                        }, context)
                        cls_data['chain_instance'] = new_dim
                    self.begin_chain_mode(
                        context,
                        next_point,
                        cls_data['chain_style_id'],
                        cls_data['chain_offset_dir'],
                        cls_data['chain_offset_dist'],
                        cls_data.get('chain_line_dir'),
                    )
                    self.update_proxy_text(context)
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

                cls_data['p2'] = cls_data['snap_loc'].copy()
                cls_data['step'] = 2
                self.update_step_header(context)
                self.calculate_combined_offset(context)
                self.update_proxy_text(context)

            elif cls_data['step'] == 2:
                if cls_data.get('editing_dim_line'):
                    obj = cls_data.get('chain_instance')
                    if obj and "points_json" in obj:
                        import json
                        pts = [Vector(p) for p in json.loads(obj["points_json"])]
                        style_id = obj.get("style_id", get_active_style(context.scene).style_id)
                        data = {
                            'points': pts,
                            'offset_dir': cls_data['offset_dir'],
                            'offset_dist': cls_data['offset_dist'],
                            'style_id': style_id,
                        }
                        create_real_dimension(data, context, existing_instance=obj)
                        if hasattr(bpy.ops.ed, 'undo_push'):
                            bpy.ops.ed.undo_push(message="Edit Dimension Line")
                    self.stop_ui(context)
                    return {'FINISHED'}

                style_id = get_active_style(context.scene).style_id
                
                first_dim = create_real_dimension({
                    'points': [cls_data['p1'], cls_data['p2']],
                    'offset_dir': cls_data['offset_dir'],
                    'offset_dist': cls_data['offset_dist'],
                    'style_id': style_id,
                }, context)
                cls_data['chain_instance'] = first_dim
                
                self.clear_snap_cache()
                if hasattr(bpy.ops.ed, 'undo_push'):
                    bpy.ops.ed.undo_push(message="Add Dimension")
                cls_data['chain_history_anchors'] = [cls_data['p1'].copy(), cls_data['p2'].copy()]
                cls_data['chain_history_dims'] = [first_dim]
                self.begin_chain_mode(
                    context,
                    cls_data['p2'],
                    style_id,
                    cls_data['offset_dir'],
                    cls_data['offset_dist'],
                    (cls_data['p2'] - cls_data['p1']).normalized(),
                )
                self.update_proxy_text(context)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

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
        offset_snap_point = None
        offset_snap_color = (1.0, 0.75, 0.2, 1.0)

        dim_line_cache = self.ensure_dim_line_snap_cache(context)
        dim_snap_loc, _dim_snap_dist = self.get_dim_line_snap_candidate(
            region,
            rv3d,
            dim_line_cache,
            self.mouse_pos,
            18.0,
        )

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

        if dim_snap_loc is not None:
            final_dist = (dim_snap_loc - p1).dot(best_dir)
            offset_snap_point = dim_snap_loc
            snap_color = offset_snap_color

        cls_data['offset_dir'] = best_dir
        cls_data['offset_dist'] = final_dist
        cls_data['snap_color'] = snap_color
        cls_data['offset_snap_point'] = offset_snap_point
        cls_data['d1'] = p1 + best_dir * final_dist
        cls_data['d2'] = p2 + best_dir * final_dist

    def draw_callback_px(self, context):
        if context.area is None:
            return

        cls_data = self.__class__.data
        step = cls_data['step']
        region = context.region
        rv3d = context.space_data.region_3d

        marker_loc = cls_data['snap_loc']
        if step == 1 and cls_data.get('p2_constraint') and cls_data.get('snap_loc_raw') is not None:
            marker_loc = cls_data['snap_loc_raw']

        if step < 2 and marker_loc:
            snap_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, marker_loc)
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

        if step == 2 and cls_data.get('offset_snap_point') is not None:
            snap_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, cls_data['offset_snap_point'])
            if snap_2d:
                size = 7
                batch = batch_for_shader(
                    SHADER,
                    'LINES',
                    {"pos": [(snap_2d.x - size, snap_2d.y), (snap_2d.x + size, snap_2d.y), (snap_2d.x, snap_2d.y - size), (snap_2d.x, snap_2d.y + size)]},
                )
                SHADER.bind()
                SHADER.uniform_float("color", (1.0, 0.75, 0.2, 1.0))
                batch.draw(SHADER)

        if (step != 2 and not cls_data.get('chain_mode')) or not cls_data['d1'] or not cls_data['d2']:
            return

        style = get_style_by_id(context.scene, cls_data.get('chain_style_id')) if cls_data.get('chain_mode') else get_active_style(context.scene)
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
            'offset_dist': None,
            'snap_color': tuple(active_style.dim_text_color),
            'p2_constraint': None,
            'snap_cache': None,
            'dim_line_snap_cache': None,
            'offset_snap_point': None,
            'chain_mode': False,
            'chain_style_id': None,
            'chain_offset_dir': None,
            'chain_offset_dist': None,
            'chain_line_dir': None,
            'chain_instance': None,
            'editing_dim_line': False,
        }

        if self.edit_mode or self.edit_dim_line_mode:
            obj = context.active_object
            if obj and obj.get("is_dim_instance"):
                import json
                pts = [Vector(p) for p in json.loads(obj["points_json"])]
                if len(pts) > 0:
                    cls_data = self.__class__.data
                    cls_data['chain_mode'] = True
                    cls_data['chain_instance'] = obj
                    cls_data['chain_style_id'] = obj.get("style_id", active_style.style_id)
                    cls_data['chain_offset_dir'] = Vector(obj["offset_dir"])
                    cls_data['chain_offset_dist'] = obj["offset_dist"]
                    
                    if len(pts) > 1:
                        line_dir = (pts[1] - pts[0]).normalized()
                    else:
                        line_dir = Vector((1, 0, 0))
                    
                    cls_data['chain_line_dir'] = line_dir
                    
                    if self.edit_dim_line_mode:
                        cls_data['step'] = 2
                        cls_data['p1'] = pts[0].copy()
                        cls_data['p2'] = pts[1].copy() if len(pts)>1 else pts[0].copy()
                        cls_data['editing_dim_line'] = True
                    else:
                        cls_data['step'] = 1
                        cls_data['p1'] = pts[-1].copy()
                        cls_data['editing_dim_line'] = False

        remove_preview_text()
        preview_font = bpy.data.curves.new(name="Preview_Dim_Font", type='FONT')
        preview_font.align_x = 'CENTER'
        preview_font.align_y = 'BOTTOM'
        preview_text = bpy.data.objects.new("Preview_Dim_Text", preview_font)
        context.collection.objects.link(preview_text)
        set_object_material(preview_text, get_preview_material(context.scene))
        preview_text.hide_viewport = True

        self.mouse_pos = Vector((event.mouse_region_x, event.mouse_region_y))
        self.clear_snap_cache()
        self.update_step_header(context)
        self.__class__._handle = bpy.types.SpaceView3D.draw_handler_add(self.draw_callback_px, (context,), 'WINDOW', 'POST_PIXEL')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def stop_ui(self, context):
        self.__class__._is_running = False
        if context.area:
            context.area.header_text_set(None)

        remove_preview_text()
        self.clear_snap_cache()
        auto_cleanup_dim_data()

        if getattr(self.__class__, "_handle", None):
            bpy.types.SpaceView3D.draw_handler_remove(self.__class__._handle, 'WINDOW')
            self.__class__._handle = None
        if context.area:
            context.area.tag_redraw()

class OT_EditWitnessLines(bpy.types.Operator):
    bl_idname = "view3d.edit_witness_lines"
    bl_label = "Edit Witness Lines"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if context.active_object and context.active_object.get("is_dim_instance"):
            return True
        return False
        
    def execute(self, context):
        bpy.ops.view3d.sketchup_pro_dim('INVOKE_DEFAULT', edit_mode=True)
        return {'FINISHED'}

class OT_EditDimLine(bpy.types.Operator):
    bl_idname = "view3d.edit_dim_line"
    bl_label = "Edit Dim Line"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if context.active_object and context.active_object.get("is_dim_instance"):
            return True
        return False
        
    def execute(self, context):
        bpy.ops.view3d.sketchup_pro_dim('INVOKE_DEFAULT', edit_dim_line_mode=True)
        return {'FINISHED'}

