import bpy


from .utils import *
from .operators import *

class VIEW3D_UL_DimStyles(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.prop(item, "name", text="", emboss=False, icon='MATERIAL')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon='MATERIAL')


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
            
            if context.active_object and context.active_object.get("is_dim_instance"):
                layout.operator("view3d.edit_witness_lines", text="Edit Witness Lines", icon='GREASEPENCIL')
                layout.operator("view3d.edit_dim_line", text="Edit Dim Line", icon='DRIVER_TRANSFORM')

            row = layout.row(align=True)
            row.operator("view3d.update_dim_anchors", text="Update Selected Dims", icon='FILE_REFRESH').update_all = False
            row.operator("view3d.update_dim_anchors", text="Update All Dims", icon='FILE_REFRESH').update_all = True
            
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




