import bpy

def update_callback_deferred():
    try:
        from .utils import update_all_dimensions
        update_all_dimensions(None, bpy.context)
    except Exception:
        pass
    return None

def update_callback(self, context):
    if not bpy.app.timers.is_registered(update_callback_deferred):
        bpy.app.timers.register(update_callback_deferred, first_interval=0.01)


class DimStyleItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Style Name", default="Style", update=update_callback)
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
        update=update_callback,
    )
    dim_show_suffix: bpy.props.BoolProperty(name="Show Suffix", default=False, update=update_callback)
    dim_precision: bpy.props.IntProperty(name="Decimals", default=0, min=0, max=5, update=update_callback)
    dim_font_path: bpy.props.StringProperty(
        name="Font Path",
        description="Path to font file (.ttf, .otf)",
        subtype='FILE_PATH',
        update=update_callback,
    )

    dim_text_color: bpy.props.FloatVectorProperty(
        name="Text Color",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0),
        update=update_callback,
    )
    dim_scale_x: bpy.props.FloatProperty(name="Scale", default=100.0, min=1.0, update=update_callback)
    dim_text_size_mm: bpy.props.FloatProperty(name="Text Size", default=2.0, min=0.1, update=update_callback)
    dim_text_gap_mm: bpy.props.FloatProperty(name="Text Gap", default=0.5, min=0.0, update=update_callback)
    dim_ext_overshoot_mm: bpy.props.FloatProperty(name="Overshoot", default=0.0, min=0.0, update=update_callback)
    dim_arrow_style: bpy.props.EnumProperty(
        name="Arrow Style",
        items=[('ARROW', "Arrow", ""), ('TICK', "Tick", "")],
        default='ARROW',
        update=update_callback,
    )
    dim_arrow_size_mm: bpy.props.FloatProperty(name="Arrow Size", default=1.5, min=0.1, update=update_callback)
    dim_ext_use_fixed: bpy.props.BoolProperty(name="Fixed Ext Lines", default=True, update=update_callback)
    dim_ext_fixed_len_mm: bpy.props.FloatProperty(name="Fixed Length", default=4.0, min=1.0, update=update_callback)


