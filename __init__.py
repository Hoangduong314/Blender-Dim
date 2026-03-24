bl_info = {
    "name": "Pro 3D Dimension Tool",
    "author": "Pro CAD User",
    "version": (1, 2, 1),
    "blender": (3, 0, 0),
    "location": "View3D > UI (N Panel) > Pro Dim",
    "description": "A professional 3D dimensioning tool for Blender",
    "category": "3D View",
}

if "bpy" in locals():
    import importlib
    importlib.reload(constants)
    importlib.reload(properties)
    importlib.reload(utils)
    importlib.reload(operators)
    importlib.reload(ui)
else:
    from . import constants, properties, utils, operators, ui

import bpy
from bpy.app.handlers import persistent

classes = (
    properties.DimStyleItem,
    ui.VIEW3D_UL_DimStyles,
    operators.OT_DimStyleAdd,
    operators.OT_DimStyleRemove,
    operators.OT_DimAssignActiveStyle,
    operators.OT_SketchupProDim,
    operators.OT_EditWitnessLines,
    operators.OT_EditDimLine,
    ui.VIEW3D_PT_ProDim,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.dim_styles = bpy.props.CollectionProperty(type=properties.DimStyleItem)
    bpy.types.Scene.dim_active_style_index = bpy.props.IntProperty(name="Active Style Index", default=0)

    if utils.auto_cleanup_dim_data_handler not in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.append(utils.auto_cleanup_dim_data_handler)

def unregister():
    if utils.auto_cleanup_dim_data_handler in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(utils.auto_cleanup_dim_data_handler)

    if getattr(operators.OT_SketchupProDim, "_handle", None):
        bpy.types.SpaceView3D.draw_handler_remove(operators.OT_SketchupProDim._handle, 'WINDOW')
        operators.OT_SketchupProDim._handle = None
    operators.OT_SketchupProDim._is_running = False
    utils.remove_preview_text()

    if hasattr(bpy.types.Scene, "dim_styles"):
        del bpy.types.Scene.dim_styles
    if hasattr(bpy.types.Scene, "dim_active_style_index"):
        del bpy.types.Scene.dim_active_style_index

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
