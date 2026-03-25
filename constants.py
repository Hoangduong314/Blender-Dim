import gpu


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
