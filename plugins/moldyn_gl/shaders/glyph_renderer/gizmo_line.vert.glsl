#version 430

#include "glyph_renderer/inc/uniforms.inc.glsl"
#include "glyph_renderer/inc/options.inc.glsl"
#include "glyph_renderer/inc/ssbo_data.inc.glsl"
#include "glyph_renderer/inc/flags.inc.glsl"
#include "core/tflookup.inc.glsl"
#include "core/bitflags.inc.glsl"
#include "glyph_renderer/inc/quaternion_to_matrix.inc.glsl"
#include "glyph_renderer/inc/cube_geometry.inc.glsl"
#include "glyph_renderer/inc/compute_color.inc.glsl"
#include "glyph_renderer/inc/gizmo_line_vert.inc.glsl"
