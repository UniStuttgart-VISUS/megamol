#ifdef INTERLEAVED

struct SphereParams {

#ifdef COL_LOWER_VERT
#include "sphere_renderer/inc/make_color_declaration_string.inc.glsl"
#include "sphere_renderer/inc/make_vertex_declaration_string.inc.glsl"
#else // COL_LOWER_VERT
#include "sphere_renderer/inc/make_vertex_declaration_string.inc.glsl"
#include "sphere_renderer/inc/make_color_declaration_string.inc.glsl"
#endif // COL_LOWER_VERT

};

layout(SSBO_GENERATED_SHADER_ALIGNMENT, binding = SSBO_VERTEX_BINDING_POINT) buffer shader_data {
    SphereParams theBuffer[];
};

#else // INTERLEAVED

struct SpherePosParams {
#include "sphere_renderer/inc/make_vertex_declaration_string.inc.glsl"
};

struct SphereColParams {
#include "sphere_renderer/inc/make_color_declaration_string.inc.glsl"
};

layout(SSBO_GENERATED_SHADER_ALIGNMENT, binding = SSBO_VERTEX_BINDING_POINT) buffer shader_data {
    SpherePosParams thePosBuffer[];
};

layout(SSBO_GENERATED_SHADER_ALIGNMENT, binding = SSBO_COLOR_BINDING_POINT) buffer shader_data2 {
    SphereColParams theColBuffer[];
};

#endif // INTERLEAVED
