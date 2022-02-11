uniform float color_interpolation;
uniform float scaling;
uniform float radius_scaling;
uniform float min_radius;
uniform uint options;

uniform vec2 tf_range;

uniform vec4 cam;
uniform vec4 global_color;
uniform vec4 view_attr;
uniform vec4 clip_data;

uniform mat4 mvp;
uniform mat4 mvp_i;
uniform mat4 mvp_t;

uniform sampler1D tf_texture;

// arrow vertex
uniform float length_filter;

uniform vec4 light_dir;

uniform mat4 mv_i;
