#version 450

#include "glyph/uniforms.glsl"

#include "glyph/options.glsl"

in Point {
    vec4 objPos;
    vec3 camPos;
    vec4 vertColor;

    vec3 invRad;

    flat vec3 dirColor;

    flat vec3 transformedNormal;

    mat3 rotate_world_into_tensor;
    mat3 rotate_points;
}
pp;

out layout(location = 0) vec4 albedo_out;
out layout(location = 1) vec3 normal_out;
out layout(location = 2) float depth_out;

void main() {
    depth_out = gl_FragCoord.z;
    albedo_out = vec4(mix(pp.dirColor, pp.vertColor.rgb, colorInterpolation), 1.0);
    normal_out = pp.transformedNormal;
}
