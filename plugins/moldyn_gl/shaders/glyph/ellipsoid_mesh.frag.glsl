#version 450

#include "glyph/uniforms.glsl"

#include "glyph/options.glsl"

in Point {
    vec4 objPos;
    vec3 camPos;
    vec4 vertColor;

    vec3 invRad;

    flat vec3 dirColor;

    mat3 rotate_world_into_tensor;
    mat3 rotate_points;
}
pp;

out layout(location = 0) vec4 albedo_out;
out layout(location = 1) vec3 normal_out;
out layout(location = 2) float depth_out;

void main() {
    vec4 coord;
    vec3 ray, tmp;
    float lambda;

    // transform fragment coordinates from window coordinates to view coordinates.
    coord = gl_FragCoord * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0) + vec4(-1.0, -1.0, -1.0, 1.0);


    // transform fragment coordinates from view coordinates to object coordinates.
    coord = MVP_I * coord;
    coord /= coord.w;
    coord -= pp.objPos; // ... and move
    // ... and rotate mit rotMat transposed
    ray = pp.rotate_world_into_tensor * coord.xyz;

    ray *= pp.invRad;

    // calc the viewing ray
    ray = normalize(ray - pp.camPos.xyz);
    // calculate the geometry-ray-intersection (sphere with radius = 1)
    float d1 = -dot(pp.camPos.xyz, ray);                  // projected length of the cam-sphere-vector onto the ray
    float d2s = dot(pp.camPos.xyz, pp.camPos.xyz) - d1 * d1; // off axis of cam-sphere-vector and ray
    float radicand = 1.0 - d2s;                        // square of difference of projected length and lambda
    if (radicand < 0.0) {
        discard;
    }
    lambda = d1 - sqrt(radicand);                        // lambda
    vec3 sphereintersection = lambda * ray + pp.camPos.xyz; // intersection point


    // "calc" normal at intersection point
    vec3 normal = sphereintersection;
    vec3 color = pp.vertColor.rgb;
    vec3 s = step(normal, vec3(0));
    color = s;

    normal = pp.rotate_points * normal;
    ray = pp.rotate_points * ray;

    tmp = sphereintersection / pp.invRad;

    //sphereintersection.x = dot(rotMatT0, tmp.xyz);
    //sphereintersection.y = dot(rotMatT1, tmp.xyz);
    //sphereintersection.z = dot(rotMatT2, tmp.xyz);
    sphereintersection = pp.rotate_world_into_tensor * tmp.xyz;

    // phong lighting with directional light, colors are hardcoded as they're inaccessible for some reason
    //out_frag_color = vec4(LocalLighting(ray, normal, lightPos.xyz, color), 1.0);
    //out_frag_color = vec4(LocalLighting(ray, normalize(sphereintersection), lightPos.xyz, color), 1.0);

    albedo_out = vec4(mix(pp.dirColor, pp.vertColor.rgb, colorInterpolation), 1.0);
    normal_out = normal;

    sphereintersection += pp.objPos.xyz;

    vec4 Ding = vec4(sphereintersection, 1.0);
    float depth = dot(MVP_T[2], Ding);
    float depthW = dot(MVP_T[3], Ding);
    depth_out = ((depth / depthW) + 1.0) * 0.5;

    //depth_out = gl_FragCoord.z;
    //albedo_out = mix(dirColor, vertColor.rgb, colorInterpolation);
    //normal_out = transformedNormal;
}
