in vec4 wsPos;
in vec4 vertColor;

in vec3 invRad;

in flat vec3 dirColor;

in flat vec3 normal;
in flat vec3 transformedNormal;

layout (location = 0) out vec4 out_frag_color;

void main() {
    //vec3 normal = vec3(1,0,0);
    //normal = rotMatIT * normal;

    vec3 color = mix(dirColor, vertColor.rgb, colorInterpolation);

    vec3 ray = wsPos.xyz - cam.xyz;
    //color = transformedNormal;
    out_frag_color = vec4(color, 1.0);
    //out_frag_color = vec4(LocalLighting(ray, transformedNormal, lightPos.xyz, color), 1.0);
}