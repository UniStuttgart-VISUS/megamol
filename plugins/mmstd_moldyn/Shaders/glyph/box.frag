in vec4 wsPos;
in vec4 vertColor;

in vec3 invRad;

in flat vec3 dirColor;

in flat vec3 normal;
in flat vec3 transformedNormal;

//layout (location = 0) out vec4 out_frag_color;
out layout(location = 0) vec3 albedo_out;
out layout(location = 1) vec3 normal_out;
out layout(location = 2) float depth_out;

void main() {
    //vec3 normal = vec3(1,0,0);
    //normal = rotMatIT * normal;

    //vec3 color = mix(dirColor, vertColor.rgb, colorInterpolation);
    albedo_out = mix(dirColor, vertColor.rgb, colorInterpolation);
    normal_out = transformedNormal;
    depth_out = gl_FragCoord.z;
    //vec3 ray = normalize(wsPos.xyz - cam.xyz);
    //out_frag_color = vec4(LocalLighting(ray, transformedNormal, light.xyz, color), 1.0);
}