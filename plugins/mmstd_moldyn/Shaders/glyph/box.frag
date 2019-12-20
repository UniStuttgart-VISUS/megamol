in vec4 wsPos;
in vec4 vertColor;

in vec3 invRad;

in flat vec3 dirColor;

in flat vec3 normal;
in flat vec3 transformedNormal;
in vec3 viewRay;

//layout (location = 0) out vec4 out_frag_color;
out layout(location = 0) vec3 albedo_out;
out layout(location = 1) vec3 normal_out;
out layout(location = 2) float depth_out;

void main() {
    depth_out = gl_FragCoord.z;
    albedo_out = mix(dirColor, vertColor.rgb, colorInterpolation);
    normal_out = transformedNormal;
}