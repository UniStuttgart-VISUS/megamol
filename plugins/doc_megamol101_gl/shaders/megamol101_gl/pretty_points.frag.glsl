#version 400

out layout(location = 0) vec4 frag_color;

in vec4 myColor;
in float radius;
in vec2 texCoords;
flat in vec4 pos;

uniform vec3 camDir;
uniform vec3 camPos;

uniform mat4 mvp;
uniform mat4 view;
uniform mat4 model = mat4(1.0);
uniform mat4 invModel = mat4(1.0);
uniform vec4 lightParams = vec4(0.2, 0.8, 0.4, 10.0);
uniform vec3 lightPos = vec3(-50.0, 50.0, 20.0);

void main() {
    frag_color = myColor;
    //frag_color = vec4(0,1,0,1);
    vec4 newCol = myColor;

    vec2 middle = vec2(0.5, 0.5);
    if (length(middle - texCoords) > 0.5) {
        discard;
        frag_color = vec4(1.0, 1.0, 0.0, 1.0);
        return;
    }

    vec2 help = texCoords - middle;
    help *= 2.0;

    vec3 cPos = vec3(invModel * vec4(camPos, 1.0));

    // surface normal (view space)
    vec3 n = normalize(vec3(help.xy, sqrt(1 - help.x * help.x - help.y * help.y)));
    // light position vector (view space)
    vec4 l = vec4(lightPos, 1.0);
    // surface point position (world space)
    vec3 nPos = pos.xyz - normalize(pos.xyz - cPos) * radius * n.z;
    // light direction (view space)
    vec3 lightDir = normalize(l.xyz - (view * model * vec4(nPos.xyz, 1.0)).xyz);

    // calculate correct depth
    vec4 h = mvp * vec4(nPos, 1.0);
    float md = h.z / h.w;
    gl_FragDepth = 0.5 * md + 0.5; // magic

    // lighting calculations (Ambient Light + Lambert)
    float nDOTl = dot(n, lightDir);
    vec3 r = normalize(2.0 * nDOTl * n - lightDir);
    frag_color = lightParams.x * newCol; // ambient
    frag_color += lightParams.y * newCol * max(nDOTl, 0.0); // diffuse
    // no specular light (i.e., no highlights; often too distracting)
    // color values must be within 0..1:
    frag_color = clamp(frag_color, 0.0, 1.0);
    frag_color.w = myColor.w;
}
