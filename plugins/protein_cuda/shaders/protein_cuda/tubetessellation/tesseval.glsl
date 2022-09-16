#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_EXT_gpu_shader4 : require

layout(quads, equal_spacing, cw) in;

in int id[];
in vec4 myColor[];
out vec4 vertColor;

uniform bool interpolateColors = false;
uniform float tubewidth = 0.001;
uniform float structurewidth = 0.003;

struct CAType {
    float x, y, z;
    int type;
};

layout(std430, binding = 2) buffer shader_data {
    CAType cAlphas[];
};

void main() {
    CAType alph0 = cAlphas[id[0] - 1];
    CAType alph1 = cAlphas[id[0]];
    CAType alph2 = cAlphas[id[1]];
    CAType alph3 = cAlphas[id[1] + 1];

    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    vec4 p2 = gl_in[2].gl_Position;
    vec4 p3 = gl_in[3].gl_Position;

    vertColor = myColor[1];

    vec4 colors[4];

    for(int i = 0; i < 4; i++) {
        int mytype = cAlphas[id[0] + i - 1].type;
        if(mytype == 1) // beta sheet
        {
            colors[i] = vec4(0, 0, 1, 1);
        }
        else if(mytype == 2) // alpha helix
        {
            colors[i] = vec4(1, 0, 0, 1);
        }
        else if(mytype == 3) // turn
        {
            colors[i] = vec4(0, 1, 0, 1);
        }
        else // unclassified
        {
            colors[i] = vec4(0, 0, 0, 1);
        }
    }

    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;

    // Catmull-Rom Spline
    //gl_Position = 0.5 *( (2.0 * p1) + (-p0 + p2) * u + ( 2.0 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1- 3 * p2 + p3) * u*u*u);

    // Cubic B-Spline
    u += 3;
    float q = ( u - 1.0) / 3.0;
    vec4 d10 = p0 * ( 1.0 - q) + p1 * q;
    float q1 = ( u - 2.0) / 3.0;
    vec4 d11 =  p1 * ( 1.0 - q1) + p2 * q1;
    float q2 = ( u - 3.0) / 3.0;
    vec4 d12 =  p2 * ( 1.0 - q2) + p3 * q2;

    float q3 = ( u - 2.0) / 2.0;
    vec4 d20 = d10 * ( 1.0 - q3) + d11 * q3;
    float q4 = ( u - 3.0) / 2.0;
    vec4 d21 = d11 * ( 1.0 - q4) + d12 * q4;

    float q5 = ( u - 3.0);
    vec4 myPosition = d20 * ( 1.0 - q5) + d21 * q5;

    if(interpolateColors)
    {
        vec4 c10 = colors[0] * (1.0 - q) + colors[1] * q;
        vec4 c11 = colors[1] * (1.0 - q1) + colors[2] * q1;
        vec4 c12 = colors[2] * (1.0 - q2) + colors[3] * q2;

        vec4 c20 = c10 * (1.0 - q3) + c11 * q3;
        vec4 c21 = c11 * (1.0 - q4) + c12 * q4;

        vertColor = c20 * (1.0 - q5) + c21 * q5;
    }
    else
    {
        vertColor = colors[2];
    }

    vec2 tangent = normalize(d21.xy - d20.xy); // direction of the backbone
    vec2 normal = vec2(-tangent.y, tangent.x); // orthogonal to the backbone

    int mytype = cAlphas[id[1]].type;
    int mytype2 = cAlphas[id[1] + 1].type;

    bool change = false;
    int where = 3;
    int lasttype = mytype2;

    for(int i = 3; i > -1; i = i - 1 ) {
        int mytype = cAlphas[id[0] + i - 1].type;
        if(mytype != lasttype)
        {
            change = true;
            where = i;
        }
        lasttype = mytype;
    }

    vec2 left = myPosition.xy + normal * tubewidth;
    vec2 right = myPosition.xy - normal * tubewidth;

    if(mytype != 0) {
        left = myPosition.xy + normal * structurewidth;
        right = myPosition.xy - normal * structurewidth;
    }

    if(change && (where == 2) && (mytype == 1)) { // arrow heads
        u = gl_TessCoord.x;
        float factor = mix(structurewidth * 2.5, tubewidth, u);
        left = myPosition.xy + normal * factor;
        right = myPosition.xy - normal * factor;
    }

    myPosition.xy = mix(left, right, v);

    gl_Position =  myPosition;
}
