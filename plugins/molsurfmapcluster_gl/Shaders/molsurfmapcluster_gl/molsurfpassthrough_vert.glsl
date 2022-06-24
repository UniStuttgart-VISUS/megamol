layout(std430, binding = 11) buffer vertexData
{
    vec4 values[];
};

uniform mat4 mvp = mat4(1.0);
uniform bool coloredmode = false;

out vec4 overridecolor;

void main()
{
    vec4 in_position;
    if(!coloredmode) {
        in_position = values[gl_VertexID];
    } else {
        in_position = values[gl_VertexID * 2];
        overridecolor = values[(gl_VertexID * 2) + 1];
    }
    gl_Position = mvp * in_position;
}
