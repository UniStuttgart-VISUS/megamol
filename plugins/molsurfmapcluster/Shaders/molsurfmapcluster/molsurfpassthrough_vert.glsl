layout(std430, binding = 11) buffer vertexData
{
    vec3 positions[];
};

uniform mat4 mvp = mat4(1.0);

void main()
{
    vec3 in_position = positions[gl_VertexID];
    gl_Position = mvp * vec4(in_position, 1.0);
}