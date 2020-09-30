uniform mat4 modelViewProjection;
uniform vec2 mouse = vec2(0, 0);
uniform float pickRadius = 0.1;

smooth out vec2 circleCoord;

void main()
{
    vec2 vertices[6] = {
        // b_l, b_r, t_r
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        // t_r, t_l, b_l
        vec2(1.0, 1.0), vec2(-1.0, 1.0), vec2(-1.0, -1.0)
    };

    circleCoord = vertices[gl_VertexID];

    vec4 vertex = vec4(mouse + pickRadius * circleCoord, 0.0f, 1.0f);

    gl_Position = modelViewProjection * vertex;
}
