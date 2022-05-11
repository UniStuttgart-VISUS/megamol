#version 460

uniform mat4 mvp;
uniform vec2 lower_left;
uniform vec2 upper_right;

out vec2 texcoord;

void main(void) {
    const vec4 vertices[6] = vec4[6](vec4( lower_left, 0.0, 0.0 ),
                                     vec4( upper_right, 1.0 ,1.0 ),
                                     vec4( lower_left.x, upper_right.y, 0.0, 1.0 ),
                                     vec4( upper_right, 1.0, 1.0 ),
                                     vec4( lower_left, 0.0, 0.0 ),
                                     vec4( upper_right.x, lower_left.y, 1.0, 0.0 ) );

    vec4 vertex = vertices[gl_VertexID];
    
    texcoord = vertex.zw;
    gl_Position = mvp * vec4(vertex.xy, 0.0, 1.0);
}