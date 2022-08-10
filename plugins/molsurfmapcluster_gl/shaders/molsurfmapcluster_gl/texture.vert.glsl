#version 460

uniform vec2 lowerleft = vec2(0.0, 0.0);
uniform vec2 upperright = vec2(1.0, 1.0);

uniform mat4 mvp = mat4(1.0);

out vec2 texCoord;

void main() {
    const vec2 vertices[6] = vec2[6](vec2(0.0,0.0),
                                     vec2(1.0,1.0),
                                     vec2(0.0,1.0),
                                     vec2(1.0,1.0),
                                     vec2(0.0,0.0),
                                     vec2(1.0,0.0));

    vec2 vertex = vertices[gl_VertexID];
    
    texCoord = vertex.xy;
    float x_pos = mix(lowerleft.x, upperright.x, vertex.x);
    float y_pos = mix(lowerleft.y, upperright.y, vertex.y);
    gl_Position = mvp * vec4(x_pos, y_pos, -1.0, 1.0);
}
