uniform mat4 modelView = mat4(1.0);
uniform mat4 projection = mat4(1.0);
uniform float posX = 1.0;
uniform float posY = 1.0;
uniform float width = 1.0;
uniform float height = 1.0;

void main()
{
    vec3 pos = vec3(0.0);
    if (gl_VertexID == 0) { // bottom left
        pos = vec3(0.0, 0.0, 0.0);
    } else if (gl_VertexID == 1) { // bottom right
        pos = vec3(1.0, 0.0, 0.0);
    } else if (gl_VertexID == 2) { // top left
        pos = vec3(0.0, 1.0, 0.0);
    } else if (gl_VertexID == 3) { // top right
        pos = vec3(1.0, 1.0, 0.0);
    }
    pos.x = pos.x * width + posX;
    pos.y = pos.y * height + posY;
    gl_Position = projection * modelView * vec4(pos, 1.0);
}
