uniform mat4 modelView = mat4(1.0);
uniform mat4 projection = mat4(1.0);
uniform float posX = 1.0;
uniform float posY = 1.0;
uniform float width = 1.0;
uniform float height = 1.0;

layout(location = 0) in vec3 in_position;

void main()
{
    vec3 pos = in_position;
    pos.x = pos.x * width + posX;
    pos.y = pos.y * height + posY;
    gl_Position = projection * modelView * vec4(pos, 1.0);
}
