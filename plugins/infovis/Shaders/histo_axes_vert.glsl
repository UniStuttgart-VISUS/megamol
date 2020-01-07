uniform mat4 modelView = mat4(1.0);
uniform mat4 projection = mat4(1.0);
uniform vec2 colTotalSize = vec2(1.0, 1.0);
uniform vec2 colDrawSize = vec2(1.0, 1.0);
uniform vec2 colDrawOffset = vec2(1.0, 1.0);
uniform int mode = 0;

void main()
{
    if (mode == 0) {
        float posX = colTotalSize.x * gl_InstanceID + colDrawOffset.x + colDrawSize.x * gl_VertexID;
        gl_Position = projection * modelView * vec4(posX, colDrawOffset.y, 0.0, 1.0);
    } else {
        gl_Position = projection * modelView * vec4(colDrawOffset.x, colDrawOffset.y + colDrawSize.y * gl_VertexID, 0.0, 1.0);
    }
}
