#version 430

uniform mat4 modelView = mat4(1.0);
uniform mat4 projection = mat4(1.0);
uniform vec2 componentTotalSize = vec2(1.0, 1.0);
uniform vec2 componentDrawSize = vec2(1.0, 1.0);
uniform vec2 componentDrawOffset = vec2(1.0, 1.0);
uniform int mode = 0;

void main() {
    if (mode == 0) {
        float posX = componentTotalSize.x * gl_InstanceID + componentDrawOffset.x + componentDrawSize.x * gl_VertexID;
        gl_Position = projection * modelView * vec4(posX, componentDrawOffset.y, 0.0, 1.0);
    } else {
        gl_Position = projection * modelView * vec4(componentDrawOffset.x, componentDrawOffset.y + componentDrawSize.y * gl_VertexID, 0.0, 1.0);
    }
}
