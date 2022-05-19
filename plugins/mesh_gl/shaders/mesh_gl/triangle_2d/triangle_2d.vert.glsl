#version 330

layout(location = 0) in vec2 in_position;
layout(location = 1) in float in_value;
layout(location = 2) in float in_mask;

uniform mat4 model_view_matrix;
uniform mat4 projection_matrix;

uniform float min_value;
uniform float max_value;

uniform vec4 mask_color;

uniform sampler1D transfer_function;

out vec4 vertex_color;

void main() {
    gl_Position = projection_matrix * model_view_matrix * vec4(in_position, 0.0f, 1.0f);
    vertex_color = in_mask == 1.0f ?
        texture(transfer_function, (min_value == max_value) ? 0.5f : ((in_value - min_value) / (max_value - min_value)))
        : mask_color;
}
