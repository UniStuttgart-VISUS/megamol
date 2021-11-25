#version 430

#include "protein_gl/simplemolecule/sm_common_defines.glsl"
#include "protein_gl/simplemolecule/sm_common_input_vert.glsl"

out vec3 move_pos;

void main(void) {
    gl_Position = MVP * vert_position;
    move_color = vec4(vert_color, 1);
    move_pos = vert_position.xyz;

    if(applyFiltering) {
        if(vert_filter == 0) {
            gl_PointSize = 0;
        }
    }
}
