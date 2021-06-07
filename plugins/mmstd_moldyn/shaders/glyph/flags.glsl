uniform vec4 flag_selected_col;
uniform vec4 flag_softselected_col;
uniform uint flag_offset;

layout(std430, binding = 4) buffer flags {
    uint flagsArray[];
};