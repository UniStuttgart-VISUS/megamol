#define SSBO_FLAGS_BINDING_POINT 2

uniform uint flags_enabled;
uniform vec4 flag_selected_col;
uniform vec4 flag_softselected_col;
uniform uint flags_offset;

#ifdef FLAGS_AVAILABLE
    layout(std430, binding = SSBO_FLAGS_BINDING_POINT) buffer flags {
        uint inFlags[];
    };
#endif // FLAGS_AVAILABLE
