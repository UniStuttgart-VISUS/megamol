// if intensity is used, it is transported in in_color.r
vec4 compute_color(vec4 in_color, uint flag, sampler1D tf_sampler, vec2 tf_range, vec4 global_color, vec4 flag_selected_color, vec4 flag_soft_selected_color, uint options) {

    bool use_global_color = (options & OPTIONS_USE_GLOBAL_COLOR) > 0;
    bool use_transfer_function = (options & OPTIONS_USE_TRANSFERFUNCTION) > 0;
    bool are_flags_available = (options & OPTIONS_USE_FLAGS) > 0;
    vec4 color = in_color;
    if (use_global_color)  {
        color = global_color;
    }
    if (use_transfer_function) {
        color = tflookup(tf_sampler, tf_range, color.r);
    }
    // Overwrite color depending on flags
    if (are_flags_available) {
        if (bitflag_test(flag, FLAG_SELECTED, FLAG_SELECTED)) {
            color = flag_selected_color;
        } else if (bitflag_test(flag, FLAG_SOFTSELECTED, FLAG_SOFTSELECTED)) {
            color = flag_soft_selected_color;
        }
    }
    return color;
}