uniform ivec3 instancing_index = ivec3(1, 1, 1);
uniform vec3 instancing_offset = vec3(0.0, 0.0, 0.0);

vec3 applyInstancing(vec3 pos) {
    return pos + instancing_offset * instancing_index;
}
