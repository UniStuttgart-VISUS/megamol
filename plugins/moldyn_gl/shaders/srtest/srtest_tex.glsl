uniform sampler1D data_tex;
uniform uint num_points;
uniform uint base_size;

void access_data(uint idx, out vec3 objPos, out vec4 objColor, out float rad) {
    float x = texelFetch(data_tex, int(idx * 4 + 0), 0).x;
    float y = texelFetch(data_tex, int(idx * 4 + 1), 0).x;
    float z = texelFetch(data_tex, int(idx * 4 + 2), 0).x;
    objPos = vec3(x,y,z);

    if (useGlobalRad) {
        rad = globalRad;
    } else {
        rad = 0.01f;
    }

    objColor = vec4(1);
}
