// https://stackoverflow.com/questions/28375338/cube-using-single-gl-triangle-strip
const vec3 cube_strip[14] = vec3[14](
    vec3(-1.f, 1.f, 1.f),     // Front-top-left
    vec3(1.f, 1.f, 1.f),      // Front-top-right
    vec3(-1.f, -1.f, 1.f),    // Front-bottom-left
    vec3(1.f, -1.f, 1.f),     // Front-bottom-right
    vec3(1.f, -1.f, -1.f),    // Back-bottom-right
    vec3(1.f, 1.f, 1.f),      // Front-top-right
    vec3(1.f, 1.f, -1.f),     // Back-top-right
    vec3(-1.f, 1.f, 1.f),     // Front-top-left
    vec3(-1.f, 1.f, -1.f),    // Back-top-left
    vec3(-1.f, -1.f, 1.f),    // Front-bottom-left
    vec3(-1.f, -1.f, -1.f),   // Back-bottom-left
    vec3(1.f, -1.f, -1.f),    // Back-bottom-right
    vec3(-1.f, 1.f, -1.f),    // Back-top-left
    vec3(1.f, 1.f, -1.f)      // Back-top-right
);

// provoking vertex is LAST by default
const vec3 cube_normals[14] = vec3[14](
    vec3(0.0f,0.0f,0.0f),     // never last
    vec3(0.0f,0.0f,0.0f),     // never last
    vec3(0.f, 0.f, 1.f),    // front1
    vec3(0.f, 0.f, 1.f),     // front2
    vec3(0.f, -1.f, 0.f),    // bottom1
    vec3(1.f, 0.f, 0.f),      // right1
    vec3(1.f, 0.f, 0.f),     // right2
    vec3(0.f, 1.f, 0.f),     // top1
    vec3(0.f, 1.f, 0.f),    // top2
    vec3(-1.f, 0.f, 0.f),    // left1
    vec3(-1.f, 0.f, 0.f),   // left2
    vec3(0.f, -1.f, 0.f),    // bottom2
    vec3(0.f, 0.f, -1.f),    // back1
    vec3(0.f, 0.f, -1.f)      // back2
);