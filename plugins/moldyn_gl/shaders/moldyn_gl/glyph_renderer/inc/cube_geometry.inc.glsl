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

// this instead is a half cube made from 3 vertex strips, each representing the "positive" face of one axis
// [face][vertex]
const vec3 cube_faces[3][4] = {
    // right
    {
        vec3( 1.f,  1.f,  1.f),
        vec3( 1.f, -1.f,  1.f),
        vec3( 1.f,  1.f, -1.f),
        vec3( 1.f, -1.f, -1.f)
    },
    // top
    {
        vec3(-1.f,  1.f,  1.f),
        vec3( 1.f,  1.f,  1.f),
        vec3(-1.f,  1.f, -1.f),
        vec3( 1.f,  1.f, -1.f)
    },
    // front
    {
        vec3(-1.f, -1.f,  1.f),
        vec3( 1.f, -1.f,  1.f),
        vec3(-1.f,  1.f,  1.f),
        vec3( 1.f,  1.f,  1.f)
    }
};

const vec3 base_cube_verts[8] = {vec3(1.f, 1.f, 1.f), vec3(1.f, -1.f, 1.f), vec3(1.f, 1.f, -1.f), vec3(1.f, -1.f, -1.f),
    vec3(-1.f, 1.f, 1.f), vec3(-1.f, 1.f, -1.f), vec3(-1.f, -1.f, 1.f), vec3(-1.f, -1.f, -1.f)};

const uvec3 face_ind_front[3][2] = {
    {uvec3(0, 1, 2), uvec3(1, 3, 2)}, {uvec3(4, 0, 5), uvec3(0, 2, 5)}, {uvec3(6, 1, 4), uvec3(1, 0, 4)}};
const uvec3 face_ind_back[3][2] = {
    {uvec3(7, 5, 6), uvec3(5, 4, 6)}, {uvec3(3, 7, 1), uvec3(7, 6, 1)}, {uvec3(2, 5, 3), uvec3(5, 7, 3)}};

// the normals can be trivially indexed by face (see above)
const vec3 cube_face_normals[3] = {
    vec3(1.f, 0.f, 0.f), vec3(0.f, 1.f, 0.f), vec3(0.f, 0.f, 1.f)
};
