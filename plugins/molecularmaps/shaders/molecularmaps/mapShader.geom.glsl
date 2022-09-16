#extension GL_EXT_geometry_shader4 : enable
#extension GL_EXT_gpu_shader4 : enable
#extension GL_ARB_shader_storage_buffer_object : enable

layout(std430, binding = 12) buffer vertexIndices
{
    int indexmap[];
};

layout(std430, binding = 13) buffer vertexValues
{
    float valuespervertex[];
};

out vec4 colorval;
out float value;
out float index;

in int vertid[];

float idxToVal(int idx) {
    return float(vertid[idx]);
}

void main(void) {
    vec3 v[3];
    v[0] = vec3(gl_PositionIn[0].xyz) / gl_PositionIn[0].w;
    v[1] = vec3(gl_PositionIn[1].xyz) / gl_PositionIn[1].w;
    v[2] = vec3(gl_PositionIn[2].xyz) / gl_PositionIn[2].w;

    /*
    if( gl_PositionIn[0].z > 0.0 && gl_PositionIn[1].z > 0.0 && gl_PositionIn[2].z > 0.0) {
        return;
    }
    */
    // TODO distortion colors!

    // shift positions that are outside the map
    if (v[0].x > 1.0) {
        v[0].x -= 2.0;
    }
    if (v[0].x < -1.0) {
        v[0].x += 2.0;
    }
    if (v[1].x > 1.0) {
        v[1].x -= 2.0;
    }
    if (v[1].x < -1.0) {
        v[1].x += 2.0;
    }
    if (v[2].x > 1.0) {
        v[2].x -= 2.0;
    }
    if (v[2].x < -1.0) {
        v[2].x += 2.0;
    }
    //sort vectors
    int idx0 = 0;
    int idx1 = 1;
    int idx2 = 2;
    if (v[0].x < v[1].x) {
        if (v[0].x < v[2].x) {
            if (v[1].x > v[2].x) {
                idx1 = 2;
                idx2 = 1;
            }
        } else {
            idx0 = 2;
            idx1 = 0;
            idx2 = 1;
        }
    } else {
        if (v[1].x < v[2].x) {
            if (v[0].x < v[2].x) {
                idx0 = 1;
                idx1 = 0;
            } else {
                idx0 = 1;
                idx1 = 2;
                idx2 = 0;
            }
        } else {
            idx0 = 2;
            idx2 = 0;
        }
    }

    // duplicate overlapping triangles
    if( v[idx0].x < -0.5 && v[idx1].x < -0.5 && v[idx2].x > 0.5 ) {
        gl_FrontColor = gl_FrontColorIn[idx0];
        colorval = gl_FrontColor;
        value = valuespervertex[indexmap[vertid[idx0]]];
        gl_Position = vec4( v[idx0], 1.0);
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[idx1];
        colorval = gl_FrontColor;
        value = valuespervertex[indexmap[vertid[idx1]]];
        gl_Position = vec4( v[idx1], 1.0);
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[idx2];
        colorval = gl_FrontColor;
        value = valuespervertex[indexmap[vertid[idx2]]];
        gl_Position = vec4( v[idx2] - vec3(2.0, 0.0, 0.0), 1.0);
        EmitVertex();
        EndPrimitive();

        gl_FrontColor = gl_FrontColorIn[idx0];
        colorval = gl_FrontColor;
        value = valuespervertex[indexmap[vertid[idx0]]];
        gl_Position = vec4( v[idx0] + vec3(2.0, 0.0, 0.0), 1.0);
        EmitVertex();

        gl_FrontColor =  gl_FrontColorIn[idx1];
        colorval = gl_FrontColor;
        value = valuespervertex[indexmap[vertid[idx1]]];
        gl_Position = vec4( v[idx1] + vec3(2.0, 0.0, 0.0), 1.0);
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[idx2];
        colorval = gl_FrontColor;
        value = valuespervertex[indexmap[vertid[idx2]]];
        gl_Position = vec4( v[idx2], 1.0);
        EmitVertex();
        EndPrimitive();
    } else if( v[idx0].x < -0.5 && v[idx1].x > 0.5 && v[idx2].x > 0.5 ) {
        gl_FrontColor = gl_FrontColorIn[idx0];
        colorval = gl_FrontColor;
        value = valuespervertex[indexmap[vertid[idx0]]];
        gl_Position = vec4( v[idx0], 1.0);
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[idx1];
        colorval = gl_FrontColor;
        value = valuespervertex[indexmap[vertid[idx1]]];
        gl_Position = vec4( v[idx1] - vec3(2.0, 0.0, 0.0), 1.0);
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[idx2];
        colorval = gl_FrontColor;
        value = valuespervertex[indexmap[vertid[idx2]]];
        gl_Position = vec4( v[idx2] - vec3(2.0, 0.0, 0.0), 1.0);
        EmitVertex();
        EndPrimitive();

        gl_FrontColor = gl_FrontColorIn[idx0];
        colorval = gl_FrontColor;
        value = valuespervertex[indexmap[vertid[idx0]]];
        gl_Position = vec4( v[idx0] + vec3(2.0, 0.0, 0.0), 1.0);
        EmitVertex();

        gl_FrontColor =  gl_FrontColorIn[idx1];
        colorval = gl_FrontColor;
        value = valuespervertex[indexmap[vertid[idx1]]];
        gl_Position = vec4( v[idx1], 1.0);
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[idx2];
        colorval = gl_FrontColor;
        value = valuespervertex[indexmap[vertid[idx2]]];
        gl_Position = vec4( v[idx2], 1.0);
        EmitVertex();
        EndPrimitive();
    } else {
        gl_FrontColor = gl_FrontColorIn[idx0];
        colorval = gl_FrontColor;
        value = valuespervertex[indexmap[vertid[idx0]]];
        gl_Position = vec4( v[idx0], 1.0);
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[idx1];
        colorval = gl_FrontColor;
        value = valuespervertex[indexmap[vertid[idx1]]];
        gl_Position = vec4( v[idx1], 1.0);
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[idx2];
        colorval = gl_FrontColor;
        value = valuespervertex[indexmap[vertid[idx2]]];
        gl_Position = vec4( v[idx2], 1.0);
        EmitVertex();
        EndPrimitive();
    }
}
