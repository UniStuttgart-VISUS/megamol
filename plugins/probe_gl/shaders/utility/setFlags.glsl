
layout(std430, binding = 0) readonly buffer InputIDs { 
    uint ids[]; 
};

layout(std430, binding = 1) buffer Flags {
    uint flagsArray[];
};


uniform uint id_cnt;
uniform uint flags_cnt;//?


layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main()
{
    uvec3 gID = gl_GlobalInvocationID.xyz;

    if (gID.x >= id_cnt) {
        return;
    }

    uint flag_id = ids[gID.x];
    bitflag_set(flagsArray[flag_id], FLAG_ENABLED, true);
    bitflag_set(flagsArray[flag_id], FLAG_FILTERED, false);

}
