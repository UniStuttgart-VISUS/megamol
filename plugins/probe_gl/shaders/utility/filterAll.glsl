
layout(std430, binding = 1) buffer Flags {
    uint flagsArray[];
};

uniform uint flag_cnt;//?

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main()
{
    uvec3 gID = gl_GlobalInvocationID.xyz;

    if (gID.x >= flag_cnt) {
        return;
    }

    bitflag_set(flagsArray[gID.x], FLAG_ENABLED, false);
    bitflag_set(flagsArray[gID.x], FLAG_FILTERED, true);

}
