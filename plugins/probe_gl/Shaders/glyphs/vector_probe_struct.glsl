
struct MeshShaderParams
{
    vec4 glpyh_position;
    vec4 probe_direction;
    float scale;

    float padding0;
    float padding1;

    float sample_cnt;
    vec4 samples[32];
};
