
struct MeshShaderParams
{
    vec4 glpyh_position;
    vec4 probe_direction;
    float scale;

    int probe_id;
    int state;

    float sample_cnt;
    
    int cluster_id;
    int total_cluster_cnt;
    int padding1;
    int padding2;
};

struct PerFrameData
{
   int use_interpolation;

   int padding0;
   int padding1;
   int padding2;
};
