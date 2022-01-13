
struct MeshShaderParams
{
    vec4 glpyh_position;
    vec4 probe_direction;
    float scale;

    int probe_id;
    int state;

    float sample_cnt;
    vec4 samples[32];
};

struct PerFrameData
{
   int use_interpolation;

   int show_canvas;

   int padding0;
   int padding1;

   vec4 canvas_color;

   uvec2 tf_texture_handle;
   float tf_min;
   float tf_max;
};
