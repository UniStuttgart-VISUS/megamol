#extension GL_ARB_compute_shader: enable
#define FLT_MAX 3.402823466e+38
#define FLT_MIN 1.175494351e-38
#define PI      3.14159265

/* matrices */
uniform mat4 view_mx;
uniform mat4 proj_mx;

/* render targete resolution*/
uniform vec2 rt_resolution;

/* bounding box size */
uniform vec3 boxMin;
uniform vec3 boxMax;

/* voxel size */
uniform float voxelSize;
uniform vec3 halfVoxelSize;

/* sampling frequency */
uniform float rayStepRatio;

/* value range */
uniform vec2 valRange;

/* background color */
uniform vec4 background;

/* lighting */
uniform bool use_lighting;
uniform vec3 material_col;

/* texture that houses the volume data */
uniform highp sampler3D volume_tx3D;
