#version 450

#include "common/common.inc.glsl"
#include "mmstd_gl/common/tflookup.inc.glsl"
#include "mmstd_gl/common/tfconvenience.inc.glsl"

in vec2 uvCoords;
out vec4 fragOut;

uniform int dual_space_width;
uniform int dual_space_height;
layout (binding=7) uniform usampler2DArray imgRead;
layout (binding=8) uniform usampler2DArray slctRead;

layout (binding=9) uniform usampler2DArray dual_centroidX_tex;
layout (binding=10) uniform usampler2DArray dual_centroidY_tex;
layout (binding=11) uniform usampler2DArray dual_coMoment_tex;

#define PI        3.1415926538
#define HALF_PI    1.5707963268
#define QUARTER_PI 0.7853981634

float fromFixPoint(uint v){
    return float(v) / 1000.0;
}

float bilinearInterpolation(usampler2DArray sampler, vec3 coords){
    ivec3 texture_resolution = textureSize(sampler,0);

    vec2 shifted_coord = vec2(
        clamp(coords.x * float(texture_resolution.x) - 0.5,0.0,float(texture_resolution.x)),
        clamp(coords.y * float(texture_resolution.y) - 0.5,0.0,float(texture_resolution.y))
    );
    ivec2 base_texel = ivec2( floor(shifted_coord.x), floor(shifted_coord.y));
    vec2 f = vec2( fract(shifted_coord.x), fract(shifted_coord.y) );

    float s00 = fromFixPoint(texelFetch(sampler, ivec3(base_texel+ivec2(0,0),int(coords.z)), 0).x);
    float s10 = fromFixPoint(texelFetch(sampler, ivec3(base_texel+ivec2(1,0),int(coords.z)), 0).x);
    float s01 = fromFixPoint(texelFetch(sampler, ivec3(base_texel+ivec2(0,1),int(coords.z)), 0).x);
    float s11 = fromFixPoint(texelFetch(sampler, ivec3(base_texel+ivec2(1,1),int(coords.z)), 0).x);

    return mix( mix(s00,s10,f.x), mix(s01, s11, f.x), f.y);
}

float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    bool selected = false;
    int cdim = int(floor(uvCoords.x));
    float relx = fract(uvCoords.x);
    float result = 0.0;


    //      // Variant One
    //      {
    //          float left_sum_result = 0.0;
    //          float right_sum_result = 0.0;
    //          float random_sample_cnt = 3.0;
    //          //if(relx > 0.0)
    //          {
    //              for(int left_axis = 0; left_axis<dual_space_height; ++left_axis){
    //                  float left_axis_normalized = (float(left_axis))/float(dual_space_height-1);
    //      
    //                  for(float r_sample_idx = 0; r_sample_idx<random_sample_cnt; ++r_sample_idx)
    //                  {
    //                      left_axis_normalized += (r_sample_idx/ max(1.0,(random_sample_cnt-1)*2.0) ) * rand(uvCoords*vec2(r_sample_idx)) / dual_space_height;
    //      
    //                      vec2 from = vec2(0.0,left_axis_normalized);
    //                      vec2 to = vec2(relx,uvCoords.y);
    //                      vec2 v = to-from;
    //                      v = v * (1.0/relx);
    //                      float right_axis_normalized = from.y + v.y;
    //                      if(right_axis_normalized >= 0.0 && right_axis_normalized <= 1.0)
    //                      {
    //                          //result += fromFixPoint(texelFetch(imgRead, ivec3(left_axis, int(round(right_axis_normalized*(dual_space_height-1.0f))), cdim), 0).x);
    //                          left_sum_result += bilinearInterpolation(imgRead, vec3(left_axis_normalized,right_axis_normalized,float(cdim))) / random_sample_cnt;
    //                      }
    //                  }
    //              }
    //          }
    //          //else
    //          {
    //              for(int right_axis = 0; right_axis<dual_space_height; ++right_axis){
    //                  float right_axis_normalized = (float(right_axis))/float(dual_space_height-1);
    //      
    //                  for(float r_sample_idx = 0; r_sample_idx<random_sample_cnt; ++r_sample_idx)
    //                  {
    //                      right_axis_normalized += (r_sample_idx/ max(1.0,(random_sample_cnt-1)*2.0) ) * rand(uvCoords*vec2(r_sample_idx)) / dual_space_height;
    //      
    //                      vec2 from = vec2(1.0,right_axis_normalized);
    //                      vec2 to = vec2(relx,uvCoords.y);
    //                      vec2 v = to-from;
    //                      v = v /(1.0-relx);
    //                      float left_axis_normalized = from.y + v.y;
    //                      if(left_axis_normalized >= 0.0 && left_axis_normalized <= 1.0)
    //                      {
    //                          //result += fromFixPoint(texelFetch(imgRead, ivec3(left_axis_normalized*(dual_space_height-1.0f), right_axis, cdim), 0).x);
    //                          right_sum_result += bilinearInterpolation(imgRead, vec3(left_axis_normalized,right_axis_normalized,float(cdim))) / random_sample_cnt;
    //                      }
    //                  }
    //              }
    //          }
    //          result = mix(left_sum_result,right_sum_result,1.0-relx);
    //          //result = relx > 0.5 ? left_sum_result : right_sum_result;
    //      }

    float left_min = 0.0;
    float left_max = 1.0;
    float right_min = 0.0;
    float right_max = 1.0;

    // using max left range, update right range
    vec2 to = vec2(relx,uvCoords.y);
    vec2 from = vec2(0.0,0.0);
    right_max = min(1.0,((to-from) * (1.0/relx)).y + from.y);
    from = vec2(0.0,1.0);
    right_min = max(0.0,((to-from) * (1.0/relx)).y + from.y);

    // using updated right range, update left range
    from = vec2(1.0,right_min);
    left_max = min(1.0,((to-from) /(1.0-relx)).y + from.y);
    from = vec2(1.0,right_max);
    left_min = max(0.0,((to-from) /(1.0-relx)).y + from.y);

    float left_range = (left_max-left_min);
    float right_range = (right_max-right_min);

    float tri_area_left = 0.5 * left_range * relx;
    float tri_area_right = 0.5 * right_range * (1.0-relx);

    for(int sample_idx = 0; sample_idx < dual_space_height; ++sample_idx){
        float left = left_min + (float(sample_idx)/float(dual_space_height-1)) * left_range;
        float right = right_max - (float(sample_idx)/float(dual_space_height-1)) * right_range;

        result += bilinearInterpolation(imgRead, vec3(left,right,float(cdim))) * (tri_area_left+tri_area_right)/0.5;
    }

    //fragOut = vec4(tri_area_left+tri_area_right,0.0,0.0,1.0);
    //return;

    if(result > 0 || selected){
        if(!selected){
            fragOut = tflookup((result-1));
        }else{
            fragOut = vec4(1,0,0,1);
        }
    } else {
        discard;
    }
}
