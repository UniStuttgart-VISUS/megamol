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

void main() {
    bool selected = false;
    int cdim = int(floor(uvCoords.x));
    float relx = fract(uvCoords.x);
    float result = 0.0;

    int prev_b_idx = 0;

    for(int m_idx = 0; m_idx<dual_space_width; ++m_idx){
        float m_normalized = float(m_idx)/float(dual_space_width-1);
        float m_angle = (m_normalized * HALF_PI + QUARTER_PI);
        float b_normalized = uvCoords.y + (relx * tan(HALF_PI - m_angle));

        float b = b_normalized * dual_space_height;
        int current_b_idx = int(floor(b));

        // need to scan all dual_space_height between current hit and last theta's hit?
        for(int b_idx = prev_b_idx; b_idx >= current_b_idx; --b_idx){
        
            b_normalized = float(b_idx)/ float(dual_space_height-1);
        
            // filter out hits above/below axis
            if(b_normalized < 0.0 || b_normalized > 1.0){
                continue;
            }
        
            // update m_idx to match b
            vec2 from = vec2(0.0, b_normalized);
            vec2 to = vec2(relx, uvCoords.y);
            vec2 lineDir = normalize(to - from);
            vec2 orthLine = vec2(-lineDir.y, lineDir.x);
            float m_angle_corrected = acos(dot(orthLine, vec2(1.0, 0.0)));
            float m_normalized_corrected = (m_angle_corrected - QUARTER_PI)/(HALF_PI);
            int theta_index_corrected = int(floor((m_angle_corrected - QUARTER_PI) * (dual_space_width) / (HALF_PI))); 
            
            //result += fromFixPoint(texelFetch(imgRead, ivec3(m_idx, b_idx, cdim), 0).x);
            result += bilinearInterpolation(imgRead,vec3( m_normalized ,b_normalized,float(cdim)));
        
            //result += 1.0;
        }
        prev_b_idx = current_b_idx;

        //int b_idx_0 = int(floor(b));
        //int b_idx_1 = int(ceil(b));
        //
        //float centroid_x_0 = fromFixPoint(texelFetch(dual_centroidX_tex, ivec3(m_idx, b_idx_0, cdim), 0).x);
        //float centroid_y_0 = fromFixPoint(texelFetch(dual_centroidY_tex, ivec3(m_idx, b_idx_0, cdim), 0).x);
        //float coMoment_0 = fromFixPoint(texelFetch(dual_coMoment_tex, ivec3(m_idx, b_idx_0, cdim), 0).x);
        //
        //float centroid_x_1 = fromFixPoint(texelFetch(dual_centroidX_tex, ivec3(m_idx, b_idx_1, cdim), 0).x);
        //float centroid_y_1 = fromFixPoint(texelFetch(dual_centroidY_tex, ivec3(m_idx, b_idx_1, cdim), 0).x);
        //float coMoment_1 = fromFixPoint(texelFetch(dual_coMoment_tex, ivec3(m_idx, b_idx_1, cdim), 0).x);
        //
        //result += uint( (float(b_idx_1) - b) * float(texelFetch(imgRead, ivec3(m_idx, b_idx_0, cdim), 0).x));
        //result += uint( (b - float(b_idx_0)) * float(texelFetch(imgRead, ivec3(m_idx, b_idx_1, cdim), 0).x));
    }


    //      for(int b = 0; b<dual_space_height; ++b){
    //          float b_normalized = (float(b)+0.5)/float(dual_space_height);
    //      
    //          // compute theta for current b and current pixel
    //          vec2 from = vec2(0.0, b_normalized);
    //          vec2 to = vec2(relx, uvCoords.y);
    //          vec2 lineDir = normalize(to - from);
    //          vec2 orthLine = vec2(-lineDir.y, lineDir.x);
    //          float m_angle = acos(dot(orthLine, vec2(1.0, 0.0)));
    //          if(m_angle < 2*quarterPI || m_angle > (3*quarterPI)){
    //              continue;
    //          }
    //          float theta = (m_angle - quarterPI) * (dual_space_width) / (halfPI);
    //      
    //          // read from two closest theta entries
    //          int theta_0 = int(floor(theta));
    //          int theta_1 = int(ceil(theta));
    //          float centroid_x_0 = fromFixPoint(texelFetch(dual_centroidX_tex, ivec3(theta_0, b, cdim), 0).x);
    //          float centroid_y_0 = fromFixPoint(texelFetch(dual_centroidY_tex, ivec3(theta_0, b, cdim), 0).x);
    //          float coMoment_0 = fromFixPoint(texelFetch(dual_coMoment_tex, ivec3(theta_0, b, cdim), 0).x);
    //      
    //          float centroid_x_1 = fromFixPoint(texelFetch(dual_centroidX_tex, ivec3(theta_0, b, cdim), 0).x);
    //          float centroid_y_1 = fromFixPoint(texelFetch(dual_centroidY_tex, ivec3(theta_0, b, cdim), 0).x);
    //          float coMoment_1 = fromFixPoint(texelFetch(dual_coMoment_tex, ivec3(theta_0, b, cdim), 0).x);
    //      
    //          //TODO check distance to centroid and compare with co-moment
    //      
    //          result += int(texelFetch(imgRead, ivec3(theta_0, b, cdim), 0).x);
    //          result += int(texelFetch(imgRead, ivec3(theta_1, b, cdim), 0).x);
    //      }

    //fragOut = vec4(relx,uvCoords.y,0,1);
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
