#version 450

#include "common/common.inc.glsl"
#include "mmstd_gl/common/tflookup.inc.glsl"
#include "mmstd_gl/common/tfconvenience.inc.glsl"

in vec2 uvCoords;
out vec4 fragOut;
// height is actually number of bins
uniform int axPxHeight;
uniform int axPxWidth;
uniform float debugFloat;
uniform int binsNr;
uniform int thetas;
uniform int rhos;
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
    uvec4 integer_samples = textureGather(sampler, coords, 0);
    ivec3 texture_resolution = textureSize(sampler,0);
    vec2 pixel_coords = vec2(coords.x * float(texture_resolution.x - 1),coords.y * float(texture_resolution.y - 1));
    vec4 weights = vec4(
        (1.0-fract(pixel_coords.x))*fract(pixel_coords.y),
        fract(pixel_coords.x)*fract(pixel_coords.y),
        fract(pixel_coords.x)*(1.0-fract(pixel_coords.y)),
        (1.0-fract(pixel_coords.x))*(1.0-fract(pixel_coords.y))
    );

    vec4 samples = vec4(fromFixPoint(integer_samples.x),fromFixPoint(integer_samples.y),fromFixPoint(integer_samples.z),fromFixPoint(integer_samples.w));
    vec4 weighted_samples = samples * weights;

    return (weighted_samples.x + weighted_samples.y + weighted_samples.z + weighted_samples.w);
}

void main() {
    bool selected = false;
    int axPxDist = axPxWidth / int(dimensionCount-1);
    int cdim = int(floor(uvCoords.x));
    float relx = fract(uvCoords.x);
    float result = 0.0;

    int prev_b_idx = 0;

    for(int m_idx = 0; m_idx<thetas; ++m_idx){
        float m_normalized = (float(m_idx)+0.5)/float(thetas);
        float m_angle = (m_normalized * HALF_PI + QUARTER_PI);
        float b_normalized = uvCoords.y + (relx * tan(HALF_PI - m_angle));

        float b = b_normalized * (rhos-1);
        int current_b_idx = int(floor(b));

        // need to scan all rhos between current hit and last theta's hit?
        for(int b_idx = prev_b_idx; b_idx >= current_b_idx; --b_idx){
        
            b_normalized = (float(b_idx)+0.5) / float(rhos);
        
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
            int theta_index_corrected = int(floor((m_angle_corrected - QUARTER_PI) * (thetas) / (HALF_PI))); 
            
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


    //      for(int b = 0; b<rhos; ++b){
    //          float b_normalized = (float(b)+0.5)/float(rhos);
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
    //          float theta = (m_angle - quarterPI) * (thetas) / (halfPI);
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
