#version 430

#include "protein_gl/deferred/blinn_phong.glsl"
#include "protein_gl/deferred/lambert.glsl"
#include "protein_gl/deferred/gbuffer_output.glsl"

/////////////
// DEFINES //
/////////////
#define EPS 0.0001

////////////
// LAYOUT //
////////////  
layout (depth_greater) out float gl_FragDepth; // we think this is right
// this should be wrong //layout (depth_less) out float gl_FragDepth;

/////////
// IN //
//////// 
in vec4 mycol;
in vec3 normal;
in vec3 basenormal;

in vec3 uncValues;  // Uncertainty and DITHERING values ([0]uncertainty, [1]dithering threshold max, [2]dithering threshold min)
                      
//////////
// OUT //
/////////

/////////////////////
// INPUT variables //
/////////////////////

struct LightParams
{
    float x,y,z,intensity;
};

layout(std430, binding = 4) readonly buffer PointLightParamsBuffer { LightParams point_light_params[]; };
layout(std430, binding = 5) readonly buffer DistantLightParamsBuffer { LightParams distant_light_params[]; };

uniform vec4 viewAttr;

uniform mat4 MVP;
uniform mat4 MV;
uniform mat4 MVPinv;
uniform mat4 MVPtransp;
uniform mat4 ProjInv;

uniform vec4 diffuseColor   = vec4(1.0);
uniform vec4 ambientColor   = vec4(1.0); 
uniform vec4 specularColor  = vec4(1.0);    
uniform vec4 phong          = vec4(0.6, 0.6, 0.8, 10.0);             // ambient - diffuse - specular - exponent
uniform vec4 phongUncertain = vec4(0.6, 0.6, 0.8, 10.0);    

uniform int  alphaBlending  = 0;                                     // Using alpha blending instead of dithering
uniform int  ditherCount    = 0;                                     // DITHERING - current dithering pass - in range [0,number of struct types - 1]                   
uniform int  outlineMode    = 0;
uniform vec3 outlineColor;
uniform int point_light_cnt;
uniform int distant_light_cnt;
uniform vec3 camPos;

const float dithMat[64] =   { 0, 32,  8, 40,  2, 34, 10, 42,        // DITHERING - dither matrix bayer 8x8
                             48, 16, 56, 24, 50, 18, 58, 26,
                             12, 44,  4, 36, 14, 46,  6, 38, 
                             60, 28, 52, 20, 62, 30, 54, 22,
                              3, 35, 11, 43,  1, 33,  9, 41,
                             51, 19, 59, 27, 49, 17, 57, 25,
                             15, 47,  7, 39, 13, 45,  5, 37,
                             63, 31, 55, 23, 61, 29, 53, 21}; 

//////////
// MAIN //
//////////
void main(void) {

    if (outlineMode > 0) { //  -> != OUTLINE_NONE
        albedo_out = vec4(outlineColor.xyz, 1.0);
    }
    else {
        vec4 mat = phong;
    
        // dithering - check if fragment is discarded
        if (ditherCount > 0) {
            if (alphaBlending == 0) {
                if ((uncValues[1] - uncValues[2]) < EPS) {
                    discard;  
                }
                else {
                    int index = (int(gl_FragCoord.y)%8 * 8 + int(gl_FragCoord.x)%8);
                    float t   = dithMat[index] / 64.0;
                    // draw fragment only if (uncValues[1] = start = max) >= t >=  (uncValues[2] = end = min)
                    if ((t > uncValues[1]) || (uncValues[2] > t)) { 
                        discard;
                    }   
                }
            }
        }
    
        // assign uncertain material properties if probability of assigned structure is > 0.0
        if (uncValues[0]  > 0.0) {
            mat = phongUncertain;
        }
            
        // calculate Blinn-Phong shading
        vec4 pos      = MVPinv * gl_FragCoord.xyzw;
        vec3 n       = normalize(basenormal);
        vec3 world_pos = pos.xyz;

        vec3 reflected_light = vec3(0.0);
        for(int i = 0; i < point_light_cnt; ++i) {
            vec3 light_dir = vec3(point_light_params[i].x, point_light_params[i].y, point_light_params[i].z) - world_pos;
            float d = length(light_dir);
            light_dir = normalize(light_dir);
            vec3 view_dir = normalize(camPos - world_pos);
            reflected_light += blinnPhong(n, light_dir, view_dir, ambientColor, diffuseColor, specularColor, mat) * point_light_params[i].intensity * (1.0/(d*d));
        }
        for(int i = 0; i < distant_light_cnt; ++i) {
            vec3 light_dir = -1.0 * vec3(distant_light_params[i].x, distant_light_params[i].y, distant_light_params[i].z);
            vec3 view_dir = normalize(camPos - world_pos);
            reflected_light += blinnPhong(n, light_dir, view_dir, ambientColor, diffuseColor, specularColor, mat) * distant_light_params[i].intensity;
        }
    
        albedo_out = vec4(reflected_light * mycol.rgb, mycol.a);
    
        if ((ditherCount > 0) && (alphaBlending == 1)) {
            albedo_out = vec4(reflected_light, uncValues[0]);
        }
    }
}
