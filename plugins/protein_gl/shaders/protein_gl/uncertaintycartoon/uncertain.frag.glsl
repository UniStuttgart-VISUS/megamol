#version 430

#define CLIP
#define DEPTH
#define SMALL_SPRITE_LIGHTING
//#define CALC_CAM_SYS

//#define RETICLE
//#define DISCARD_COLOR_MARKER
#ifdef DEBUG
#undef CLIP
#define RETICLE
#define AXISHINTS
#endif // DEBUG

//#define BULLSHIT

#ifndef FLACH
#define FLACH
#endif

// DIRECTIONAL LIGHTING (Blinn Phong)

// ray:      the eye to fragment ray vector
// normal:   the normal of this fragment
// lightdir: the direction of the light 
// color:    the base material color

//#define USE_SPECULAR_COMPONENT

vec3 LocalLighting(const in vec3 ray, const in vec3 normal, const in vec3 lightdir, const in vec3 color) {

    vec3 lightdirn = normalize(-lightdir); // (negativ light dir for directional lighting)

    vec4 lightparams = vec4(0.2, 0.8, 0.4, 10.0);
#define LIGHT_AMBIENT  lightparams.x
#define LIGHT_DIFFUSE  lightparams.y
#define LIGHT_SPECULAR lightparams.z
#define LIGHT_EXPONENT lightparams.w

    float nDOTl = dot(normal, lightdirn);

    vec3 specular_color = vec3(0.0, 0.0, 0.0);
#ifdef USE_SPECULAR_COMPONENT
    vec3 r = normalize(2.0 * vec3(nDOTl) * normal - lightdirn);
    specular_color = LIGHT_SPECULAR * vec3(pow(max(dot(r, -ray), 0.0), LIGHT_EXPONENT));
#endif // USE_SPECULAR_COMPONENT

    return LIGHT_AMBIENT  * color 
         + LIGHT_DIFFUSE  * color * max(nDOTl, 0.0)
         + specular_color;
}

//////////////
// FRAGMENT //
//////////////

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

in vec3 uncValues;  // Uncertainty and DITHERING values ([0]uncertainty, [1]dithering threshold max, [2]dithering threshold min)
                      
//////////
// OUT //
/////////
out layout(location = 0) vec4 outCol;

/////////////////////
// INPUT variables //
/////////////////////
uniform vec4 viewAttr;

uniform mat4 MVP;
uniform mat4 MV;
uniform mat4 MVPinv;
uniform mat4 MVPtransp;
uniform mat4 ProjInv;

uniform vec4 lightPos;
uniform vec4 diffuseColor   = vec4(1.0);
uniform vec4 ambientColor   = vec4(1.0); 
uniform vec4 specularColor  = vec4(1.0);    
uniform vec4 phong          = vec4(0.6, 0.6, 0.8, 10.0);             // ambient - diffuse - specular - exponent
uniform vec4 phongUncertain = vec4(0.6, 0.6, 0.8, 10.0);    

uniform int  alphaBlending  = 0;                                     // Using alpha blending instead of dithering
uniform int  ditherCount    = 0;                                     // DITHERING - current dithering pass - in range [0,number of struct types - 1]                   
uniform int  outlineMode    = 0;
uniform vec3 outlineColor;

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
        /*
        vec3 tmpCol = uncValues[0]*outlineColor.xyz;
        if (outlineColor.xyz == vec3(0.0)) {
            tmpCol = (1.0 - uncValues[0])*vec3(1.0);
        }
        outCol = vec4(tmpCol.xyz, 1.0);
        */
        outCol = vec4(outlineColor.xyz, 1.0);
    }
    else {
        vec4 material = phong;
    
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
            material = phongUncertain;
        }
            
		// calculate Blinn-Phong shading
        vec4 pos      = ProjInv * gl_FragCoord.xyzw;
        vec3 lightDir = normalize(lightPos.xyz);
		    //vec4 light    = MV * lightPos;
            //vec3 lightDir = normalize(light.xyz - pos.xyz);
        vec3  n       = normal;
        vec3  e       = pos.xyz;
        vec3  h       = normalize(lightDir + e);
        float d       = dot(n, lightDir);
        vec3  r       = normalize(2.0 * d * n - lightDir);
        vec3  c       = material.x * mycol.xyz + material.y * mycol.xyz * max(d, 0.0);
        vec4  eyenew  = vec4(0.0, 0.0, 0.0, 1.0);

        if(dot(lightDir, n) < 0.0) {
            c += vec3(0.0);
        }
        else {
            c += material.z * vec3(pow(max(dot(r, lightDir), 0.0), material.w));
        }
    
        outCol = vec4(c, 1.0);
    
        if ((ditherCount > 0) && (alphaBlending == 1)) {
            outCol = vec4(c, uncValues[0]);
        }
    }
}
