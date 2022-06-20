#version 430
//////////////////////////////
// TESSELATION - evaluation //
//////////////////////////////
                  
/////////////
// DEFINES //
/////////////
#define M_PI          3.1415926535897932384626433832795
#define EPS           0.0001

// should be equal to 'UncertaintyDataCall::secStructure':
#define G_310_HELIX   0
#define T_H_TURN      1
#define H_ALPHA_HELIX 2
#define I_PI_HELIX    3
#define S_BEND        4
#define C_COIL        5
#define B_BRIDGE      6
#define E_EXT_STRAND  7
#define NOTDEFINED    8
#define STRUCT_COUNT  9                                  

// should be equal to 'UncertaintyCartoonRenderer::coloringModes':    
#define COLOR_MODE_STRUCT        0
#define COLOR_MODE_UNCERTAIN     1
#define COLOR_MODE_CHAIN         2
#define COLOR_MODE_AMINOACID     3
#define COLOR_MODE_RESIDUE_DEBUG 4
        
// should be equal to 'UncertaintyCartoonRenderer::uncVisModeualisations':   
#define UNC_VIS_NONE   0
#define UNC_VIS_SIN_U  1
#define UNC_VIS_SIN_V  2    
#define UNC_VIS_SIN_UV 3
#define UNC_VIS_TRI_U  4
#define UNC_VIS_TRI_UV 5

// should be equal to 'UncertaintyCartoonRenderer::outlineOptions':   
#define OUTLINE_NONE           0
#define OUTLINE_LINE           1
#define OUTLINE_FULL_UNCERTAIN 2
#define OUTLINE_FULL_CERTAIN   3
    
////////////
// LAYOUT //
////////////          
layout(quads, equal_spacing, ccw) in;
//layout(isolines, equal_spacing) in;

////////
// IN //
////////   
in int   id[];

/////////
// OUT //
/////////        
out vec4 color;
out vec3 n;

out vec3 uncVal;                  // Uncertainty and DITHERING values - from tess eval

///////////////
// variables //
///////////////    
uniform float minDistance = 1.0;                                    // UNUSED ?
uniform float maxDistance = 1.0;                                    // UNUSED ?

uniform int                 alphaBlending     = 0;                  // Using alpha blending instead of dithering
uniform int                 ditherCount       = 0;                  // DITHERING - current dithering pass - in range [1,number of sec truct types - 1], dithering is disabled for 0
uniform bool                interpolateColors = false;
uniform bool                onlyTubes         = false;  
uniform float               pipeWidth         = 1.0;
uniform int                 colorMode         = 0;
uniform int                 uncVisMode        = 0;
uniform vec4[STRUCT_COUNT]  structCol;                              // the RGB colors for the strucutre types (order as in 'UncertaintyDataCall::secStructure')
uniform vec2                uncDistor;                              // uncertainty distortion parameters: (0) amplification for sin>0, (1) repeat of sin(2*PI)
uniform int                 outlineMode       = 0;
uniform float               outlineScale      = 1.0f;

struct CAlpha                                       
{
    vec4  pos;
    vec3  dir;
    int   colIdx;                                                   // UNUSED - don't delete ! ... otherwise shader has 'strange' behaviour - WHY?
    vec4  col;
    float uncertainty;
    int   flag;                                        
    float unc[STRUCT_COUNT];
    int   sortedStruct[STRUCT_COUNT];
};

layout(std430, binding = 2) buffer shader_data {
    CAlpha atoms[];
};

const float arrowLUT[10] =                                          // LUT = look-up-table
{
    5.0, 5.0, 5.0, 8.0, 7.0, 6.0, 4.5, 3.0, 2.0, 1.0
};

///////////////
// FUNCTIONS //
///////////////

float unitSin(float x) {

    return (sin((x*M_PI*2.0) - M_PI/2.0) + 1.0);
}

float unitTri(float x) {

    return ((1.0 - 2.0*abs(0.5 - (x - floor(x)))) * 2.0);
}

//////////
// MAIN //
//////////
void main() {

    // start(max) and end(min) values for dithering threshold
    uncVal[0] = atoms[id[1]].uncertainty;  // uncertainty
    uncVal[1] = 1.0;                       // dithering threshol max
    uncVal[2] = 0.0;                       // dithering threshold min
    // default assignment 
    gl_Position = vec4(0.0);
    n           = vec3(0.0);
    color       = vec4(0.0);
        
    // visualised structure typ (0 = structure which is most likely)
    int structIndex = 0;
    
    // for dithering passes: calculate start and end value for dithering threshold
    if (ditherCount > 0) { 
        CAlpha alpha = atoms[id[1]];  
        if (outlineMode == OUTLINE_NONE) {
            structIndex = ditherCount-1;
            // accumulate all uncertainties up to current structure type (sum uncertainty from minimum to current)
            for (int i = 0; i < structIndex; i++) {
                uncVal[1] -= alpha.unc[alpha.sortedStruct[i]];
            }
            uncVal[2] = uncVal[1] - alpha.unc[alpha.sortedStruct[structIndex]];
        }
        else { // draw ouline for structure with "biggest" geometry (helix or strand)
            for (int i = 0; i < ditherCount; i++) {
                int structure =  alpha.sortedStruct[i];
                if (alpha.unc[structure] > EPS) {
                    if ((structure == E_EXT_STRAND) || (structure == B_BRIDGE)){
                        structIndex = i;
                        break;
                    }
                    else if ((structure == H_ALPHA_HELIX) || (structure == G_310_HELIX) || (structure == I_PI_HELIX)){
                        structIndex = i;
                    }
                }
            }
        }
    }
    // calculate geometry
    if ((uncVal[1] - uncVal[2]) > EPS) { // threshold for uncertainty values ? ...
                  
        CAlpha alph0 = atoms[id[0] - 1];                            // (id[0] - 1) doesn't exist for first vertex call ?
        CAlpha alph1 = atoms[id[0]];
        CAlpha alph2 = atoms[id[1]];
        CAlpha alph3 = atoms[id[1] + 1];
    
        vec4 p0 = alph0.pos;
        vec4 p1 = alph1.pos;
        vec4 p2 = alph2.pos;
        vec4 p3 = alph3.pos;
   
        float u = gl_TessCoord.x;
        float v = gl_TessCoord.y;
    
        // Cubic B-Spline
        u += 3;
        float q = ( u - 1.0) / 3.0;
        vec4 d10 = p0 * ( 1.0 - q) + p1 * q;
        float q1 = ( u - 2.0) / 3.0;
        vec4 d11 =  p1 * ( 1.0 - q1) + p2 * q1;
        float q2 = ( u - 3.0) / 3.0; 
        vec4 d12 =  p2 * ( 1.0 - q2) + p3 * q2;
    
        float q3 = ( u - 2.0) / 2.0; 
        vec4 d20 = d10 * ( 1.0 - q3) + d11 * q3;
        float q4 = ( u - 3.0) / 2.0; 
        vec4 d21 = d11 * ( 1.0 - q4) + d12 * q4;
    
        float q5 = ( u - 3.0); 
        vec4 mypos = d20 * (1.0 - q5) + d21 * q5;
        vec4 savepos = mypos;
        gl_Position =  d20 * ( 1.0 - q5) + d21 * q5;
    
        // interpolate directions
        vec3 dir10 = mix(alph0.dir, alph1.dir, q);
        vec3 dir11 = mix(alph1.dir, alph2.dir, q1);
        vec3 dir12 = mix(alph2.dir, alph3.dir, q2);
    
        vec3 dir20 = mix(dir10, dir11, q3);
        vec3 dir21 = mix(dir11, dir12, q4);
    
        vec3 dir = normalize(mix(dir20, dir21, q5));
    
        vec3 tangent = normalize(d21.xyz - d20.xyz);
            // vec3 normal = normalize(cross(tangent, dir)); - UNUSED
    
        // compute corrected direction that is truly orthogonal to the tangent vector
        vec3 nDir    = dir;
        nDir         = dir -  dot(dir, tangent) * tangent;
        nDir         = -normalize(nDir);
                               
        // assign current structure type
        int mytype   = alph2.sortedStruct[structIndex];
                
        // assign colors
        
        // Uncertainty color blending: uCol is color of max uncertainty - cCol is color of max certainty - alpha is considered too!
        //vec4 cCol = vec4(1.0, 1.0, 1.0, 1.0);   // certain
        //vec4 cuCol = vec4(0.0, 0.0, 1.0, 1.0);  // 1/2 certain and 1/2 uncertain
        //vec4 uCol = vec4(0.0, 0.0, 0.0, 1.0);   // uncertain
        vec4 cCol = vec4(0.0, 0.0, 1.0, 1.0);   // certain
        vec4 cuCol = vec4(1.0, 1.0, 1.0, 1.0);  // 1/2 certain and 1/2 uncertain
        vec4 uCol = vec4(1.0, 0.0, 0.0, 1.0);   // uncertain
        
        // switch from two color interpolation to three color interpolation
        bool threeColInt = true;
        
        if(interpolateColors) // interpolate colors
        {      
            vec4 colors[4];
            
            for(int i = 0; i < 4; i++)
            {        
                if (colorMode == COLOR_MODE_STRUCT) {
                    colors[i] = structCol[atoms[id[0] + i - 1].sortedStruct[structIndex]];
                }     
                else if (colorMode == COLOR_MODE_UNCERTAIN) {
                
                    /*UNCERTAINTY to COLOR coding (with interpolation) */
                    float tempUnc = atoms[id[0] + i - 1].uncertainty;
                    
                    if (!threeColInt) {
                        colors[i] = cCol*(1.0 - tempUnc) + uCol*tempUnc; 
                    }
                    else {
                        if (tempUnc < 0.5) {
                            colors[i] = cCol*(1.0 - tempUnc) + cuCol*tempUnc; 
                        }
                        else {
                            colors[i] = cuCol*(1.0 - tempUnc) + uCol*tempUnc; 
                        }
                    }
                
                    if(tempUnc < 0.0) {
                        colors[i] = vec4(0,1,0,1);
                    } else if(tempUnc > 1.0) {
                        colors[i] = vec4(0,1,1,1);
                    }
                }                    
                else if ((colorMode == COLOR_MODE_CHAIN) || (colorMode == COLOR_MODE_AMINOACID)) {
                    colors[i] = atoms[id[0] + i - 1].col; 
                }
                else { // if (colorMode == COLOR_MODE_RESIDUE_DEBUG) {
                    int colorIndex = id[0] + i - 1;
                    colors[i] = vec4(1.0 - float(colorIndex % 5) / 4.0, float(colorIndex % 3) / 2.0, float(colorIndex % 5) / 4.0, 1.0);
                }
            }     
                                 
            vec4 c10 = colors[0] * (1.0 - q) + colors[1] * q;
            vec4 c11 = colors[1] * (1.0 - q1) + colors[2] * q1;
            vec4 c12 = colors[2] * (1.0 - q2) + colors[3] * q2;
        
            vec4 c20 = c10 * (1.0 - q3) + c11 * q3;
            vec4 c21 = c11 * (1.0 - q4) + c12 * q4;
        
            color = c20 * (1.0 - q5) + c21 * q5;
        }
        else // current C-alpha is alph2
        {
            if (colorMode == COLOR_MODE_STRUCT) {
                color = structCol[mytype];
            }     
            else if (colorMode == COLOR_MODE_UNCERTAIN) {
                /*UNCERTAINTY to COLOR coding*/             
                if (!threeColInt) {
                    color = cCol*(1.0 - uncVal[0]) + uCol*uncVal[0]; 
                }
                else {
                    if (uncVal[0] < 0.5) {
                        color = cCol*(1.0 - uncVal[0]) + cuCol*uncVal[0];  
                    }
                    else {
                        color = cuCol*(1.0 - uncVal[0]) + uCol*uncVal[0];  
                    }
                }
            }                    
            else if ((colorMode == COLOR_MODE_CHAIN) || (colorMode == COLOR_MODE_AMINOACID)) {
                color = alph2.col; 
                
                color *= (1.0 - uncVal[0]);  // (TEMP:) When monochrome coloring use lightness as additional "uncertainty variable"            
            }
            else { // if (colorMode == COLOR_MODE_RESIDUE_DEBUG) {
                int colorIndex = id[1];
                color = vec4(1.0 - float(colorIndex % 5) / 4.0, float(colorIndex % 3) / 2.0, float(colorIndex % 5) / 4.0, 1.0);
            }
        }
        
        if(onlyTubes || (mytype == S_BEND) || (mytype == T_H_TURN) || (mytype == NOTDEFINED)) {
            mytype = C_COIL;
        }
        bool changeStart = false; 
        bool changeEnd   = false;
        /*if (uncVal[0] > EPS) {
            changeStart = true; 
            changeEnd   = true;
        }*/
        if (mytype != C_COIL) {
            // check if geometry type changes at the beginning from tube to helix or strand
            int edgeType     = alph1.sortedStruct[0];
            /*if (alph1.unc[edgeType] < (1.0 - EPS)) {
                changeStart = true;
            }
            else*/ if (((edgeType == C_COIL) || (edgeType == S_BEND) || (edgeType == T_H_TURN) || (edgeType == NOTDEFINED))) {
                changeStart = true;
            }
            // check if geometry type changes at the end      
            edgeType       = alph3.sortedStruct[0];        
            /*if (alph3.unc[edgeType] < (1.0 - EPS)) {
                changeEnd = true;
            }
            else*/ if (((edgeType == C_COIL) || (edgeType == S_BEND) || (edgeType == T_H_TURN) || (edgeType == NOTDEFINED))) {
                changeEnd = true;
            }
        }
        
        // DEBUG - mark the position of the arrow heads white
        /*if((mytype == E_EXT_STRAND) || (mytype == B_BRIDGE))
        {
            if(changeEnd) 
                color = vec4(1.0, 1.0, 1.0, 1.0);
        }*/
        
        // calculate geometry and normal
        if (mytype == C_COIL)  // backbone
        {
            // rotate negative dir vector around tangent vector
            float a = 2.0 * M_PI - v * 2.0 * M_PI;            // has to be this angle because of backface culling
            vec3 vrot = nDir * cos(a) + cross(tangent, nDir) * sin(a) + tangent * dot(tangent, nDir) * (1.0 - cos(a)); 
            vrot = normalize(vrot);
        
            mypos.xyz += pipeWidth * vrot;
            n = vrot;
        } 
        else  // alpha helix and beta sheets             
        {
            vec3 curPos = mypos.xyz + nDir * pipeWidth;             // point <width> under the backbone
            vec3 normalDown = nDir;
            vec3 normalLeft = normalize(cross(tangent, -nDir));
            u = gl_TessCoord.x;
            v = gl_TessCoord.y;
            float factor = EPS;
            float thresh = EPS;
        
            float val = 1.0 / float(gl_TessLevelOuter[1]);
            float lutVal = 5.0f;
            if(changeEnd) 
            {
                if((mytype == E_EXT_STRAND) || (mytype == B_BRIDGE))  // arrow heads
                {
                    lutVal = u * 9.0f;   // the size of the LUT is hardcoded to be 10
                    float mymin = arrowLUT[int(floor(lutVal))];
                    float mymax = arrowLUT[int(ceil(lutVal))];
                    float theU  = (lutVal - floor(lutVal)) / (ceil(lutVal) - floor(lutVal));
                    if(ceil(lutVal) - floor(lutVal) < 0.5) 
                    {
                        theU = 0.0f;
                        if(ceil(lutVal) > 8.9f)
                            theU = 1.0f;
                    }
                    lutVal = mix(mymin, mymax, theU);
                }
            }
        
            if(v < val)
            {
                mypos.xyz += nDir * pipeWidth * lutVal + normalLeft * mix(0, pipeWidth, v / val);
                n = normalDown; // down (part 1)
            }
            else if( v >= val && v < 0.5 - val)
            {
                mypos.xyz += nDir * pipeWidth * lutVal + normalLeft * pipeWidth - nDir * mix(0, pipeWidth * lutVal * 2.0, (v - val)/(0.5 - 2 * val) );
                n = normalLeft; // left
            }
            else if(v >= 0.5 - val && v < 0.5 + val)
            {
                mypos.xyz += -nDir * pipeWidth * lutVal + pipeWidth * normalLeft - normalLeft * mix(0, 2 * pipeWidth, (v - (0.5 - val))/(2 * val));
                n = -normalDown; // up
            }
            else if(v >= 0.5 + val && v < 1.0 - val)
            {
                mypos.xyz += -nDir * pipeWidth * lutVal - pipeWidth * normalLeft + nDir * mix(0, pipeWidth * lutVal * 2.0, (v- (0.5 + val))/(0.5 - 2 * val));
                n = -normalLeft; // right
            }
            else 
            {
                mypos.xyz += nDir * pipeWidth * lutVal - normalLeft * mix(pipeWidth, 0, ((v - (1.0 - val))/ val));
                n = normalDown; // down (part 2)
            }
        
            // if change in secondary structure is at the beginning: make transition from spline to tube surface
            if((changeStart) && (u < factor)) {
                vec3 surfdir = mypos.xyz - savepos.xyz;    // direction from spline to tube surface.
                vec3 newpos = savepos.xyz + normalize(surfdir) * pipeWidth;
                mypos.xyz = mix(newpos, mypos.xyz, u / factor);
                mypos.xyz += thresh * (-tangent);
                n = -tangent;
            }
            
            // if change in secondary structure is at the end: make transition from spline to tube surface
            if((changeEnd) && (u > (1.0 - factor))) {
                vec3 surfdir = mypos.xyz - savepos.xyz;   // direction from spline to tube surface.
                vec3 newpos = savepos.xyz + normalize(surfdir) * pipeWidth;
                mypos.xyz = mix(mypos.xyz, newpos, (u - (1.0 - factor))/ factor);
                mypos.xyz += thresh * tangent;
                n = tangent;
            }
        }
        
        // apply surface displacement parameter
        u = gl_TessCoord.x;
        v = gl_TessCoord.y;
        float ampLevel  = uncDistor[0]/10.0;  
        float distorLev = uncDistor[1];       
        float uncFac    = uncVal[0];
        // displacement of geometry and adjustment of normal vector
        float uNdisp = 0.0;
        float vNdisp = 0.0;
        float disp   = 0.0;
        // UNCERTAINTY visualisation
        if (uncVisMode == UNC_VIS_SIN_U) { // SINUS U
            float uDisp   = u * ceil(distorLev * uncFac);
            float fac     = uncFac * ampLevel; 
            disp          = fac * unitSin(uDisp);             
            
            float uM      = fac * cos(uDisp);                          // gradient = derivation of fac*sin(uDisp)
            if (uM >= EPS) {
                vec2  uNormal = normalize(vec2(1.0, -1.0/uM));          // new normal from gradient in UV-space
                float alpha   = acos(dot(vec2(0.0, 1.0), uNormal));     // angle between n (in UV) and uNormal
                float beta    = M_PI/2.0 - alpha;                       // angle between uNormal and x-axis
                uNdisp        = 1.0/sin(beta) * sin(alpha);             // with |n|=1
            }
            mypos.xyz    += (disp * n);           
            n             = normalize(n + uNdisp * tangent);                
        }
        else if (uncVisMode == UNC_VIS_SIN_V) { // SINUS V    
            float vDisp   = v * distorLev * uncFac;
            float fac     = uncFac * ampLevel;                
            disp          = fac * unitSin(vDisp);     
            
            float vM          = fac * cos(vDisp);                       // gradient = derivation of fac*sin(uDisp)            
            if (vM >= EPS) {
                vec2  vNormal = normalize(vec2(1.0, -1.0/vM));          // new normal from gradient in UV-space
                float alpha   = acos(dot(vec2(0.0, 1.0), vNormal));     // angle between n (in UV) and uNormal
                float beta    = M_PI/2.0 - alpha;                       // angle between uNormal and x-axis
                vNdisp        = 1.0/sin(beta) * sin(alpha);             // with |n|=1
            }
            
            if (((1.0 - EPS) > u) && (u > EPS)) {
                mypos.xyz    += (disp * n);
            }        
            n             = normalize(n + vNdisp * normalize(cross(n, tangent)));                
        }    
        else if (uncVisMode == UNC_VIS_SIN_UV) {  // SINUS U,V
            float uDisp   = u * ceil(distorLev * uncFac);
            float vDisp   = v * distorLev * uncFac;
            float fac     = uncFac * ampLevel;    
            float dispU   = fac * unitSin(uDisp);
            float dispV   = fac * unitSin(vDisp);
            disp          = (dispU + dispV)/2.0;
            float uM      = fac * cos(uDisp);                          // gradient = derivation of fac*sin(uDisp)
            if (uM >= EPS) {
                vec2  uNormal = normalize(vec2(1.0, -1.0/uM));          // new normal from gradient in UV-space
                float alpha   = acos(dot(vec2(0.0, 1.0), uNormal));     // angle between n (in UV) and uNormal
                float beta    = M_PI/2.0 - alpha;                       // angle between uNormal and x-axis
                uNdisp        = 1.0/sin(beta) * sin(alpha);             // with |n|=1
            }
            
            float vM          = fac * cos(vDisp);                       // gradient = derivation of fac*sin(uDisp)            
            if (vM >= EPS) {
                vec2  vNormal = normalize(vec2(1.0, -1.0/vM));          // new normal from gradient in UV-space
                float alpha   = acos(dot(vec2(0.0, 1.0), vNormal));     // angle between n (in UV) and uNormal
                float beta    = M_PI/2.0 - alpha;                       // angle between uNormal and x-axis
                vNdisp        = 1.0/sin(beta) * sin(alpha);             // with |n|=1
            }
            
            if (((1.0 - EPS) > u) && (u > EPS)) {                   // for smooth transition between amino-acids
                mypos.xyz    += (disp * n);
            }
            n = normalize(n + uNdisp * tangent + vNdisp * normalize(cross(n, tangent)));
        }
        else if (uncVisMode == UNC_VIS_TRI_U) {  // TRIANGLE U
            float uMod    = u * ceil(distorLev * uncFac);
            float fac     = uncFac * ampLevel;                
            disp          = fac * unitTri(uMod);
            float uM      = fac;                                        // gradient = derivation of fac*sin(uDisp)
            if ((0.5 - (uMod - floor(uMod))) < 0.0) {
                uM       *= -1.0;
            }
                         
            if (uM >= EPS) {
                vec2  uNormal = normalize(vec2(1.0, -1.0/uM));          // new normal from gradient in UV-space
                float alpha   = acos(dot(vec2(0.0, 1.0), uNormal));     // angle between n (in UV) and uNormal
                float beta    = M_PI/2.0 - alpha;                       // angle between uNormal and x-axis
                uNdisp        = 1.0/sin(beta) * sin(alpha);             // with |n|=1
            }
            
            mypos.xyz    += (n * disp); 
            n             = normalize(n + uNdisp * tangent);    
        }
        else if (uncVisMode == UNC_VIS_TRI_UV) {  // TRIANGLE U,V
            float uMod    = u * ceil(distorLev * uncFac);
            float vMod    = v * distorLev * uncFac;
            float uFunc   = 0.5 - (uMod - floor(uMod));
            float vFunc   = 0.5 - (vMod - floor(vMod));
            float fac     = uncFac * ampLevel;                
            float uTri    = fac * unitTri(uMod);
            float vTri    = fac * unitTri(vMod);
            float uM      = fac;
            if ((0.5 - (uMod - floor(uMod))) < 0.0) {
                uM       *= -1.0;
            }
            if (uM >= EPS) {
                vec2  uNormal = normalize(vec2(1.0, -1.0/uM));          // new normal from gradient in UV-space
                float alpha   = acos(dot(vec2(0.0, 1.0), uNormal));     // angle between n (in UV) and uNormal
                float beta    = M_PI/2.0 - alpha;                       // angle between uNormal and x-axis
                uNdisp        = 1.0/sin(beta) * sin(alpha);             // with |n|=1
            }
            
            float vM      = fac;
            if ((0.5 - (vMod - floor(vMod))) < 0.0) {
                vM       *= -1.0;
            }
            if (vM >= EPS) {
                vec2  vNormal = normalize(vec2(1.0, -1.0/vM));          // new normal from gradient in UV-space
                float alpha   = acos(dot(vec2(0.0, 1.0), vNormal));     // angle between n (in UV) and uNormal
                float beta    = M_PI/2.0 - alpha;                       // angle between uNormal and x-axis
                vNdisp        = 1.0/sin(beta) * sin(alpha);             // with |n|=1
            }
            
            float factor = EPS;
            if (((1.0 - factor) > u) && (u > factor)) {
                mypos.xyz   += (n * (uTri + vTri)/2.0);  
            }
            n = normalize(n + uNdisp * tangent + vNdisp * normalize(cross(n, tangent)));
        }
        // OUTLINING
        if ((outlineMode == OUTLINE_FULL_UNCERTAIN) || (outlineMode == OUTLINE_FULL_CERTAIN)) {
            // invert uncertainty factor if certainty should be shown
            if (outlineMode == OUTLINE_FULL_CERTAIN) {
                uncFac = 1.0 - uncFac;
            }
            // shift and scale vertex position in normal direction
            mypos.xyz = mypos.xyz + (outlineScale/10.0)*(uncFac)*n; 

            // add caps for OUTLINING to tube strucutres and if dithering is enabled to helix and strand too
            if (mytype == C_COIL)  // backbone
            {
                float factor = EPS;
                if(u > (1.0 - factor)) {
                    vec3 surfdir = mypos.xyz - savepos.xyz;             // direction from spline to tube surface.
                    vec3 newpos = savepos.xyz + normalize(surfdir) * pipeWidth;
                    mypos.xyz = mix(mypos.xyz, newpos, (u - (1.0 - factor))/ factor);
                    mypos.xyz += factor * tangent;
                }
                else if(u < factor) {
                    vec3 surfdir = mypos.xyz - savepos.xyz;             // direction from spline to tube surface.
                    vec3 newpos = savepos.xyz + normalize(surfdir) * pipeWidth;
                    mypos.xyz = mix(newpos, mypos.xyz, u / factor);
                    mypos.xyz += factor * (-tangent);
                }  
            }  
        }
        gl_Position = mypos;
    }
    
    if ((ditherCount > 0) && (alphaBlending == 1)) {
        CAlpha alpha = atoms[id[1]]; 
        uncVal[0] = alpha.unc[alpha.sortedStruct[0]];
    }
}
