#version 430
        
layout(quads, equal_spacing, ccw) in;
//layout(isolines, equal_spacing) in;

uniform float minDistance = 1.0;
uniform float maxDistance = 1.0;

uniform bool interpolateColors = false;

uniform vec4 coilColor = vec4(0.7, 0.7, 0.7, 1);
uniform vec4 helixColor = vec4(1, 0, 0, 1);
uniform vec4 sheetColor = vec4(0, 0, 1, 1);
uniform vec4 turnColor = vec4(0, 1, 0, 1);

uniform float pipeWidth = 1.0;

#define M_PI 3.1415926535897932384626433832795

in int id[];
out vec4 color;
out vec3 n;

struct CAlpha {
    vec4 pos;
    vec3 dir;
    int type;
};

layout(std430, binding = 2) buffer shader_data {
    CAlpha atoms[];
};

const float arrowLUT[10] = {
    5.0, 5.0, 5.0, 8.0, 7.0, 6.0, 4.5, 3.0, 2.0, 1.0
};

void main() {
    
    CAlpha alph0 = atoms[id[0] - 1];
    CAlpha alph1 = atoms[id[0]];
    CAlpha alph2 = atoms[id[1]];
    CAlpha alph3 = atoms[id[1] + 1];
    
    vec4 p0 = alph0.pos;
    vec4 p1 = alph1.pos;
    vec4 p2 = alph2.pos;
    vec4 p3 = alph3.pos;
    
    vec4 colors[4];
    
    for(int i = 0; i < 4; i++) {
        int mytype = atoms[id[0] + i - 1].type;
        
        if(mytype == 1) { // beta sheet
            colors[i] = sheetColor;
        } else if(mytype == 2) { // alpha helix
            colors[i] = helixColor;
        } else if(mytype == 3) { // turn
            colors[i] = turnColor;
        } else { // unclassified
            colors[i] = coilColor;
        }
    }
    
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
    
    // interpolate colors, too
    if(interpolateColors) {
        vec4 c10 = colors[0] * (1.0 - q) + colors[1] * q;
        vec4 c11 = colors[1] * (1.0 - q1) + colors[2] * q1;
        vec4 c12 = colors[2] * (1.0 - q2) + colors[3] * q2;
        
        vec4 c20 = c10 * (1.0 - q3) + c11 * q3;
        vec4 c21 = c11 * (1.0 - q4) + c12 * q4;
        
        color = c20 * (1.0 - q5) + c21 * q5;
    } else {
        color = colors[2];
    }
    
    // interpolate directions
    vec3 dir10 = mix(alph0.dir, alph1.dir, q);
    vec3 dir11 = mix(alph1.dir, alph2.dir, q1);
    vec3 dir12 = mix(alph2.dir, alph3.dir, q2);
    
    vec3 dir20 = mix(dir10, dir11, q3);
    vec3 dir21 = mix(dir11, dir12, q4);
    
    vec3 dir = normalize(mix(dir20, dir21, q5));
    
    vec3 tangent = normalize(d21.xyz - d20.xyz);
    vec3 normal = normalize(cross(tangent, dir));
    
    // compute corrected direction that is truly orthogonal to the tangent vector
    vec3 nDir = dir;
    nDir = dir -  dot(dir, tangent) * tangent;
    nDir = -normalize(nDir);
    
    int mytype = atoms[id[1]].type;
    int mytype2 = atoms[id[1] + 1].type;
    
    bool change = false;
    int where = 3;
    int lasttype = atoms[id[1] + 1].type;
    
    for(int i = 3; i > -1; i = i - 1 ) {
        int mytype = atoms[id[0] + i - 1].type;
        
        if(mytype != lasttype) {
            change = true;
            where = i;
        }
        
        lasttype = mytype;
    }
    
    /*if(mytype == 1) 
    {
        if(change && ((where == 2))) // the position of the arrow heads
            color = vec4(1.0, 1.0, 1.0, 1.0);
    }*/

    if(mytype != 1 && mytype != 2) { // backbone 
        // rotate negative dir vector around tangent vector
        float a = 2.0 * M_PI - v * 2.0 * M_PI; // has to be this angle becaus of backface culling
        vec3 vrot = nDir * cos(a) + cross(tangent, nDir) * sin(a) + tangent * dot(tangent, nDir) * (1.0 - cos(a)); 
        vrot = normalize(vrot);
        
        mypos.xyz += pipeWidth * vrot;
        n = vrot;
    } else { // alpha helix and beta sheets
        vec3 curPos = mypos.xyz + nDir * pipeWidth; // point <width> under the backbone
        vec3 normalDown = nDir;
        vec3 normalLeft = normalize(cross(tangent, -nDir));

        u = gl_TessCoord.x;
        float factor = 0.01;
        float thresh = 0.01;
        
        float val = 1.0 / float(gl_TessLevelOuter[1]);

        float lutVal = 5.0f;
        if(change && (where == 2)) {
            if(mytype == 1) { // arrow heads
                //color = vec4(1.0, 1.0, 1.0, 1.0);

                // the size of the LUT is hardcoded to be 10
                lutVal = u * 9.0f;
                float mymin = arrowLUT[int(floor(lutVal))];
                float mymax = arrowLUT[int(ceil(lutVal))];
                float theU = (lutVal - floor(lutVal)) / (ceil(lutVal) - floor(lutVal));

                if(ceil(lutVal) - floor(lutVal) < 0.5) {
                    theU = 0.0f;

                    if(ceil(lutVal) > 8.9f)
                        theU = 1.0f;
                }

                lutVal = mix(mymin, mymax, theU);
            }
        }
        
        if(v < val) {
            mypos.xyz += nDir * pipeWidth * lutVal + normalLeft * mix(0, pipeWidth, v / val);
            n = normalDown;
        } else if( v >= val && v < 0.5 - val) {
            mypos.xyz += nDir * pipeWidth * lutVal + normalLeft * pipeWidth - nDir * mix(0, pipeWidth * lutVal * 2.0, (v - val)/(0.5 - 2 * val) );
            n = normalLeft;
        } else if(v >= 0.5 - val && v < 0.5 + val) {
            mypos.xyz += -nDir * pipeWidth * lutVal + pipeWidth * normalLeft - normalLeft * mix(0, 2 * pipeWidth, (v - (0.5 - val))/(2 * val));
            n = -normalDown;
        } else if(v >= 0.5 + val && v < 1.0 - val) {
            mypos.xyz += -nDir * pipeWidth * lutVal - pipeWidth * normalLeft + nDir * mix(0, pipeWidth * lutVal * 2.0, (v- (0.5 + val))/(0.5 - 2 * val));
            n = -normalLeft;
        } else {
            mypos.xyz += nDir * pipeWidth * lutVal - normalLeft * mix(pipeWidth, 0, ((v - (1.0 - val))/ val));
            n = normalDown;
        }
        
        if(change && (where == 2)) {
            vec3 surfdir = mypos.xyz - savepos.xyz; // direction from spline to tube surface.
            vec3 newpos = savepos.xyz + normalize(surfdir) * pipeWidth;
            
            if(u > (1.0 - factor)) {
                mypos.xyz = mix(mypos.xyz, newpos, (u - (1.0 - factor))/ factor);
                mypos.xyz += thresh * tangent;
                n = tangent;
            }
        }
        
        if(change && (where == 1)) {
            vec3 surfdir = mypos.xyz - savepos.xyz; // direction from spline to tube surface.
            vec3 newpos = savepos.xyz + normalize(surfdir) * pipeWidth;
            
            if(u < factor)
            {
                mypos.xyz = mix(newpos, mypos.xyz, u / factor);
                mypos.xyz += thresh * (-tangent);
                n = -tangent;
            }
        }
    }
    
    gl_Position = mypos;
}
