#version 140

#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_EXT_gpu_shader4 : require
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_gpu_shader_fp64 : enable

layout (std430, binding=1) buffer x_buffer {
    double x[];
};

layout (std430, binding=2) buffer y_buffer {
    double y[];
};

layout (std430, binding=3) buffer z_buffer {
    double z[];
};


#define CLIP
#define DEPTH
#define SMALL_SPRITE_LIGHTING
//#define CALC_CAM_SYS

#ifdef DEBUG
#undef CLIP
#define RETICLE
#define AXISHINTS
#endif // DEBUG

//#define BULLSHIT

#ifndef FLACH
#define FLACH
#endif

uniform vec4 viewAttr;

uniform float scaling;

#ifndef CALC_CAM_SYS
uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;
#endif // CALC_CAM_SYS

// clipping plane attributes
uniform vec4 clipDat;
uniform vec4 clipCol;
uniform int instanceOffset;


uniform mat4 modelview;
uniform mat4 project;

out mat4 MVP;
out mat4 MVPinv;
out mat4 MVPtransp;

// uniform mat4 MVinv;
// uniform mat4 MVP;
// uniform mat4 MVPinv;
// uniform mat4 MVPtransp;

uniform vec4 lightPos_u;

uniform vec4 inConsts1;
attribute float colIdx;
uniform sampler1D colTab;

out vec4 objPos;
out vec4 camPos;
out vec4 lightPos;
out float squarRad;
out float rad;
out vec4 vertColor;

#ifdef DEFERRED_SHADING
out float pointSize;
#endif

#ifdef RETICLE
out vec2 centerFragment;
#endif // RETICLE

#define CONSTRAD inConsts1.x
#define MIN_COLV inConsts1.y
#define MAX_COLV inConsts1.z
#define COLTAB_SIZE inConsts1.w

void main(void) {
    mat4 MVinv = inverse(modelview);
    MVP = project*modelview;
    MVPinv = inverse(MVP);
    MVPtransp = transpose(MVP);

    rad = CONSTRAD;

    float theColIdx;
    vec4 theColor = vec4(1.0);
    vec4 inPos = vec4(x[gl_VertexID], y[gl_VertexID], z[gl_VertexID], 1.0);

    // remove the sphere radius from the w coordinates to the rad varyings
    //vec4 inPos = gl_Vertex;
    //rad = (CONSTRAD < -0.5) ? inPos.w : CONSTRAD;
    //inPos.w = 1.0;
    //inPos = vec4(0.0, 0.0, 0.0, 1.0);
    //rad = 1.0;
    float cid = MAX_COLV - MIN_COLV;
    if (cid < 0.000001) {
        vertColor = theColor;
    } else {
        cid = (theColIdx - MIN_COLV) / cid;
        cid = clamp(cid, 0.0, 1.0);
        
        cid *= (1.0 - 1.0 / COLTAB_SIZE);
        cid += 0.5 / COLTAB_SIZE;
        
        vertColor = texture(colTab, cid);
    }

    rad *= scaling;

    squarRad = rad * rad;

    // object pivot point in object space    
    objPos = inPos; // no w-div needed, because w is 1.0 (Because I know)

    // calculate cam position
    camPos = MVinv[3]; // (C) by Christoph
    camPos.xyz -= objPos.xyz; // cam pos to glyph space

    // calculate light position in glyph space
    lightPos = MVinv * lightPos_u;

    // clipping
    float od = clipDat.w - 1.0;
    if (any(notEqual(clipDat.xyz, vec3(0, 0, 0)))) {
        od = dot(objPos.xyz, clipDat.xyz) - rad;
    }

    // Sphere-Touch-Plane-Approach™
    vec2 winHalf = 2.0 / viewAttr.zw; // window size

    vec2 d, p, q, h, dd;

    // get camera orthonormal coordinate system
    vec4 tmp;

#ifdef CALC_CAM_SYS
    // camera coordinate system in object space
    tmp = MVinv[3] + MVinv[2];
    vec3 camIn = normalize(tmp.xyz);
    tmp = MVinv[3] + MVinv[1];
    vec3 camUp = tmp.xyz;
    vec3 camRight = normalize(cross(camIn, camUp));
    camUp = cross(camIn, camRight);
#endif // CALC_CAM_SYS

    vec2 mins, maxs;
    vec3 testPos;
    vec4 projPos;

    // projected camera vector
    vec3 c2 = vec3(dot(camPos.xyz, camRight), dot(camPos.xyz, camUp), dot(camPos.xyz, camIn));

    vec3 cpj1 = camIn * c2.z + camRight * c2.x;
    vec3 cpm1 = camIn * c2.x - camRight * c2.z;

    vec3 cpj2 = camIn * c2.z + camUp * c2.y;
    vec3 cpm2 = camIn * c2.y - camUp * c2.z;
    
    d.x = length(cpj1);
    d.y = length(cpj2);

    dd = vec2(1.0) / d;

    p = squarRad * dd;
    q = d - p;
    h = sqrt(p * q);
    //h = vec2(0.0);
    
    p *= dd;
    h *= dd;

    cpj1 *= p.x;
    cpm1 *= h.x;
    cpj2 *= p.y;
    cpm2 *= h.y;

    // TODO: rewrite only using four projections, additions in homogenous coordinates and delayed perspective divisions.
    testPos = objPos.xyz + cpj1 + cpm1;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = projPos.xy;
    maxs = projPos.xy;

    testPos -= 2.0 * cpm1;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    testPos = objPos.xyz + cpj2 + cpm2;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    testPos -= 2.0 * cpm2;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);
    
    testPos = objPos.xyz - camIn * rad;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;

    gl_Position = vec4((mins + maxs) * 0.5, projPos.z, (od > clipDat.w) ? 0.0 : 1.0);
    maxs = (maxs - mins) * 0.5 * winHalf;
    gl_PointSize = max(maxs.x, maxs.y) + 0.5;
#ifdef DEFERRED_SHADING
    pointSize = gl_PointSize;
#endif

#ifdef SMALL_SPRITE_LIGHTING
    // for normal crowbaring on very small sprites
    lightPos.w = (clamp(gl_PointSize, 1.0, 5.0) - 1.0) / 4.0;
#else
    lightPos.w = 1.0;
#endif // SMALL_SPRITE_LIGHTING
    
#ifdef RETICLE
    centerFragment = gl_Position.xy / gl_Position.w;
#endif // RETICLE
    //gl_Position = MVP * vec4(inPos.xyz, 1.0);
    //gl_Position /= gl_Position.w;
    //gl_PointSize = 8.0;

}