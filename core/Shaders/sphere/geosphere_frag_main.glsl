#extension GL_ARB_explicit_attrib_location : require   // glsl version 130

uniform vec4 viewAttr; // TODO: check fragment position if viewport starts not in (0, 0)

uniform mat4 MVPinv;
uniform mat4 MVPtransp;

// clipping plane attributes
uniform vec4 clipDat;
uniform vec4 clipCol;

in vec4 color;
in vec4 objPos;
in vec4 camPos;
in vec4 lightPos;
in float rad;
in float squarRad;

#ifndef CALC_CAM_SYS
uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;
#endif // CALC_CAM_SYS

layout(location = 0) out vec4 outColor;


// Forward declaration of lighting function.
vec3 LocalLighting(const in vec3 ray, const in vec3 normal, const in vec3 lightPos, const in vec3 color);


void main(void) {
    
    vec4 coord;
    vec3 ray;
    float lambda;
    vec3 colOut;
    vec3 sphereintersection = vec3( 0.0);
    vec3 normal;

    // transform fragment coordinates from window coordinates to view coordinates.
    coord = gl_FragCoord 
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0) 
        + vec4(-1.0, -1.0, -1.0, 1.0);
    
    // transform fragment coordinates from view coordinates to object coordinates.
    coord = MVPinv * coord;
    coord /= coord.w;
    coord -= objPos; // ... and to glyph space
    
    // calc the viewing ray
    ray = normalize(coord.xyz - camPos.xyz);

    // calculate the geometry-ray-intersection
    float d1 = -dot(camPos.xyz, ray);                       // projected length of the cam-sphere-vector onto the ray
    float d2s = dot(camPos.xyz, camPos.xyz) - d1 * d1;      // off axis of cam-sphere-vector and ray
    float radicand = squarRad - d2s;                        // square of difference of projected length and lambda
#ifdef CLIP
    lambda = d1 - sqrt(radicand);                           // lambda

    float radicand2 = 0.0;
    if( radicand < 0.0 ) {
        discard;
    } else {
        // chose color for lighting
        colOut = color.rgb;
        sphereintersection = lambda * ray + camPos.xyz;    // intersection point
        // "calc" normal at intersection point
        normal = sphereintersection / rad;
    }    
#endif // CLIP

    if (any(notEqual(clipDat.xyz, vec3(0, 0, 0)))) {
        vec3 planeNormal = normalize(clipDat.xyz);
        vec3 clipPlaneBase = planeNormal * clipDat.w;
        float d = -dot(planeNormal, clipPlaneBase - objPos.xyz);
        float dist1 = dot(sphereintersection, planeNormal) + d;
        float dist2 = d;
        float t = -(dot(planeNormal, camPos.xyz) + d) / dot(planeNormal, ray);
        vec3 planeintersect = camPos.xyz + t * ray;
        if (dist1 > 0.0) {
            if (dist2 < rad) {
                if (length(planeintersect) < rad) {
                    sphereintersection = planeintersect;
                    normal = planeNormal;
                    colOut = mix(colOut, clipCol.rgb, clipCol.a);
                } else {
                    discard;
                }
            } else {
                discard;
            }
        }
    }

    // phong lighting with directional light
    //outColor = vec4(LocalLighting(ray, normal, lightPos.xyz, color), 1.0);
    outColor = vec4(LocalLighting(ray, normal, lightPos.xyz, colOut), color.w);
    
    // calculate depth
#ifdef DEPTH
    vec4 Ding = vec4(sphereintersection + objPos.xyz, 1.0);
    float depth = dot(MVPtransp[2], Ding);
    float depthW = dot(MVPtransp[3], Ding);
    gl_FragDepth = ((depth / depthW) + 1.0) * 0.5;
    
#ifndef CLIP
    gl_FragDepth = (radicand < 0.0) ? 1.0 : ((depth / depthW) + 1.0) * 0.5;
    outColor.rgb = (radicand < 0.0) ? color.rgb : outColor.rgb;
#endif // CLIP

#endif // DEPTH

    
#ifdef RETICLE
    coord = gl_FragCoord 
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0) 
        + vec4(-1.0, -1.0, -1.0, 1.0);
    if (min(abs(coord.x - centerFragment.x), abs(coord.y - centerFragment.y)) < 0.002) {
        //outColor.rgb = vec3(1.0, 1.0, 0.5);
        outColor.rgb += vec3(0.3, 0.3, 0.5);
    }
#endif // RETICLE

//outColor.rgb = normal; // DEBUG
//outColor = color; // DEBUG
}