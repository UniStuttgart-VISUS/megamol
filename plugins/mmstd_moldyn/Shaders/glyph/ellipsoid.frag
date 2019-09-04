#define DEPTH
#define CLIP

in vec4 objPos;
in vec4 camPos;
in vec4 lightPos;
in vec4 vertColor;

in vec3 invRad;

in mat3 rotMat;
in vec3 rotMatT0;
in vec3 rotMatT1; // rotation matrix from the quaternion
in vec3 rotMatT2;
in mat3 rotMatIT;

layout (location = 0) out vec4 out_frag_color;

void main() {
    vec4 coord;
    vec3 ray, tmp;
    float lambda;

    // transform fragment coordinates from window coordinates to view coordinates.
    coord = gl_FragCoord 
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0) 
        + vec4(-1.0, -1.0, -1.0, 1.0);
    

    // transform fragment coordinates from view coordinates to object coordinates.
    coord = MVP_I * coord;
    coord /= coord.w;
    coord -= objPos; // ... and move
    // ... and rotate mit rotMat transposed
    ray = rotMatT0 * coord.x + rotMatT1 * coord.y + rotMatT2 * coord.z;

    ray *= invRad;

    // calc the viewing ray
    ray = normalize(ray - camPos.xyz);


    // calculate the geometry-ray-intersection (sphere with radius = 1)
    float d1 = -dot(camPos.xyz, ray);                       // projected length of the cam-sphere-vector onto the ray
    float d2s = dot(camPos.xyz, camPos.xyz) - d1 * d1;      // off axis of cam-sphere-vector and ray
    float radicand = 1.0 - d2s;                             // square of difference of projected length and lambda
    #ifdef CLIP
    if (radicand < 0.0) { discard; }
    #endif // CLIP
    lambda = d1 - sqrt(radicand);                           // lambda
    vec3 sphereintersection = lambda * ray + camPos.xyz;    // intersection point


    // "calc" normal at intersection point
    vec3 normal = sphereintersection;
    #ifdef SMALL_SPRITE_LIGHTING
    normal = mix(-ray, normal, lightPos.w);
    #endif // SMALL_SPRITE_LIGHTING
    vec3 color = vertColor.rgb;
    vec3 s = step(normal, vec3(0));
    color = s;

    normal = rotMatIT * normal;
    ray = rotMatIT * ray;
    // chose color for lighting
    // Because of some reason color is either not set or not accessible
    
    // Set the default global color from case 1 in the renderer manually 
    //color = vec3(0.698, 0.698, 0.698);
    
    // calculate depth
    // TODO: rewrite with rotation with untransposed
    #ifdef DEPTH
    tmp = sphereintersection / invRad;

    //sphereintersection = rotMatT0 * tmp.x + rotMatT1 * tmp.y + rotMatT2 * tmp.z;
    //sphereintersection.x = rotMatT0.x * tmp.x + rotMatT1.x * tmp.y + rotMatT2.x * tmp.z;
    //sphereintersection.y = rotMatT0.y * tmp.x + rotMatT1.y * tmp.y + rotMatT2.y * tmp.z;
    //sphereintersection.z = rotMatT0.z * tmp.x + rotMatT1.z * tmp.y + rotMatT2.z * tmp.z;
    sphereintersection.x = dot(rotMatT0, tmp.xyz);
    sphereintersection.y = dot(rotMatT1, tmp.xyz);
    sphereintersection.z = dot(rotMatT2, tmp.xyz);

    // phong lighting with directional light, colors are hardcoded as they're inaccessible for some reason
    out_frag_color = vec4(LocalLighting(ray, normal, lightPos.xyz, color), 1.0);
    //out_frag_color = vec4(LocalLighting(ray, normalize(sphereintersection), lightPos.xyz, color), 1.0);


    sphereintersection += objPos.xyz;

    vec4 Ding = vec4(sphereintersection, 1.0);
    float depth = dot(MVP_T[2], Ding);
    float depthW = dot(MVP_T[3], Ding);
    gl_FragDepth = ((depth / depthW) + 1.0) * 0.5;
    #ifndef CLIP
    gl_FragDepth = (radicand < 0.0) ? 1.0 : ((depth / depthW) + 1.0) * 0.5;
    out_frag_color.rgb = (radicand < 0.0) ? vertColor.rgb : out_frag_color.rgb;
    #endif // CLIP
    #endif // DEPTH

    #ifdef RETICLE
    coord = gl_FragCoord 
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0) 
        + vec4(-1.0, -1.0, -1.0, 1.0);
    if (min(abs(coord.x - centerFragment.x), abs(coord.y - centerFragment.y)) < 0.002) {
    	//out_frag_color.rgb = vec3(1.0, 1.0, 0.5);
    	out_frag_color.rgb += vec3(0.3, 0.3, 0.5);
    }
    #endif //RETICLE
}