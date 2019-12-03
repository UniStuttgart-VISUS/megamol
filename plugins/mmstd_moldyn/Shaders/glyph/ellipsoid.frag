in vec4 objPos;
in vec3 camPos;
in vec4 lightPos;
in vec4 vertColor;

in vec3 invRad;

in flat vec3 dirColor;

in flat vec3 transformedNormal;
in vec3 viewRay;

in mat3 rotate_world_into_tensor;
in mat3 rotate_points;

out layout(location = 0) vec3 albedo_out;
out layout(location = 1) vec3 normal_out;
out layout(location = 2) float depth_out;

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
    ray = rotate_world_into_tensor * coord.xyz;

    ray *= invRad;

    // calc the viewing ray
    ray = normalize(ray - camPos.xyz);
    // calculate the geometry-ray-intersection (sphere with radius = 1)
    float d1 = -dot(camPos.xyz, ray);                       // projected length of the cam-sphere-vector onto the ray
    float d2s = dot(camPos.xyz, camPos.xyz) - d1 * d1;      // off axis of cam-sphere-vector and ray
    float radicand = 1.0 - d2s;                             // square of difference of projected length and lambda
    if (radicand < 0.0) { discard; }
    lambda = d1 - sqrt(radicand);                           // lambda
    vec3 sphereintersection = lambda * ray + camPos.xyz;    // intersection point


    // "calc" normal at intersection point
    vec3 normal = sphereintersection;
    vec3 color = vertColor.rgb;
    vec3 s = step(normal, vec3(0));
    color = s;

    normal = rotate_points * normal;
    ray = rotate_points * ray;

    tmp = sphereintersection / invRad;

    //sphereintersection.x = dot(rotMatT0, tmp.xyz);
    //sphereintersection.y = dot(rotMatT1, tmp.xyz);
    //sphereintersection.z = dot(rotMatT2, tmp.xyz);
    sphereintersection = rotate_world_into_tensor * tmp.xyz;

    // phong lighting with directional light, colors are hardcoded as they're inaccessible for some reason
    //out_frag_color = vec4(LocalLighting(ray, normal, lightPos.xyz, color), 1.0);
    //out_frag_color = vec4(LocalLighting(ray, normalize(sphereintersection), lightPos.xyz, color), 1.0);

    albedo_out = mix(dirColor, vertColor.rgb, colorInterpolation);
    normal_out = normal;

    sphereintersection += objPos.xyz;

    vec4 Ding = vec4(sphereintersection, 1.0);
    float depth = dot(MVP_T[2], Ding);
    float depthW = dot(MVP_T[3], Ding);
    depth_out = ((depth / depthW) + 1.0) * 0.5;

    //depth_out = gl_FragCoord.z;
    //albedo_out = mix(dirColor, vertColor.rgb, colorInterpolation);
    //normal_out = transformedNormal;
}