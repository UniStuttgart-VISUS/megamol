// TODO: Implementation is wrong! Does positional Lighting instead of directional lighting!

// ray:      the eye to fragment ray vector
// normal:   the normal of this fragment
// lightPos: the position of the light source
// color:    the base material color
vec3 LocalLighting(const in vec3 ray, const in vec3 normal, const in vec3 lightPos, const in vec3 color) {
    // TODO: rewrite!
    vec3 lightDir = normalize(lightPos);

    vec4 lightparams = vec4(0.2, 0.8, 0.9, 10.0);
#define LIGHT_AMBIENT lightparams.x
#define LIGHT_DIFFUSE lightparams.y
#define LIGHT_SPECULAR lightparams.z
#define LIGHT_EXPONENT lightparams.w
    float nDOTl = dot(normal, lightDir);

    vec3 r = normalize(2.0 * vec3(nDOTl) * normal - lightDir);
    return LIGHT_AMBIENT  * color 
         + LIGHT_DIFFUSE  * color * max(nDOTl, 0.0) 
         + LIGHT_SPECULAR * vec3(pow(max(dot(r, -ray), 0.0), LIGHT_EXPONENT));
}
