
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
