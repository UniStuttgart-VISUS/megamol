uniform vec3 inObjLightDir;
uniform vec3 inObjCamPos;
				
vec3 evaluateLocalLighting(const in vec3 objPos, const in vec3 objNormal, const in vec3 matColor)
{
	vec4 lightparams = vec4(0.2, 0.8, 0.4, 10.0);
#define LIGHT_AMBIENT lightparams.x
#define LIGHT_DIFFUSE lightparams.y
#define LIGHT_SPECULAR lightparams.z
#define LIGHT_EXPONENT lightparams.w
	float nDOTl = dot(objNormal, inObjLightDir);

	vec3 reflected = normalize(2.0 * vec3(nDOTl) * objNormal - inObjLightDir);
	vec3 posToCam = -normalize(objPos - inObjCamPos);
	
	return LIGHT_AMBIENT * matColor + 
		   LIGHT_DIFFUSE * matColor * max(nDOTl, 0.0) + 
		   LIGHT_SPECULAR * vec3(pow(max(dot(reflected, posToCam), 0.0), LIGHT_EXPONENT));
}