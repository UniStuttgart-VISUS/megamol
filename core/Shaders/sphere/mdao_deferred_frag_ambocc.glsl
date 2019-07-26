uniform sampler3D inDensityTex;
uniform float inAmbVolShortestEdge;
uniform float inAmbVolMaxLod;
uniform float inAOOffset;
uniform float inAOStrength;
uniform float inAOConeLength;

uniform vec3 inBoundsMin;
uniform vec3 inBoundsSize;


// Find a perpendicular vector to v
vec3 perp(vec3 v)
{
	vec3 b = cross(v, vec3(1, 0, 0));
	return normalize(b); 
}


float evaluateAmbientOcclusion(const in vec3 objPos, const in vec3 objNormal) 
{
	// Number of samples per ray
	const int samplesPerRay = 5;
	// Number of total samples
	const int samplesCount = NUM_CONEDIRS*samplesPerRay;

	// Calculate a rotation that transforms the direction vectors to lie around the normal
	// The untransformed direction vectors are distributed around the Y axis
	vec3 new_y = normalize(objNormal);
	vec3 new_x = cross(new_y, normalize(objPos));
	vec3 new_z = cross(new_x, new_y);
	mat3 dirRot = (mat3(new_x, new_y, new_z));

	float ambient = 0.0;
	
	// Convert the object position to a normalized position
	vec3 normalizedPos = (objPos-inBoundsMin)/inBoundsSize;

	// Calculate the ambient factors
	
	for (int i=0; i<NUM_CONEDIRS; ++i) {
		vec3 dir = (dirRot * coneDirs[i].xyz);

		float ambientCone = 0.0;
		float rayStep = 0.0;
		float t = inAOOffset + 0.0001;
		float voxelSize = 0.0;
		float volume = 0.0;
		do {

			float lod = clamp(log2(t*coneDirs[i].w*inAmbVolShortestEdge), 0.0, inAmbVolMaxLod);
			
			vec3 pos = normalizedPos + t*dir;
			
			float vis = textureLod(inDensityTex, pos, lod).r;
			
			ambientCone += (1.0 - ambientCone) * vis;
			if (ambientCone > 0.99) 
				break;

			t += t * coneDirs[i].w;
		} while (t<inAOConeLength); 
		
		ambient += ambientCone;
	} 
	

	ambient = 1.0 - clamp(inAOStrength * ambient / float(NUM_CONEDIRS), 0.0, 1.0);

	return ambient;
}