
#version 150

uniform float inWidth;
uniform float inHeight;

uniform mat4 MVPinv;

uniform sampler2D inColorTex;
uniform sampler2D inNormalsTex;
uniform sampler2D inDepthTex;

uniform bool inUseHighPrecision;

uniform vec3 inObjLightDir;
uniform vec3 inObjCamPos;

out vec4 outColor;


// Declaration for local lighting
vec3 LocalLighting(const in vec3 ray, const in vec3 normal, const in vec3 light_dir, const in vec3 color);


// Declaration for ambient occlusion
uniform sampler3D inDensityTex;
uniform float inAmbVolShortestEdge;
uniform float inAmbVolMaxLod;
uniform float inAOOffset;
uniform float inAOStrength;
uniform float inAOConeLength;
uniform vec3 inBoundsMin;
uniform vec3 inBoundsSize;

float evaluateAmbientOcclusion(const in vec3 objPos, const in vec3 objNormal);


void main()
{
	ivec2 texelCoord = ivec2(gl_FragCoord.xy);
	
	float depth = texelFetch(inDepthTex, texelCoord, 0).r;

	if (depth == 1.0) {
		discard;
		return;
	}

	// Reconstruct object coordinates
	vec4 objPos = MVPinv * (vec4(gl_FragCoord.xy/vec2(inWidth, inHeight), depth, 1.0) * 2.0 - 1.0);
	objPos /= objPos.w;

	vec3 color = texelFetch(inColorTex, texelCoord, 0).xyz;
	vec4 normal = texelFetch(inNormalsTex, texelCoord, 0);
	
	if (!inUseHighPrecision)
		normal = normal * 2.0 - 1.0;

	vec3 ray = normalize(objPos.xyz - inObjCamPos.xyz);
	vec3 lightCol = LocalLighting(ray, normal.xyz, inObjLightDir, color);

	if (normal.w < 1.0)
		lightCol *= evaluateAmbientOcclusion(objPos.xyz, normal.xyz);
	
	outColor = vec4(lightCol, 1.0);
	
	gl_FragDepth = depth;
}
