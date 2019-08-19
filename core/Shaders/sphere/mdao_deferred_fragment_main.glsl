#version 150

uniform float inWidth;
uniform float inHeight;

uniform mat4 MVPinv;

uniform sampler2D inColorTex;
uniform sampler2D inNormalsTex;
uniform sampler2D inDepthTex;

uniform bool inUseHighPrecision;

out vec4 outColor;


// Declaration for local lighting
vec3 evaluateLocalLighting(const in vec3 objPos, const in vec3 objNormal, const in vec3 matColor);

// Declaration for ambient occlusion
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

	vec3 lightCol = evaluateLocalLighting(objPos.xyz, normal.xyz, color);
	if (normal.w < 1.0)
		lightCol *= evaluateAmbientOcclusion(objPos.xyz, normal.xyz);
	
	outColor = vec4(lightCol, 1.0);
	
	gl_FragDepth = depth;
}