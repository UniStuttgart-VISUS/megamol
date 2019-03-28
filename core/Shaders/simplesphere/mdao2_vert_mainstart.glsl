//#extension GL_ARB_gpu_shader_fp64 : enable
#define CLIP
#define DEPTH

in vec4 position;
in vec4 color;

out vec4 vsColor;

#ifdef WITH_SCALING
uniform float scaling;
#endif // WITH_SCALING

uniform vec4 inViewAttr;
uniform vec3 inCamFront;
uniform vec3 inCamUp;
uniform vec3 inCamRight;
uniform vec4 inCamPos;

uniform mat4 inMvp;
uniform mat4 inMvpInverse;

uniform float inGlobalRadius;
uniform bool inUseGlobalColor;
uniform vec4 inGlobalColor;

uniform bool inUseTransferFunction;
uniform sampler1D inTransferFunction;
uniform vec2 inIndexRange;

uniform vec4 inClipDat;
uniform vec4 inClipCol;

out vec4 vsObjPos;
out vec4 vsCamPos;
out float vsSquaredRad;
out float vsRad;

void main(void) {

    // remove the sphere radius from the w coordinates to the rad varyings
    vsObjPos = position;
    vsRad = (inGlobalRadius == 0.0) ? vsObjPos.w : inGlobalRadius;
    vsObjPos.w = 1.0;

	vsColor = color;
	if (inUseGlobalColor)
		vsColor = inGlobalColor;
	else
	if (inUseTransferFunction) {
		float texOffset = 0.5/float(textureSize(inTransferFunction, 0));
		float normPos = (color.r - inIndexRange.x)/(inIndexRange.y - inIndexRange.x);
		vsColor = texture(inTransferFunction, normPos * (1.0 - 2.0*texOffset) + texOffset);
	}

#ifdef WITH_SCALING
    vsRad *= scaling;
#endif // WITH_SCALING

    vsSquaredRad = vsRad * vsRad;

    // calculate cam position
    vsCamPos.xyz = inCamPos.xyz -  vsObjPos.xyz; // cam pos to glyph space

    // Sphere-Touch-Plane-Approach
    vec2 winHalf = 2.0 / inViewAttr.zw; // window size