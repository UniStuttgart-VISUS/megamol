#extension GL_ARB_gpu_shader_fp64 : enable   // glsl version 150

in vec4 position;
in vec4 color;

#ifdef WITH_SCALING
uniform float scaling;
#endif // WITH_SCALING

uniform vec4 inViewAttr;
uniform vec3 inCamFront;
uniform vec3 inCamUp;
uniform vec3 inCamRight;

uniform mat4 inMvp;
uniform mat4 MVPinv;

uniform float inGlobalRadius;
uniform bool inUseGlobalColor;
uniform vec4 inGlobalColor;

uniform bool inUseTransferFunction;
uniform sampler1D inTransferFunction;
uniform vec2 inIndexRange;

uniform vec4 inClipDat;
uniform vec4 inClipCol;

out vec4 objPos;
out vec4 camPos;
out float squareRad;
out float rad;
out vec4 vsColor;

void main(void) {

    // remove the sphere radius from the w coordinates to the rad varyings
    objPos = position;
    rad = (inGlobalRadius == 0.0) ? objPos.w : inGlobalRadius;
    objPos.w = 1.0;

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
    rad *= scaling;
#endif // WITH_SCALING

    squareRad = rad * rad;

    // calculate cam position 
    camPos = MVinv[3]; // (C) by Christoph 
    camPos.xyz -= objPos.xyz; // cam pos to glyph space 
