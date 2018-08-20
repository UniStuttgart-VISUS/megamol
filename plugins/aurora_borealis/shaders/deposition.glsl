#version 460 compatibility

// work_group_size_x, work_group_size_y, work_group_size_z
layout (local_size_x = 32, local_size_y = 32) in;

#define M_PI 2.0 * asin(1)


uniform float height;
//uniform float energy[];

layout(r32f, binding = 0) uniform image2D simTex;
//layout(r32f, binding = 1) uniform image2D energy;
layout(r32f, binding = 2) uniform image3D deposition;

void main() {
	// start at height = 100km
	ivec2 position = ivec2(gl_GlobalInvocationID.xy);
	float initValue = imageLoad(simTex, position).x;
	barrier();

	if(initValue > 0.02f) {
		for(int i = 0; i < int(height); ++i) {
			float value = initValue;// * sin((height - float(i)) / height * (M_PI / 2.f));
			
			// initial energy: 20keV --> index 745 to 893
			// initial height 100km = index + 10
			// each pixel is 2km --> + i
			// float value = initValue * energy[745 + 10 + i];
			imageStore(deposition, ivec3(position.x, i, position.y), vec4(value));
		}
	}
}