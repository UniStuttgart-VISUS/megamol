#version 460 compatibility

// work_group_size_x, work_group_size_y, work_group_size_z
layout (local_size_x = 32, local_size_y = 32) in;

layout(r32f, binding = 0) uniform image2D x_tex;
layout(r32f, binding = 1) uniform image2D x0_tex;
layout(r32f, binding = 2) uniform image2D help_diffuse_tex;

uniform float a;
uniform float aq;
uniform int xsize;
uniform int ysize;

ivec2 position = ivec2(gl_GlobalInvocationID.xy);

void main() {
	bool even = ((position.x + position.y * (xsize - 1)) % 2) == 0;	
	bool boundary = ((position.x == 0) || (position.x == xsize - 1) || (position.y == 0) || (position.y == ysize - 1));

	if(!boundary && even) {
		//vec4 value = (imageLoad(x0_tex, position) + a * (imageLoad(help_diffuse_tex, position + ivec2(-1, 0)) + imageLoad(help_diffuse_tex, position + ivec2(1, 0)) 
		//				+ imageLoad(help_diffuse_tex, position + ivec2(0, -1)) + imageLoad(help_diffuse_tex, position + ivec2(0, 1)))) / aq;

		vec4 value = (imageLoad(x0_tex, position) + a * (imageLoad(x_tex, position + ivec2(-1, 0)) + imageLoad(x_tex, position + ivec2(1, 0)) 
						+ imageLoad(x_tex, position + ivec2(0, -1)) + imageLoad(x_tex, position + ivec2(0, 1)))) / aq;
		barrier();
		imageStore(x_tex, position, value);
	} // else do nothing
}