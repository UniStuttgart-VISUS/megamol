#version 460 compatibility

// work_group_size_x, work_group_size_y, work_group_size_z
layout (local_size_x = 32, local_size_y = 32) in;

layout(r32f, binding = 0) uniform image2D u_tex;
layout(r32f, binding = 1) uniform image2D v_tex;
layout(r32f, binding = 2) uniform image2D p_tex;

uniform int xsize_p2;
uniform int ysize_p2;

ivec2 position = ivec2(gl_GlobalInvocationID.xy);

void main() {
	bool boundary = (position.x == 0) || (position.x == xsize_p2 - 1) || (position.y == 0) || (position.y == ysize_p2 - 1);

	if(!boundary) {
		vec4 value = imageLoad(u_tex, position) - 0.5f * (xsize_p2 - 2) * (imageLoad(p_tex, position + ivec2(1, 0)) - imageLoad(p_tex, position + ivec2(-1, 0)));
		barrier();
		imageStore(u_tex, position, value);
		value = imageLoad(v_tex, position) - 0.5f * (ysize_p2 - 2) * (imageLoad(p_tex, position + ivec2(0, 1)) - imageLoad(p_tex, position + ivec2(0, -1)));
		barrier();
		imageStore(v_tex, position, value);
	} // else do nothing
}