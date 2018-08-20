#version 460 compatibility

// work_group_size_x, work_group_size_y, work_group_size_z
layout (local_size_x = 32, local_size_y = 32) in;

layout(r32f, binding = 0) uniform image2D u_tex;
layout(r32f, binding = 1) uniform image2D v_tex;
layout(r32f, binding = 2) uniform image2D p_tex;
layout(r32f, binding = 3) uniform image2D div_tex;

uniform int xsize_p1;
uniform int ysize_p1;

ivec2 position = ivec2(gl_GlobalInvocationID.xy);

void main() {
	bool boundary = (position.x == 0) || (position.x == xsize_p1 - 1) || (position.y == 0) || (position.y == ysize_p1 - 1);
	float h = 1.f / float(xsize_p1 - 2);

	if(!boundary) {
		vec4 value = imageLoad(u_tex, position + ivec2(1, 0)) - imageLoad(u_tex, position + ivec2(-1, 0)) + imageLoad(v_tex, position + ivec2(0, 1)) - imageLoad(v_tex, position + ivec2(0, -1));
		imageStore(div_tex, position, -0.5f * h * value);
		imageStore(p_tex, position, vec4(0.f));
	} // else do nothing
}