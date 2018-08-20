#version 460 compatibility

// work_group_size_x, work_group_size_y, work_group_size_z
layout (local_size_x = 32, local_size_y = 32) in;

layout(r32f, binding = 0) uniform image2D u_tex;
layout(r32f, binding = 1) uniform image2D v_tex;
layout(r32f, binding = 2) uniform image2D vel_tex;

uniform int xsize;
uniform int ysize;

ivec2 position = ivec2(gl_GlobalInvocationID.xy);

void main() {
	//bool boundary = (position.x == 0) || (position.x == xsize_p2 - 1) || (position.y == 0) || (position.y == ysize_p2 - 1);

	vec4 value_u = imageLoad(u_tex, position);
	vec4 value_v = imageLoad(v_tex, position);
	barrier();
	float value = sqrt(value_u.x * value_u.x + value_v.x * value_v.x);
	imageStore(vel_tex, position, vec4(value));
}