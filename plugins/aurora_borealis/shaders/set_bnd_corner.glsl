#version 460 compatibility

// work_group_size_x, work_group_size_y, work_group_size_z
layout (local_size_x = 32, local_size_y = 32) in;

layout(r32f, binding = 0) uniform image2D x_tex_c;

uniform int xsize_c;
uniform int ysize_c;

vec4 red = vec4(1.f);
vec4 green = vec4(0.f, 1.f, 0.f, 0.f);
vec4 blue = vec4(0.f, 0.f, 1.f, 0.f);

ivec2 position = ivec2(gl_GlobalInvocationID.xy);

void main() {
	bool bottom_left = false;//(position.x == 0) && (position.y == 0);
	bool bottom_right = false;//(position.x == xsize_c - 1) && (position.y == 0);
	bool top_left = false;//(position.x == 0) && (position.y == ysize_c - 1);
	bool top_right = false;//(position.x == xsize_c - 1) && (position.y == ysize_c - 1);

	if(bottom_left) {
		vec4 value1 = imageLoad(x_tex_c, position + ivec2(1, 0));
		vec4 value2 = imageLoad(x_tex_c, position + ivec2(0, 1));
		vec4 value = 0.5f * (value1 + value2);
		imageStore(x_tex_c, position, value);
	} 
	else if (bottom_right) {
		vec4 value1 = imageLoad(x_tex_c, position + ivec2(-1, 0));
		vec4 value2 = imageLoad(x_tex_c, position + ivec2(0, 1));
		vec4 value = 0.5f * (value1 + value2);
		imageStore(x_tex_c, position, value);
	}
	else if (top_left) {
		vec4 value1 = imageLoad(x_tex_c, position + ivec2(1, 0));
		vec4 value2 = imageLoad(x_tex_c, position + ivec2(0, -1));
		vec4 value = 0.5f * (value1 + value2);
		imageStore(x_tex_c, position, value);
	}
	else if (top_right) {
		vec4 value1 = imageLoad(x_tex_c, position + ivec2(-1, 0));
		vec4 value2 = imageLoad(x_tex_c, position + ivec2(0, -1));
		vec4 value = 0.5f * (value1 + value2);
		imageStore(x_tex_c, position, value);
	} // else do nothing
}