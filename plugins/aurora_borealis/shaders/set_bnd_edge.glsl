#version 460 compatibility

// work_group_size_x, work_group_size_y, work_group_size_z
layout (local_size_x = 32, local_size_y = 32) in;

layout(r32f, binding = 0) uniform image2D x_tex_e;

// ggf uniform mitgeben, ob periodisch oder fixed

uniform int xsize_e;
uniform int ysize_e;
uniform int b;

vec4 red = vec4(1.f, 0.f, 0.f, 0.f);
vec4 green = vec4(0.f, 1.f, 0.f, 0.f);
vec4 blue = vec4(0.f, 0.f, 1.f, 0.f);

ivec2 position = ivec2(gl_GlobalInvocationID.xy);

void main() {
	bool left = (position.x == 0) && !((position.y == 0) || (position.y == ysize_e - 1));
	bool right = (position.x == (xsize_e - 1)) && !((position.y == 0) || (position.y == ysize_e - 1));
	bool bottom = (position.y == 0) && !((position.x == 0) || (position.x == xsize_e - 1));
	bool top = (position.y == (ysize_e - 1)) && !((position.x == 0) || (position.x == xsize_e - 1));

	if(left) {
		vec4 value = imageLoad(x_tex_e, ivec2(xsize_e - 2, position.y));
		imageStore(x_tex_e, position, value);
	}
	else if (right) {
		vec4 value = imageLoad(x_tex_e, ivec2(1, position.y));
		imageStore(x_tex_e, position, value);
	}
	else if (bottom) {
		//vec4 value = b == 2 ? -imageLoad(x_tex_e, position + ivec2(0, 1)) : imageLoad(x_tex_e, position + ivec2(0, 1));
		vec4 value = imageLoad(x_tex_e, ivec2(position.x, ysize_e - 2));
		imageStore(x_tex_e, position, value);
	}
	else if (top) {
		//vec4 value = b == 2 ? -imageLoad(x_tex_e, position + ivec2(0, -1)) : imageLoad(x_tex_e, position + ivec2(0, -1));
		vec4 value = imageLoad(x_tex_e,  ivec2(position.x, 1));
		imageStore(x_tex_e, position, value);
	} 
	//else {
	//	vec4 value = imageLoad(x_tex_e, position);
	//	imageStore(x_tex_e, position, vec4(0.5f));
	//}
}