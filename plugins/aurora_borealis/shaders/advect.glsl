#version 460 compatibility

// work_group_size_x, work_group_size_y, work_group_size_z
layout (local_size_x = 32, local_size_y = 32) in;

layout(r32f, binding = 0) uniform image2D d_tex;
layout(r32f, binding = 1) uniform image2D d0_tex;
layout(r32f, binding = 2) uniform image2D u_tex;
layout(r32f, binding = 3) uniform image2D v_tex;

uniform int xsize;
uniform int ysize;
uniform float dt;
uniform float diff;

ivec2 position = ivec2(gl_GlobalInvocationID.xy);


void main() {
	bool boundary = (position.x == 0) || (position.x == xsize - 1) || (position.y == 0) || (position.y == ysize - 1);
	float dt0X = dt * (xsize - 2);
	float dt0Y = dt * (ysize - 2);

	if(!boundary) {
		float x = position.x - dt0X * imageLoad(u_tex, position).r;
		float y = position.y - dt0Y * imageLoad(v_tex, position).r;
			
		//int i0 = (xsize + (int(x) % xsize)) % xsize; int i1 = (i0 + 1) % xsize;

		int i0;
		if(x >= 0.f) {
			i0 = int(x);
		} else {
			i0 = -int(1.f - x);
		}
		 float s1 = x - i0; i0 = (xsize + (i0 % xsize)) % xsize;
		 int i1 = (i0 + 1) % xsize;

		int j0;
		if(y >= 0.f) {
			j0 = int(y);
		} else {
			j0 = -int(1.f - y);
		}
		float t1 = y - j0; j0 = (ysize + (j0 % ysize)) % ysize;
		int j1 = (j0 + 1) % ysize;
			
		//if (x < 0.5) {x = 0.5;}
		//else if (x > xsize - 2 + 0.5) {x = xsize - 2 + 0.5;} 
		//int i0 = int(x); int i1 = i0 + 1;

		//if (y < 0.5) {y = 0.5;}
		//else if (y > ysize - 2 + 0.5) {y = ysize - 2 + 0.5;} 
		//int j0 = int(y); int j1 = j0 + 1;

		float s0 = 1.f - s1;
		float t0 = 1.f - t1;
		//float s1 = x - i0; float s0 = 1.f - s1;
		//float t1 = y - j0; float t0 = 1.f - t1;

		vec4 value = s0 * (t0 * imageLoad(d0_tex, ivec2(i0, j0)) + t1 * imageLoad(d0_tex, ivec2(i0, j1))) + s1 * (t0 * imageLoad(d0_tex, ivec2(i1, j0)) + t1 * imageLoad(d0_tex, ivec2(i1, j1)));
		imageStore(d_tex, position, value);
	} // else do nothing
}