#version 460 compatibility

// work_group_size_x, work_group_size_y, work_group_size_z
layout (local_size_x = 32, local_size_y = 32) in;

layout(r32f, binding = 0) uniform image2D u_prev_tex;
layout(r32f, binding = 1) uniform image2D v_prev_tex;
layout(r32f, binding = 2) uniform image2D dens_prev_tex;
layout(r32f, binding = 3) uniform image2D u_tex;
layout(r32f, binding = 4) uniform image2D v_tex;
layout(r32f, binding = 5) uniform image2D dens_tex;

uniform int xsize;
uniform int ysize;
uniform float dt;
uniform bool initFlag;

// add source uniforms
//-----------------------

//-----------------------

// global defs
//---------------------------------
ivec2 position = ivec2(gl_GlobalInvocationID.xy);

float M_PI = 3.1415926;
vec3 red = vec3(1.f, 0.f, 0.f);
vec3 green = vec3(0.f, 1.f, 0.f);
vec3 blue = vec3(0.f, 0.f, 1.f);
//---------------------------------



void init() {
	bool boundary = (position.x == 0) || (position.x == xsize - 1) || (position.y == 0) || (position.y == ysize - 1);
	bool bound = !boundary;

	float x = (position.x - 1) / (xsize - 2.f);
	float y = (position.y - 1) / (ysize - 2.f);
	float sinus_y = ((sin(x * 2.f * 3.1415965359) + 1.f) / 2.f) * (ysize - 2) / 2.f + ((ysize - 2) / 4.f);
	float spline_y = ((sin(x * 2.f * 3.1415965359) + 1.f) / 2.f) * (4) / 2.f + ((ysize - 2) / 4.f);
	
	
	float value = 1.f;
	if(!boundary) {
		if(position.y <= spline_y) {
			float vel = -0.25f;
			float vert = 0.f;//0.2f * sin(2.0 * M_PI * x);
			float dens = 0.f;
			imageStore(u_prev_tex, position, vec4(vel, 0.f, 0.f, 1.f));
			imageStore(v_prev_tex, position, vec4(vert, 0.f, 0.f, 1.f));
			//imageStore(dens_prev_tex, position, vec4(dens, 0.f, 0.f, 1.f));
		}
		else {
			float vel = 0.25f;
			float vert = 0.f;//0.2f * sin(2.0 * M_PI * x);
			float dens = 0.f;
			imageStore(u_prev_tex, position, vec4(vel, 0.f, 0.f, 1.f));
			imageStore(v_prev_tex, position, vec4(vert, 0.f, 0.f, 1.f));
			//imageStore(dens_prev_tex, position, vec4(dens, 0.f, 0.f, 1.f));
		}
	}
	else {
		imageStore(u_prev_tex, position, vec4(0.f));
		imageStore(v_prev_tex, position, vec4(0.f));
		imageStore(dens_prev_tex, position, vec4(0.f));
	}

	if(!boundary) {
		imageStore(dens_prev_tex, ivec2(position.x, spline_y), vec4(20.f, 0.f, 0.f, 1.f));
	}
	else {
		imageStore(dens_prev_tex, position, vec4(0.f, 0.f, 0.f, 1.f));
	}

	 barrier();

	imageStore(u_tex, position, imageLoad(u_tex, position) + imageLoad(u_prev_tex, position));
	imageStore(v_tex, position, imageLoad(v_tex, position) + imageLoad(v_prev_tex, position));
	imageStore(dens_tex, position, imageLoad(dens_tex, position) + imageLoad(dens_prev_tex, position));
}

void clear_tex() {
	//imageStore(u_prev_tex, position, vec4(0.f));
	//imageStore(v_prev_tex, position, vec4(0.f));
	//imageStore(dens_prev_tex, position, vec4(0.f));
}

void main() {
	if(initFlag) {
		init();
	}
	else {
		clear_tex();
	}

	barrier();

	//add_source(u_tex, u_prev_tex);
	//imageStore(u_tex, position, imageLoad(u_tex, position) + imageLoad(u_prev_tex, position));
	//add_source(v_tex, v_prev_tex);
	//imageStore(v_tex, position, imageLoad(v_tex, position) + imageLoad(v_prev_tex, position));
	//add_source(dens_tex, dens_prev_tex);
	//imageStore(dens_tex, position, imageLoad(dens_tex, position) + imageLoad(dens_prev_tex, position));
}