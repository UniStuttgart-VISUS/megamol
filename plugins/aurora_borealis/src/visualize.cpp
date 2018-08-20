#include "visualize.h"
#include "core.hpp"

#include <algorithm>
#include <vector>

// local functions and variable
/////////////////////////////////////////////////////////////////////////////////////
int texIDX, step = 0, cnt = 0, check_flag = 0, iteration = 0;




Visualize::Visualize() 
{
	FoV = 55.f;
	zNear = 1.f;
	zFar = 500.f;
	_window_width = 0;
	_window_height = 0;
}

Visualize::~Visualize() 
{
	// delete textures
	glDeleteTextures(1, &dens_prev_tex);
	glDeleteTextures(1, &dens_tex);
	glDeleteTextures(1, &u_prev_tex);
	glDeleteTextures(1, &v_prev_tex);
	glDeleteTextures(1, &u_tex);
	glDeleteTextures(1, &v_tex);
	glDeleteTextures(1, &swap_tex);
	glDeleteTextures(1, &_3dtex);
	glDeleteTextures(1, &_velocity_tex);
	glDeleteTextures(1, &_sim_tex);
	glDeleteTextures(1, &_energyTex);

	// delete programs
	glDeleteProgram(_fvProgram);
	glDeleteProgram(_init_source_Program);
	glDeleteProgram(_diffuse_odd_Program);
	glDeleteProgram(_diffuse_even_Program);
	glDeleteProgram(_advect_Program);
	glDeleteProgram(_project_1_Program);
	glDeleteProgram(_project_2_Program);
	glDeleteProgram(_set_bnd_e_Program);
	glDeleteProgram(_set_bnd_c_Program);
	glDeleteProgram(_velocity_Program);
	glDeleteProgram(_velProgram);
	glDeleteProgram(_deposition_Program);

	// delete buffers
	glDeleteBuffers(1, &_quad_vertexBuffer);
	glDeleteBuffers(1, &_quad_texcoordBuffer);
}


bool Visualize::initialize() 
{
	init();

	shaderSetup();
	bufferSetup();
	textureSetup();
	preCalculateDeposition();
	uniformLocations();
	setConstantUniforms();

	dt = 0.01f;//0.075f;
	visc = 0.f;//0.0001f;//0.00014f;// (dt * _cs_tex_width * _cs_tex_height);// 0.00001f;
	diff = 0.f;//0.0001f;// 0.000021f;

	return true;
}


void Visualize::init() 
{
	// blend the aurora to get transparency
	//glEnable(GL_BLEND);
	glBlendColor(10.f, 0.f, 0.f, 0.7f);
	glBlendFunc(GL_SRC_ALPHA, GL_CONSTANT_ALPHA);

	glEnable(GL_ALPHA_TEST);
	glAlphaFunc(GL_GREATER, 0.05f);
}

void Visualize::shaderSetup() 
{
	_vShader = glCreateShader(GL_VERTEX_SHADER);
	CompileShader(_vShader, "../share/shaders/aurora/vert.glsl");
	_fShader = glCreateShader(GL_FRAGMENT_SHADER);
	CompileShader(_fShader, "../share/shaders/aurora/frag.glsl");
	_velShader = glCreateShader(GL_FRAGMENT_SHADER);
	CompileShader(_velShader, "../share/shaders/aurora/frag_vel.glsl");
	_init_source_Shader = glCreateShader(GL_COMPUTE_SHADER);
	CompileShader(_init_source_Shader, "../share/shaders/aurora/init_source.glsl");
	_diffuse_odd_Shader = glCreateShader(GL_COMPUTE_SHADER);
	CompileShader(_diffuse_odd_Shader, "../share/shaders/aurora/diffuse_odd.glsl");
	_diffuse_even_Shader = glCreateShader(GL_COMPUTE_SHADER);
	CompileShader(_diffuse_even_Shader, "../share/shaders/aurora/diffuse_even.glsl");
	_advect_Shader = glCreateShader(GL_COMPUTE_SHADER);
	CompileShader(_advect_Shader, "../share/shaders/aurora/advect.glsl");
	_project_1_Shader = glCreateShader(GL_COMPUTE_SHADER);
	CompileShader(_project_1_Shader, "../share/shaders/aurora/project_1.glsl");
	_project_2_Shader = glCreateShader(GL_COMPUTE_SHADER);
	CompileShader(_project_2_Shader, "../share/shaders/aurora/project_2.glsl");
	_set_bnd_edge_Shader = glCreateShader(GL_COMPUTE_SHADER);
	CompileShader(_set_bnd_edge_Shader, "../share/shaders/aurora/set_bnd_edge.glsl");
	_set_bnd_corner_Shader = glCreateShader(GL_COMPUTE_SHADER);
	CompileShader(_set_bnd_corner_Shader, "../share/shaders/aurora/set_bnd_corner.glsl");
	_velocity_Shader = glCreateShader(GL_COMPUTE_SHADER);
	CompileShader(_velocity_Shader, "../share/shaders/aurora/velocity.glsl");
	_deposition_Shader = glCreateShader(GL_COMPUTE_SHADER);
	CompileShader(_deposition_Shader, "../share/shaders/aurora/deposition.glsl");

	_fvProgram = glCreateProgram();
	LinkProgram(_fvProgram, { _vShader, _fShader });
	_velProgram = glCreateProgram();
	LinkProgram(_velProgram, { _vShader, _velShader });
	_init_source_Program = glCreateProgram();
	LinkProgram(_init_source_Program, { _init_source_Shader });
	_diffuse_odd_Program = glCreateProgram();
	LinkProgram(_diffuse_odd_Program, { _diffuse_odd_Shader });
	_diffuse_even_Program = glCreateProgram();
	LinkProgram(_diffuse_even_Program, { _diffuse_even_Shader });
	_advect_Program = glCreateProgram();
	LinkProgram(_advect_Program, { _advect_Shader });
	_project_1_Program = glCreateProgram();
	LinkProgram(_project_1_Program, { _project_1_Shader });
	_project_2_Program = glCreateProgram();
	LinkProgram(_project_2_Program, { _project_2_Shader });
	_set_bnd_e_Program = glCreateProgram();
	LinkProgram(_set_bnd_e_Program, { _set_bnd_edge_Shader });
	_set_bnd_c_Program = glCreateProgram();
	LinkProgram(_set_bnd_c_Program, { _set_bnd_corner_Shader });
	_velocity_Program = glCreateProgram();
	LinkProgram(_velocity_Program, { _velocity_Shader });
	_deposition_Program = glCreateProgram();
	LinkProgram(_deposition_Program, { _deposition_Shader });

	//clean-up
	// detach shaders
	glDetachShader(_fvProgram, _vShader);
	glDetachShader(_fvProgram, _fShader);
	glDetachShader(_init_source_Program, _init_source_Shader);
	glDetachShader(_diffuse_odd_Program, _diffuse_odd_Shader);
	glDetachShader(_diffuse_even_Program, _diffuse_even_Shader);
	glDetachShader(_advect_Program, _advect_Shader);
	glDetachShader(_project_1_Program, _project_1_Shader);
	glDetachShader(_project_2_Program, _project_2_Shader);
	glDetachShader(_set_bnd_e_Program, _set_bnd_edge_Shader);
	glDetachShader(_set_bnd_c_Program, _set_bnd_corner_Shader);
	glDetachShader(_velocity_Program, _velocity_Shader);
	glDetachShader(_velProgram, _velShader);
	glDetachShader(_velProgram, _vShader);
	glDetachShader(_deposition_Program, _deposition_Shader);

	// delete shader
	glDeleteShader(_deposition_Shader);
	glDeleteShader(_velShader);
	glDeleteShader(_set_bnd_edge_Shader);
	glDeleteShader(_set_bnd_corner_Shader);
	glDeleteShader(_diffuse_odd_Shader);
	glDeleteShader(_diffuse_even_Shader);
	glDeleteShader(_advect_Shader);
	glDeleteShader(_project_1_Shader);
	glDeleteShader(_project_2_Shader);
	glDeleteShader(_init_source_Shader);
	glDeleteShader(_velocity_Shader);
	glDeleteShader(_vShader);
	glDeleteShader(_fShader);
}

void Visualize::bufferSetup() {
	static const GLfloat g_quad_vertex_buffer_data[] = {
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		-1.0f,  1.0f, 0.0f,

		-1.0f,  1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		1.0f,  1.0f, 0.0f,
	};

	glGenBuffers(1, &_quad_vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, _quad_vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	static const GLfloat g_quad_texcoord_buffer_data[] = {
		0.0f, 0.0f,
		1.0f, 0.0f,
		0.0f,  1.0f,

		0.0f,  1.0f,
		1.0f, 0.0f,
		1.0f,  1.0f,
	};

	glGenBuffers(1, &_quad_texcoordBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, _quad_texcoordBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_texcoord_buffer_data), g_quad_texcoord_buffer_data, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Visualize::textureSetup() {
	texIDX = 0; u_prev_tex = compTexture();
	texIDX = 1; v_prev_tex = compTexture();
	texIDX = 2; dens_prev_tex = compTexture();
	texIDX = 3; u_tex = compTexture();
	texIDX = 4; v_tex = compTexture();
	texIDX = 5; dens_tex = compTexture();
	texIDX = 6; _velocity_tex = compTexture();
	texIDX = 7; swap_tex = compTexture(); texIDX = 9;

	// convert into 3D texture
	glActiveTexture(GL_TEXTURE0 + 9);
	glGenTextures(1, &_3dtex);
	glBindTexture(GL_TEXTURE_3D, _3dtex);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, _cs_tex_width, _cs_tex3d_height, _cs_tex_depth, 0, GL_RED, GL_FLOAT, NULL);
	glGenerateMipmap(GL_TEXTURE_3D);
}

void Visualize::uniformLocations() { // texIDs aufräumen
	_texID = glGetUniformLocation(_fvProgram, "sim_texture");
	_alphaTexID = glGetUniformLocation(_fvProgram, "alphaTex");
	_mvpID = glGetUniformLocation(_fvProgram, "MVP");
	_stretchIDx = glGetUniformLocation(_fvProgram, "stretchx");
	_stretchIDy = glGetUniformLocation(_fvProgram, "stretchy");
	_depthID = glGetUniformLocation(_fvProgram, "depth");
	_depthResID = glGetUniformLocation(_fvProgram, "depthRes");

	_mvpIDVel = glGetUniformLocation(_velProgram, "MVP");
	_stretchIDxVel = glGetUniformLocation(_velProgram, "stretchx");
	_stretchIDyVel = glGetUniformLocation(_velProgram, "stretchy");

	_xsizeID_set_bnd_e = glGetUniformLocation(_set_bnd_e_Program, "xsize_e");
	_ysizeID_set_bnd_e = glGetUniformLocation(_set_bnd_e_Program, "ysize_e");
	_bID = glGetUniformLocation(_set_bnd_e_Program, "b");
	_xsizeID_set_bnd_c = glGetUniformLocation(_set_bnd_c_Program, "xsize_c");
	_ysizeID_set_bnd_c = glGetUniformLocation(_set_bnd_c_Program, "ysize_c");
	_bnd_e_texID = glGetUniformLocation(_set_bnd_e_Program, "x_tex_e");
	_bnd_c_texID = glGetUniformLocation(_set_bnd_c_Program, "x_tex_c");
	
	_u_prev_texID = glGetUniformLocation(_init_source_Program, "u_prev_tex");
	_v_prev_texID = glGetUniformLocation(_init_source_Program, "v_prev_tex");
	_dens_prev_texID = glGetUniformLocation(_init_source_Program, "dens_prev_tex");
	_u_texID = glGetUniformLocation(_init_source_Program, "u_tex");
	_v_texID = glGetUniformLocation(_init_source_Program, "v_tex");
	_dens_texID = glGetUniformLocation(_init_source_Program, "dens_tex");
	_dtID = glGetUniformLocation(_init_source_Program, "dt");
	_initFlagID = glGetUniformLocation(_init_source_Program, "initFlag");
	_xsizeID_init_source = glGetUniformLocation(_init_source_Program, "xsize");
	_ysizeID_init_source = glGetUniformLocation(_init_source_Program, "ysize");

	_aID_odd = glGetUniformLocation(_diffuse_odd_Program, "a");
	_aqID_odd = glGetUniformLocation(_diffuse_odd_Program, "aq");
	_xsizeID_diffuse_odd = glGetUniformLocation(_diffuse_odd_Program, "xsize");
	_ysizeID_diffuse_odd = glGetUniformLocation(_diffuse_odd_Program, "ysize");
	_x_texID_odd = glGetUniformLocation(_diffuse_odd_Program, "x_tex");
	_x0_texID_odd = glGetUniformLocation(_diffuse_odd_Program, "x0_tex");
	_help_diffuse_texID_odd = glGetUniformLocation(_diffuse_odd_Program, "help_diffuse_tex");

	_aID_even = glGetUniformLocation(_diffuse_even_Program, "a");
	_aqID_even = glGetUniformLocation(_diffuse_even_Program, "aq");
	_xsizeID_diffuse_even = glGetUniformLocation(_diffuse_even_Program, "xsize");
	_ysizeID_diffuse_even = glGetUniformLocation(_diffuse_even_Program, "ysize");
	_x_texID_even = glGetUniformLocation(_diffuse_even_Program, "x_tex");
	_x0_texID_even = glGetUniformLocation(_diffuse_even_Program, "x0_tex");
	_help_diffuse_texID_even = glGetUniformLocation(_diffuse_even_Program, "help_diffuse_tex");

	_diffID_advect = glGetUniformLocation(_advect_Program, "diff");
	_dtID_advect = glGetUniformLocation(_advect_Program, "dt");
	_xsizeID_advect = glGetUniformLocation(_advect_Program, "xsize");
	_ysizeID_advect = glGetUniformLocation(_advect_Program, "ysize");
	_d_texID = glGetUniformLocation(_advect_Program, "d_tex");
	_d0_texID = glGetUniformLocation(_advect_Program, "d0_tex");
	_u_advect_texID = glGetUniformLocation(_advect_Program, "u_tex");
	_v_advect_texID = glGetUniformLocation(_advect_Program, "v_tex");

	_xsizeID_project1 = glGetUniformLocation(_project_1_Program, "xsize_p1");
	_ysizeID_project1 = glGetUniformLocation(_project_1_Program, "ysize_p1");
	_xsizeID_project2 = glGetUniformLocation(_project_2_Program, "xsize_p2");
	_ysizeID_project2 = glGetUniformLocation(_project_2_Program, "ysize_p2");

	_vel_uID = glGetUniformLocation(_velocity_Program, "u_tex");
	_vel_vID = glGetUniformLocation(_velocity_Program, "v_tex");
	_velocity_texID = glGetUniformLocation(_velocity_Program, "vel_tex");
	
	_heightID = glGetUniformLocation(_deposition_Program, "height");
}

void Visualize::setConstantUniforms() {
	glUseProgram(0);

	glUseProgram(_set_bnd_e_Program);
	glUniform1i(_xsizeID_set_bnd_e, _cs_tex_width);
	glUniform1i(_ysizeID_set_bnd_e, _cs_tex_height);

	glUseProgram(_set_bnd_c_Program);
	glUniform1i(_xsizeID_set_bnd_c, _cs_tex_width);
	glUniform1i(_ysizeID_set_bnd_c, _cs_tex_height);

	glUseProgram(_init_source_Program);
	glUniform1i(_xsizeID_init_source, _cs_tex_width);
	glUniform1i(_ysizeID_init_source, _cs_tex_height);

	glUseProgram(_diffuse_odd_Program);
	glUniform1i(_xsizeID_diffuse_odd, _cs_tex_width);
	glUniform1i(_ysizeID_diffuse_odd, _cs_tex_height);

	glUseProgram(_diffuse_even_Program);
	glUniform1i(_xsizeID_diffuse_even, _cs_tex_width);
	glUniform1i(_ysizeID_diffuse_even, _cs_tex_height);

	glUseProgram(_advect_Program);
	glUniform1i(_xsizeID_advect, _cs_tex_width);
	glUniform1i(_ysizeID_advect, _cs_tex_height);

	glUseProgram(_project_1_Program);
	glUniform1i(_xsizeID_project1, _cs_tex_width);
	glUniform1i(_ysizeID_project1, _cs_tex_height);

	glUseProgram(_project_2_Program);
	glUniform1i(_xsizeID_project2, _cs_tex_width);
	glUniform1i(_ysizeID_project2, _cs_tex_height);

	glUseProgram(_deposition_Program);
	glUniform1f(_heightID, static_cast<float>(_cs_tex3d_height));

	glUseProgram(_fvProgram);
	glUniform1f(_stretchIDx, _stretchx);
	glUniform1f(_stretchIDy, _stretchy);
	glUniform1f(_depthResID, static_cast<float>(_cs_tex_depth));
	
	glUseProgram(0);
}

GLuint Visualize::compTexture() {
	GLuint tex = 0;

	glActiveTexture(GL_TEXTURE0 + texIDX);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	// linear allows us to scale the window up retaining reasonable quality
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	// same internal format as compute shader input
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, _cs_tex_width, _cs_tex_height, 0, GL_RED, GL_FLOAT, NULL);
	//glGenerateMipmap(GL_TEXTURE_2D);

	return tex;
}

void Visualize::preCalculateDeposition() {
	// total mass density: 
	// values from: https://ccmc.gsfc.nasa.gov/modelweb/models/msis_vitmo.php
	// settings: 
	// Date: 02.08.2016, 13:00h
	// Latitude: 60., Longitude: 15.
	// Height: 100km
	// Start: 80., Stop: 378., Stepsize: 2.

	// height: z = 80 + 2 * i (max height: 378km)
	// index: i = (z - 80) / 2

	double density[_Dcnt] = { 2.345E-08, 1.640E-08, 1.115E-08, 7.367E-09, 4.749E-09, 3.001E-09, 1.872E-09, 1.170E-09, 7.411E-10, 4.810E-10,
		3.230E-10, 2.260E-10, 1.636E-10, 1.213E-10, 9.128E-11, 6.910E-11, 5.231E-11, 3.968E-11, 3.027E-11, 2.329E-11,
		1.814E-11, 1.436E-11, 1.160E-11, 9.529E-12, 7.938E-12, 6.693E-12, 5.702E-12, 4.901E-12, 4.245E-12, 3.703E-12,
		3.250E-12, 2.867E-12, 2.542E-12, 2.264E-12, 2.024E-12, 1.816E-12, 1.635E-12, 1.477E-12, 1.337E-12, 1.214E-12,
		1.104E-12, 1.007E-12, 9.198E-13, 8.419E-13, 7.720E-13, 7.090E-13, 6.522E-13, 6.007E-13, 5.541E-13, 5.117E-13,
		4.731E-13, 4.380E-13, 4.058E-13, 3.764E-13, 3.495E-13, 3.248E-13, 3.020E-13, 2.811E-13, 2.619E-13, 2.442E-13,
		2.278E-13, 2.127E-13, 1.987E-13, 1.857E-13, 1.737E-13, 1.625E-13, 1.522E-13, 1.426E-13, 1.337E-13, 1.254E-13,
		1.177E-13, 1.105E-13, 1.037E-13, 9.749E-14, 9.166E-14, 8.621E-14, 8.111E-14, 7.635E-14, 7.190E-14, 6.774E-14,
		6.384E-14, 6.019E-14, 5.677E-14, 5.357E-14, 5.056E-14, 4.774E-14, 4.509E-14, 4.261E-14, 4.027E-14, 3.808E-14,
		3.602E-14, 3.408E-14, 3.225E-14, 3.054E-14, 2.892E-14, 2.740E-14, 2.596E-14, 2.461E-14, 2.334E-14, 2.214E-14,
		2.100E-14, 1.993E-14, 1.892E-14, 1.797E-14, 1.707E-14, 1.622E-14, 1.541E-14, 1.465E-14, 1.393E-14, 1.325E-14,
		1.262E-14, 1.201E-14, 1.144E-14, 1.089E-14, 1.037E-14, 9.877E-15, 9.411E-15, 8.969E-15, 8.550E-15, 8.152E-15,
		7.775E-15, 7.417E-15, 7.076E-15, 6.753E-15, 6.446E-15, 6.155E-15, 5.877E-15, 5.614E-15, 5.363E-15, 5.125E-15,
		4.898E-15, 4.682E-15, 4.476E-15, 4.281E-15, 4.094E-15, 3.917E-15, 3.748E-15, 3.587E-15, 3.433E-15, 3.286E-15,
		3.147E-15, 3.013E-15, 2.886E-15, 2.765E-15, 2.649E-15, 2.538E-15, 2.433E-15, 2.332E-15, 2.236E-15, 2.144E-15, };

	// atmosphere's shielding mass above height z
	double h = 2.;
	double shieldingMass[_Dcnt - 1]{};
	for (int i = 0; i < _Dcnt - 1; ++i) {
		for (int j = i; j < _Dcnt - 1; ++j) {
			// (left point rectangle + right point rectangle) / 2
			shieldingMass[i] += (density[j] * h + density[j + 1] * h) / 2.;
		}
	}

	//std::cout << "Shielding Mass: " << std::endl;
	double maxMz = *std::max_element(std::begin(shieldingMass), std::end(shieldingMass));
	for (int i = 0; i < _Dcnt - 1; ++i) {
		shieldingMass[i] /= maxMz;
		//std::cout << i * 2 + 80 << ", " << shieldingMass[i] << std::endl;
	}

	bool peakIsReached = false;
	int shiftedIndex = 0;

	//double z = 80. + i * 2.;
	double E[8] = { 1., 2., 5., 10., 15., 20., 25., 30. };
	for (int e = 0; e < 8; ++e) {
		for (int i = 0; i < _Dcnt - 1; ++i) {
			double Me = 4.6 * pow(10, -6) * pow(E[e], 1.65);
			double r = shieldingMass[i] / Me;
			double L = 4.2 * r * exp(-pow(r, 2) - r) + 0.48 * exp(-17.4 * pow(r, 1.37));
			_Az[i + e * 256] = L * E[e] * (density[i] / Me);
		}

		peakIsReached = false;
		shiftedIndex = e * 256;

		//std::cout << "Auroral Deposition Energy: " << std::endl;
		double max = *std::max_element(&_Az[e * 256], &_Az[(e + 1) * 256 - 1]);
		for (int i = e * 256; i < (e + 1) * 256; ++i) {
			// calculate normalized energy deposition rates
			// and truncate every value that is left of the peak
			float normalizedAz = _Az[i] / max;
			
			if (normalizedAz < 1.f && !peakIsReached) {
				continue;
			}
			else {
				peakIsReached = true;
				_Az[shiftedIndex] = normalizedAz / (3.f + 9.f * (normalizedAz != 1.f));
				++shiftedIndex;
			}

			//if (i == e * 256) std::cout << e * 256 << ", " << _Az[e * 256] << std::endl;
		}
	}


	glActiveTexture(GL_TEXTURE0 + 10);
	glGenTextures(1, &_energyTex);
	glBindTexture(GL_TEXTURE_2D, _energyTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 256, 8, 0, GL_RED, GL_FLOAT, _Az);
	glGenerateMipmap(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, 0);
}


void Visualize::swap(const GLuint &first, const GLuint &second) {
	GLfloat *swap = 0;
	swap = new GLfloat[_cs_tex_width * _cs_tex_height];

	// _swap_tex = first;
	glBindTexture(GL_TEXTURE_2D, first);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, swap);
	glBindTexture(GL_TEXTURE_2D, swap_tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _cs_tex_width, _cs_tex_height, GL_RED, GL_FLOAT, swap);
	// first = second;
	glBindTexture(GL_TEXTURE_2D, second);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, swap);
	glBindTexture(GL_TEXTURE_2D, first);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _cs_tex_width, _cs_tex_height, GL_RED, GL_FLOAT, swap);
	// second = _swap_tex;
	glBindTexture(GL_TEXTURE_2D, swap_tex);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, swap);
	glBindTexture(GL_TEXTURE_2D, second);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _cs_tex_width, _cs_tex_height, GL_RED, GL_FLOAT, swap);

	delete[] swap;
	swap = NULL;
}

void Visualize::set_bnd(const GLuint &x, const int b) {
	glUseProgram(_set_bnd_e_Program);

	glUniform1i(_bID, b);
	//glUniform1i(_bnd_e_texID, 0);

	glBindTexture(GL_TEXTURE_2D, x);
	glBindImageTexture(0, x, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);

	// split #workitems into workgroups for each dimension
	// 8 items per group in x-direction and 16 items per group in y-direction --> 8x16 items pro group
	glDispatchCompute(_cs_tex_width / 32, _cs_tex_height / 32, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	glUseProgram(_set_bnd_c_Program);

	//glUniform1i(_bnd_c_texID, 0);

	glBindTexture(GL_TEXTURE_2D, x);
	glBindImageTexture(0, x, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);

	// split #workitems into workgroups for each dimension
	// 8 items per group in x-direction and 16 items per group in y-direction --> 8x16 items pro group
	glDispatchCompute(_cs_tex_width / 32, _cs_tex_height / 32, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void Visualize::init_source() {
	glUseProgram(_init_source_Program);

	// send texture to cShader
	/*glUniform1i(_u_prev_texID, 0);
	glUniform1i(_v_prev_texID, 1);
	glUniform1i(_dens_prev_texID, 2);
	glUniform1i(_u_texID, 3);
	glUniform1i(_v_texID, 4);
	glUniform1i(_dens_texID, 5);*/
	glUniform1f(_dtID, dt);
	glUniform1ui(_initFlagID, _initFlag); _initFlag = false;

	// bind to image unit so can write to specific pixels from the shader
	glBindTexture(GL_TEXTURE_2D, u_prev_tex);
	glBindImageTexture(0, u_prev_tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
	glBindTexture(GL_TEXTURE_2D, v_prev_tex);
	glBindImageTexture(1, v_prev_tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
	glBindTexture(GL_TEXTURE_2D, dens_prev_tex);
	glBindImageTexture(2, dens_prev_tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
	glBindTexture(GL_TEXTURE_2D, u_tex);
	glBindImageTexture(3, u_tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
	glBindTexture(GL_TEXTURE_2D, v_tex);
	glBindImageTexture(4, v_tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
	glBindTexture(GL_TEXTURE_2D, dens_tex);
	glBindImageTexture(5, dens_tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);

	// split #workitems into workgroups for each dimension
	// 8 items per group in x-direction and 16 items per group in y-direction --> 8x16 items pro group
	glDispatchCompute(_cs_tex_width / 32, _cs_tex_height / 32, 1); // 512^2 threads in blocks of 16^2
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void Visualize::diffuse(const GLuint &x, const GLuint &x0, const float a, const float aq, const int b) {
	GLfloat *swap = 0;
	swap = new GLfloat[_cs_tex_width * _cs_tex_height];

	for (int i = 0; i < 20; ++i) {
		// odd diffusion
		glUseProgram(_diffuse_odd_Program);

		/*glUniform1i(_x_texID, 5);
		glUniform1i(_x0_texID, 2);
		glUniform1i(_help_diffuse_texID, 5);*/
		glUniform1f(_aID_odd, a);
		glUniform1f(_aqID_odd, aq);
		
		// copy x to swap_tex in order to read/write x simultaniously
		glBindTexture(GL_TEXTURE_2D, x);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, swap);
		glBindTexture(GL_TEXTURE_2D, swap_tex);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _cs_tex_width, _cs_tex_height, GL_RED, GL_FLOAT, swap);

		glBindTexture(GL_TEXTURE_2D, x);
		glBindImageTexture(0, x, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
		glBindTexture(GL_TEXTURE_2D, x0);
		glBindImageTexture(1, x0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
		glBindTexture(GL_TEXTURE_2D, swap_tex);
		glBindImageTexture(2, swap_tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);

		// number of workgroups: e. g. 512 / 16 --> 32 workgroups
		glDispatchCompute(_cs_tex_width / 32, _cs_tex_height / 32, 1); // 512^2 threads in blocks of 16^2
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


		// even diffusion
		glUseProgram(_diffuse_even_Program);

		/*glUniform1i(_x_texID, 5);
		glUniform1i(_x0_texID, 2);
		glUniform1i(_help_diffuse_texID, 5);*/
		glUniform1f(_aID_even, a);
		glUniform1f(_aqID_even, aq);
		glUniform1i(_xsizeID_diffuse_even, _cs_tex_width);
		glUniform1i(_ysizeID_diffuse_even, _cs_tex_height);

		// copy x to swap_tex in order to read/write x simultaniously
		glBindTexture(GL_TEXTURE_2D, x);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, swap);
		glBindTexture(GL_TEXTURE_2D, swap_tex);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _cs_tex_width, _cs_tex_height, GL_RED, GL_FLOAT, swap);

		glBindTexture(GL_TEXTURE_2D, x);
		glBindImageTexture(0, x, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
		glBindTexture(GL_TEXTURE_2D, x0);
		glBindImageTexture(1, x0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
		glBindTexture(GL_TEXTURE_2D, swap_tex);
		glBindImageTexture(2, swap_tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);

		// number of workgroups: e. g. 512 / 16 --> 32 workgroups
		glDispatchCompute(_cs_tex_width / 32, _cs_tex_height / 32, 1); // 512^2 threads in blocks of 16^2
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
		
		set_bnd(x, b);
		// nach jedem Schritt Texture updaten
	}

	delete[] swap;
	swap = NULL;
}

void Visualize::advect(const GLuint &d, const GLuint &d0, const GLuint &u, const GLuint &v, const int b) {
	glUseProgram(_advect_Program);

	glUniform1f(_diffID_advect, diff);
	glUniform1f(_dtID_advect, dt);

	// bind to image unit so can write to specific pixels from the shader
	glBindTexture(GL_TEXTURE_2D, d);
	glBindImageTexture(0, d, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
	glBindTexture(GL_TEXTURE_2D, d0);
	glBindImageTexture(1, d0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
	glBindTexture(GL_TEXTURE_2D, u);
	glBindImageTexture(2, u, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
	glBindTexture(GL_TEXTURE_2D, v);
	glBindImageTexture(3, v, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);

	glDispatchCompute(_cs_tex_width / 32, _cs_tex_height / 32, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	set_bnd(d, b);
}

void Visualize::project(const GLuint &u, const GLuint &v, const GLuint &p, const GLuint &div) {
	glUseProgram(_project_1_Program);

	// bind to image unit so can write to specific pixels from the shader
	glBindTexture(GL_TEXTURE_2D, u);
	glBindImageTexture(0, u, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
	glBindTexture(GL_TEXTURE_2D, v);
	glBindImageTexture(1, v, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
	glBindTexture(GL_TEXTURE_2D, p);
	glBindImageTexture(2, p, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
	glBindTexture(GL_TEXTURE_2D, div);
	glBindImageTexture(3, div, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);

	glDispatchCompute(_cs_tex_width / 32, _cs_tex_height / 32, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	set_bnd(div, 0); set_bnd(p, 0);

	diffuse(p, div, 1.f, 4.f, 0);
	
	glUseProgram(_project_2_Program);

	// bind to image unit so can write to specific pixels from the shader
	glBindTexture(GL_TEXTURE_2D, u);
	glBindImageTexture(0, u, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
	glBindTexture(GL_TEXTURE_2D, v);
	glBindImageTexture(1, v, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
	glBindTexture(GL_TEXTURE_2D, p);
	glBindImageTexture(2, p, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);

	glDispatchCompute(_cs_tex_width / 32, _cs_tex_height / 32, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	set_bnd(u, 1); set_bnd(v, 2);
}

void Visualize::velocity() {
	glUseProgram(_velocity_Program);

	glBindTexture(GL_TEXTURE_2D, u_tex);
	glBindImageTexture(0, u_tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
	glBindTexture(GL_TEXTURE_2D, v_tex);
	glBindImageTexture(1, v_tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
	glBindTexture(GL_TEXTURE_2D, _velocity_tex);
	glBindImageTexture(2, _velocity_tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);

	glDispatchCompute(_cs_tex_width / 32, _cs_tex_height / 32, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void Visualize::deposition() {
	// start with an energy X at Height Z
	// and then look up the energy values of the lookup table for the corresponding height
	// and set the energy values accordingly
	glUseProgram(_deposition_Program);
	
	// bind to image unit so can write to specific pixels from the shader
	glBindTexture(GL_TEXTURE_2D, dens_tex);
	glBindImageTexture(0, dens_tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
	//glBindTexture(GL_TEXTURE_2D, _energyTex);
	//glBindImageTexture(1, _energyTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
	glBindTexture(GL_TEXTURE_3D, _3dtex);
	glBindImageTexture(2, _3dtex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F /*GL_RGBA8UI*/);
	
	glDispatchCompute(_cs_tex_width / 32, _cs_tex_height / 32, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}


// visualization part
//----------------------------------------------------------------------
void Visualize::transformation() 
{
	mat4x4 proj, view, model;
	mat4x4_perspective(proj, FoV, aspectRatio, zNear, zFar);
	mat4x4_look_at(view, eye, lookat, vec3{ 0.f, -1.0f, 0.f });
	mat4x4_identity(model);
	mat4x4_mul(mvp, view, model);
	mat4x4_mul(mvp, proj, mvp);
}

void Visualize::draw_velocity()
{
	GLfloat *data_u = 0; GLfloat *data_v = 0;
	data_u = new GLfloat[_cs_tex_width * _cs_tex_height]; data_v = new GLfloat[_cs_tex_width * _cs_tex_height];

	// _swap_tex = first;
	glBindTexture(GL_TEXTURE_2D, u_prev_tex);	glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, data_u);
	glBindTexture(GL_TEXTURE_2D, v_prev_tex); glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, data_v);

	float hx = 1.f / static_cast<float>(_cs_tex_width);
	float hy = 1.f / static_cast<float>(_cs_tex_height);
	int idx = 0;

	glUniformMatrix4fv(_mvpIDVel, 1, GL_FALSE, &mvp[0][0]);
	glUniform1f(_stretchIDxVel, _stretchx);
	glUniform1f(_stretchIDyVel, _stretchy);

	glLineWidth(0.2f);

	float angle = -0.5f / 180.0f * 3.141592f;

	static GLfloat vel_line_buffer[_vel_buffer_size]{};
	for (int j = 8; j < _cs_tex_height; j += 16) {
		float y = (j + 0.5f) * hy * 2;
		for (int i = 8; i < _cs_tex_width; i += 16) {
			float x = (i + 0.5f) * hx * 2;
			int index = j * _cs_tex_width + i;

			//float newx = x - 1 + hx * 12;// (x + (data_u[index] * hx * 2));
			//float newy = y - 1 + hy * 12;// (y + (data_v[index] * hy * 2));

			float newx = (x - 1 + (data_u[index] * hx * 12));
			float newy = (y - 1 + (data_v[index] * hy * 12));

			float normalize = sqrt((newx - x) * (newx - x) + (newy - y) * (newy - y));
			if (normalize == 0) normalize = 1;

			float newxrr = x - newx; float newxrl = x - newx;
			float newyrr = y - newy; float newyrl = y - newy;

			// base line
			vel_line_buffer[idx + 0] = x - 1;
			vel_line_buffer[idx + 1] = y - 1;
			vel_line_buffer[idx + 2] = 0.f;
			vel_line_buffer[idx + 3] = newx;
			vel_line_buffer[idx + 4] = newy;
			vel_line_buffer[idx + 5] = 0.f;

			newxrr = newxrr * cos(angle) - newyrr * sin(angle); newxrr = (newxrr + newx - 1.f); newxrr -= (newxrr - newx) / 2.f;
			newyrr = newxrr * sin(angle) + newyrr * cos(angle); newyrr = (newyrr + newy - 1.f); newyrr -= (newyrr - newy) / 2.f;

			// rotate half base line to the right
			vel_line_buffer[idx + 6] = newx;
			vel_line_buffer[idx + 7] = newy;
			vel_line_buffer[idx + 8] = 0.f;
			vel_line_buffer[idx + 9] = newxrr;
			vel_line_buffer[idx + 10] = newyrr;
			vel_line_buffer[idx + 11] = 0.f;

			newxrl = newxrl * cos(-angle) - newyrl * sin(-angle); newxrl = (newxrl + newx - 1.f); newxrl -= (newxrl - newx) / 2.f;
			newyrl = newxrl * sin(-angle) + newyrl * cos(-angle); newyrl = (newyrl + newy - 1.f); newyrl -= (newyrl - newy) / 2.f;

			// rotate half base line to the left
			vel_line_buffer[idx + 12] = newx;
			vel_line_buffer[idx + 13] = newy;
			vel_line_buffer[idx + 14] = 0.f;
			vel_line_buffer[idx + 15] = newxrl;
			vel_line_buffer[idx + 16] = newyrl;
			vel_line_buffer[idx + 17] = 0.f;
			idx += 18;
		}
	}

	delete[] data_u; data_u = NULL;
	delete[] data_v; data_v = NULL;

	GLuint velBuff;
	glGenBuffers(1, &velBuff);
	glBindBuffer(GL_ARRAY_BUFFER, velBuff);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vel_line_buffer), vel_line_buffer, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, velBuff);
	glVertexAttribPointer(
		0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);

	glDrawArrays(GL_LINES, 0, (_cs_tex_width / 16) * (_cs_tex_height / 16) * (2 + 2 + 2)); // Starting from vertex 0; 2 vertices total -> 1 line
	glDisableVertexAttribArray(0);

	glDeleteBuffers(1, &velBuff);
}

void Visualize::transformTo3D() {
	// get 2D image data from given texture
	GLfloat *swap = 0;
	swap = new GLfloat[_cs_tex_width * _cs_tex_height];

	// _swap_tex = first;
	glBindTexture(GL_TEXTURE_2D, dens_tex);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, swap);

	// transform 2D data to 3D data
	GLfloat *convert = 0;
	convert = new GLfloat[_cs_tex_width * _cs_tex3d_height * _cs_tex_depth];

	for (int j = 0; j < _cs_tex_height; ++j) {
		for (int i = 0; i < _cs_tex_width; ++i) {
			int index = i + j * _cs_tex_width * _cs_tex3d_height;
			convert[index] = swap[i + j * _cs_tex_width];
		}
	}

	// creating the texture here and deleting it at the end of the while in render() is more memory efficient
	glBindTexture(GL_TEXTURE_3D, _3dtex);
	glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, _cs_tex_width, _cs_tex3d_height, _cs_tex_depth, GL_RED, GL_FLOAT, convert);

	delete[] swap; swap = NULL;
	delete[] convert; convert = NULL;
}

void Visualize::draw() 
{
	glUseProgram(_fvProgram);

	switch (texIDX) {
	case 0: glBindTexture(GL_TEXTURE_2D, u_prev_tex); break;
	case 1: glBindTexture(GL_TEXTURE_2D, v_prev_tex); break;
	case 2: glBindTexture(GL_TEXTURE_2D, dens_prev_tex); break;
	case 3: glBindTexture(GL_TEXTURE_2D, u_tex); break;
	case 4: glBindTexture(GL_TEXTURE_2D, v_tex); break;
	case 5: glBindTexture(GL_TEXTURE_2D, dens_tex); break;
	case 6: glBindTexture(GL_TEXTURE_2D, _velocity_tex); break;
	case 7: glBindTexture(GL_TEXTURE_2D, swap_tex); break;
	case 8: glBindTexture(GL_TEXTURE_2D, _sim_tex); break;
	case 9: glBindTexture(GL_TEXTURE_3D, _3dtex); break;
	default: break;
	}

	glBindTexture(GL_TEXTURE_2D, _energyTex);
	glUniform1i(_alphaTexID, 10);
	glUniform1i(_texID, texIDX);
	transformation();
	glUniformMatrix4fv(_mvpID, 1, GL_FALSE, &mvp[0][0]);
	glUniform1f(_depthID, _depth);

	// 1st attribute buffer : vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, _quad_vertexBuffer);
	glVertexAttribPointer(
		0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// 2nd attribute buffer : texture coordinates
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, _quad_texcoordBuffer);
	glVertexAttribPointer(
		1,                  // attribute 1. No particular reason for 1, but must match the layout in the shader.
		2,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Draw the triangle !
	glDrawArrays(GL_TRIANGLES, 0, 6); // Starting from vertex 0; 3 vertices total -> 1 triangle
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
}

void Visualize::VisRender() 
{
	try {
		glViewport(0, 0, _window_width, _window_height);
		glClearColor(0.f, 0.f, 0.f, 1.f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		// -----------------------------------------------------------------------
		// simulation compuation part
		// -----------------------------------------------------------------------
		float _a_d = dt * diff * _cs_tex_width * _cs_tex_height;
		float _a_dv = dt * visc * _cs_tex_width * _cs_tex_height;
		
		// vel step
		float _aq_d = 1.f + 4.f * _a_dv;
		init_source();
		swap(u_prev_tex, u_tex); diffuse(u_tex, u_prev_tex, _a_dv, _aq_d, 1); 
		swap(v_prev_tex, v_tex); diffuse(v_tex, v_prev_tex, _a_dv, _aq_d, 2);
		project(u_tex, v_tex, u_prev_tex, v_prev_tex);
		swap(u_prev_tex, u_tex); swap(v_prev_tex, v_tex);
		advect(u_tex, u_prev_tex, u_prev_tex, v_prev_tex, 1); 
		advect(v_tex, v_prev_tex, u_prev_tex, v_prev_tex, 2);
		project(u_tex, v_tex, u_prev_tex, v_prev_tex);
		
		// dens step
		_aq_d = 1.f + 4.f * _a_d;
		swap(dens_prev_tex, dens_tex); diffuse(dens_tex, dens_prev_tex, _a_d, _aq_d, 0);
		swap(dens_prev_tex, dens_tex); advect(dens_tex, dens_prev_tex, u_tex, v_tex, 0);
		
		// get velocity tex
		velocity();
		

		// -----------------------------------------------------------------------
		// visualization part
		// -----------------------------------------------------------------------		
		transformTo3D();
		deposition();

		_depth = 0.f;
		for (int i = 0; i < _cs_tex_depth; ++i) {
			draw();
			++_depth;
		}
		
		//glUseProgram(_velProgram);
		//draw_velocity();


		// -----------------------------------------------------------------------
		// check OpenGL error
		// -----------------------------------------------------------------------
		while ((err = glGetError()) != GL_NO_ERROR) {
			std::cout << "OpenGL error: " << err << std::endl;
		}
	} 
	catch (const std::exception &e) 
	{
		std::cerr << e.what() << "\nPress enter.\n";
		getchar();
	}
}
