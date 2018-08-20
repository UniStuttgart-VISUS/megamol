#ifndef VISUALIZE_H
#define VISUALIZE_H

#include "stdafx.h"

#include "mmcore/view/CallRender3D.h"
#include "linmath.h"

#include <iostream>

class Visualize {
public:
	Visualize();
	virtual ~Visualize();

	bool initialize();
	void VisRender();

	/////////////////////////////////////////////////////////////////////////////////////
	// camera settings
	/////////////////////////////////////////////////////////////////////////////////////
	vec3 eye{ 0.f, 10.f, 5.f };
	vec3 lookat{ 0.f, 0.f, 0.f };
	vec3 up{ 0.f, -1.f, 0.f };
	float FoV = 0.f;
	float aspectRatio = 0.f;
	float zNear = 0.f;
	float zFar = 0.f;
	
	/////////////////////////////////////////////////////////////////////////////////////
	// other settings
	/////////////////////////////////////////////////////////////////////////////////////
	GLuint _window_width, _window_height;
	

private:
	// setup
	void init();
	void shaderSetup();
	void bufferSetup();
	void textureSetup();
	void uniformLocations();
	void setConstantUniforms();
	GLuint compTexture();
	void preCalculateDeposition();
	
	// simulation
	void swap(const GLuint &first, const GLuint &second);
	void set_bnd(const GLuint &x, const int b);
	void init_source();
	void diffuse(const GLuint &x, const GLuint &x0, const float a, const float aq, const int b);
	void advect(const GLuint &d, const GLuint &d0, const GLuint &u, const GLuint &v, const int b);
	void project(const GLuint &u, const GLuint &v, const GLuint &p, const GLuint &div);
	void velocity();
	void deposition();

	// visualize
	void transformation();
	void draw_velocity();
	void transformTo3D();
	void draw();


	/////////////////////////////////////////////////////////////////////////////////////
	// variables
	/////////////////////////////////////////////////////////////////////////////////////
	static const GLuint _cs_tex_width = 256;
	static const GLuint _cs_tex_height = 256;
	static const GLuint _cs_tex_depth = 256;
	static const GLuint _cs_tex3d_height = 32;
	float _stretch = 2.f;
	float _stretchy = _stretch;
	float _stretchx = _stretch * static_cast<float>(_cs_tex_width) / static_cast<float>(_cs_tex_height);
	static const int _vel_buffer_size = (_cs_tex_width / 16) * (_cs_tex_height / 16) * (6 * 3);
	mat4x4 mvp;

	// shaders and programs
	GLuint _fShader;
	GLuint _vShader;
	GLuint _velShader;
	GLuint _init_source_Shader;
	GLuint _diffuse_odd_Shader;
	GLuint _diffuse_even_Shader;
	GLuint _advect_Shader;
	GLuint _project_1_Shader;
	GLuint _project_2_Shader;
	GLuint _set_bnd_edge_Shader;
	GLuint _set_bnd_corner_Shader;
	GLuint _velocity_Shader;
	GLuint _deposition_Shader;

	GLuint _fvProgram;
	GLuint _velProgram;
	GLuint _init_source_Program;
	GLuint _diffuse_odd_Program;
	GLuint _diffuse_even_Program;
	GLuint _advect_Program;
	GLuint _project_1_Program;
	GLuint _project_2_Program;
	GLuint _set_bnd_e_Program;
	GLuint _set_bnd_c_Program;
	GLuint _velocity_Program;
	GLuint _deposition_Program;


	// simulation IDs etc.
	GLuint _velocity_texID;
	GLuint _vel_uID;
	GLuint _vel_vID;
	GLuint _sim_tex;
	GLuint _velocity_tex;

	// frag.glsl
	GLuint _texID, _alphaTexID;
	GLuint _mvpID, _mvpIDVel;
	GLuint _stretchIDx, _stretchIDy;
	GLuint _stretchIDxVel, _stretchIDyVel;

	// set_bnd part
	GLuint _xsizeID_set_bnd_e, _ysizeID_set_bnd_e;
	GLuint _xsizeID_set_bnd_c, _ysizeID_set_bnd_c;
	GLuint _bnd_e_texID, _bnd_c_texID;
	GLuint _bID;

	// init_source part
	GLuint  _xsizeID_init_source, _ysizeID_init_source;
	GLuint _dtID;
	GLuint _initFlagID;
	GLuint _u_prev_texID, _v_prev_texID, _dens_prev_texID;
	GLuint _u_texID, _v_texID, _dens_texID;
	bool _initFlag = true;

	// diffuse part
	GLuint  _xsizeID_diffuse_odd, _ysizeID_diffuse_odd;
	GLuint _x_texID_odd, _x0_texID_odd, _help_diffuse_texID_odd;
	GLuint _aID_odd, _aqID_odd;
	GLuint  _xsizeID_diffuse_even, _ysizeID_diffuse_even;
	GLuint _x_texID_even, _x0_texID_even, _help_diffuse_texID_even;
	GLuint _aID_even, _aqID_even;

	// advect part
	GLuint _xsizeID_advect, _ysizeID_advect;
	GLuint _d_texID, _d0_texID;
	GLuint _u_advect_texID, _v_advect_texID;
	GLuint _diffID_advect, _dtID_advect;

	// project part
	GLuint _xsizeID_project1, _ysizeID_project1;
	GLuint _xsizeID_project2, _ysizeID_project2;
	GLuint _u_texID_project1, _v_texID_project1, _p_texID_project1, _div_texID_project1;
	GLuint _u_texID_project2, _v_texID_project2, _p_texID_project2;

	// 3D texture
	GLuint _3dtex;
	GLuint _depthID, _depthResID;
	float _depth;

	// deposition
	GLuint _heightID;
	GLuint _energyTex;
	static const int _Dcnt = 150;
	static const int _AzCNT = 256 * 8;
	GLfloat _Az[_AzCNT]{};

	GLuint _quad_VertexArrayID;
	GLuint _quad_vertexBuffer;
	GLuint _quad_texcoordBuffer;

	/////////////////////////////////////////////////////////////////////////////////////
	// fluid simulation textures
	/////////////////////////////////////////////////////////////////////////////////////
	float dt;
	float diff;
	float visc;

	GLuint u_prev_tex; 				// 0	
	GLuint v_prev_tex; 				// 1
	GLuint dens_prev_tex;			// 2

	GLuint u_tex;					// 3
	GLuint v_tex;					// 4
	GLuint dens_tex;				// 5

	GLuint swap_tex;

	GLenum err;
};

#endif // VISUALIZE_H