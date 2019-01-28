/*
* NGMeshStructs.h
*
* Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef NGMESH_STRUCTS_H_INCLUDED
#define NGMESH_STRUCTS_H_INCLUDED

#include <vector>

#include "vislib/graphics/gl/IncludeAllGL.h"

enum UpdateBits {
	UPDATE_ALL_BIT = 0x1,       //< delete existing and readd data afterwards 
	UPDATE_INDVIDUAL_BIT = 0x2, //< check individual elements/batches for updates
	DATA_ADDED_BIT = 0x4        //< new data added to call, keep exising and add new data

	//MATERIAL_SHADER_BIT = 0x1,
	//MATERIAL_TEXTURES_BIT = 0x2,
	//MESH_UPDATE_ALL_BIT = 0x4,
	//MESH_UPDATE_INDVIDUAL_BIT = 0x8,
	//MESH_BATCH_ADDED_BIT = 010,
	//TASK_DRAWCOMMANDS_BIT = 0x20,
	//TASK_PEROBJECTDATA_BIT = 0x40
};

struct BufferAccessor
{
	std::byte*	raw_data;
	size_t		byte_size;
};



/**
* Basic Texture Layout descriptor taken over from glOwl.
*/
struct TextureLayout
{
	TextureLayout()
		: width(0), internal_format(0), height(0), depth(0), format(0), type(0), levels(0) {}
	/**
	 * \param internal_format Specifies the (sized) internal format of a texture (e.g. GL_RGBA32F)
	 * \param width Specifies the width of the texture in pixels.
	 * \param height Specifies the height of the texture in pixels. Will be ignored by Texture1D.
	 * \param depth Specifies the depth of the texture in pixels. Will be ignored by Texture1D and Texture2D.
	 * \param format Specifies the format of the texture (e.g. GL_RGBA)
	 * \param type Specifies the type of the texture (e.g. GL_FLOAT)
	 */
	TextureLayout(GLint internal_format, int width, int height, int depth, GLenum format, GLenum type, GLsizei levels)
		: internal_format(internal_format), width(width), height(height), depth(depth), format(format), type(type), levels(levels) {}

	/**
	* \param internal_format Specifies the (sized) internal format of a texture (e.g. GL_RGBA32F)
	* \param width Specifies the width of the texture in pixels.
	* \param height Specifies the height of the texture in pixels. Will be ignored by Texture1D.
	* \param depth Specifies the depth of the texture in pixels. Will be ignored by Texture1D and Texture2D.
	* \param format Specifies the format of the texture (e.g. GL_RGBA)
	* \param type Specifies the type of the texture (e.g. GL_FLOAT)
	* \param int_parameters A list of integer texture parameters, each given by a pair of name and value (e.g. {{GL_TEXTURE_SPARSE_ARB,GL_TRUE},{...},...}
	* \param int_parameters A list of float texture parameters, each given by a pair of name and value (e.g. {{GL_TEXTURE_MAX_ANISOTROPY_EX,4.0f},{...},...}
	*/
	TextureLayout(GLint internal_format, int width, int height, int depth, GLenum format, GLenum type, GLsizei levels, std::vector<std::pair<GLenum, GLint>> const& int_parameters, std::vector<std::pair<GLenum, GLfloat>> const& float_parameters)
		: internal_format(internal_format), width(width), height(height), depth(depth), format(format), type(type), levels(levels), int_parameters(int_parameters), float_parameters(float_parameters) {}
	TextureLayout(GLint internal_format, int width, int height, int depth, GLenum format, GLenum type, GLsizei levels, std::vector<std::pair<GLenum, GLint>> && int_parameters, std::vector<std::pair<GLenum, GLfloat>> && float_parameters)
		: internal_format(internal_format), width(width), height(height), depth(depth), format(format), type(type), levels(levels), int_parameters(int_parameters), float_parameters(float_parameters) {}

	GLint internal_format;
	int width;
	int height;
	int depth;
	GLenum format;
	GLenum type;

	GLsizei levels;

	std::vector<std::pair<GLenum, GLint>> int_parameters;
	std::vector<std::pair<GLenum, GLfloat>> float_parameters;
};

#endif