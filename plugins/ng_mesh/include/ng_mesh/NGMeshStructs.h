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
	SHADER_BIT = 0x1,
	MESH_BIT = 0x2,
	DRAWCOMMANDS_BIT = 0x4,
	MESHPARAMS_BIT = 0x8,
	MATERIAL_BIT = 0x10
};

struct BufferAccessor
{
	std::byte*	raw_data;
	size_t		byte_size;
};

/**
* Basic Vertex Layout descriptor taken over from glOwl.
*/
struct VertexLayout
{
	struct Attribute
	{
		Attribute() {}
		Attribute(GLint size, GLenum type, GLboolean normalized, GLsizei offset)
			: size(size), type(type), normalized(normalized), offset(offset) {}

		inline bool operator==(Attribute const& other) {
			return ( (type == other.type) && (size == other.size) && (normalized == other.normalized) && (offset == other.offset) );
		}

		GLenum type; // component type, e.g. GL_FLOAT
		GLint size; // component count, e.g. 2 (for VEC2)
		GLboolean normalized; // GL_TRUE or GL_FALSE
		GLsizei offset;
	};

	VertexLayout() : stride(0), attributes() {}
	VertexLayout(GLsizei byte_size, const std::vector<Attribute>& attributes)
		: stride(byte_size), attributes(attributes) {}
	VertexLayout(GLsizei byte_size, std::vector<Attribute>&& attributes)
		: stride(byte_size), attributes(attributes) {}

	inline bool operator==(VertexLayout const& other) {
		bool rtn = false;
		rtn = rtn && (stride == other.stride);
		rtn = rtn && (attributes.size() == other.attributes.size());

		if (rtn)
		{
			for (int i = 0; i < attributes.size(); ++i)
				rtn = rtn && (attributes[i] == other.attributes[i]);
		}

		return rtn;
	}

	GLsizei stride;
	std::vector<Attribute> attributes;
};

inline size_t computeAttributeByteSize(VertexLayout::Attribute attr)
{
	size_t retval = 0;
	switch (attr.type)
	{
	case GL_BYTE:
		retval = sizeof(GLbyte);
		break;
	case GL_SHORT:
		retval = sizeof(GLshort);
		break;
	case GL_INT:
		retval = sizeof(GLint);
		break;
	case GL_FIXED:
		retval = sizeof(GLfixed);
		break;
	case GL_FLOAT:
		retval = sizeof(GLfloat);
		break;
	case GL_HALF_FLOAT:
		retval = sizeof(GLhalf);
		break;
	case GL_DOUBLE:
		retval = sizeof(GLdouble);
		break;
	case GL_UNSIGNED_BYTE:
		retval = sizeof(GLubyte);
		break;
	case GL_UNSIGNED_SHORT:
		retval = sizeof(GLushort);
		break;
	case GL_UNSIGNED_INT:
		retval = sizeof(GLuint);
		break;
	case GL_INT_2_10_10_10_REV:
		retval = sizeof(GLuint);
		break;
	case GL_UNSIGNED_INT_2_10_10_10_REV:
		retval = sizeof(GLuint);
		break;
	case GL_UNSIGNED_INT_10F_11F_11F_REV:
		retval = sizeof(GLuint);
		break;
	default:
		break;
	}

	retval *= attr.size;

	return retval;
};

struct DrawElementsCommand
{
	GLuint cnt;
	GLuint instance_cnt;
	GLuint first_idx;
	GLuint base_vertex;
	GLuint base_instance;
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