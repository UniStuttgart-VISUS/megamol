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

		GLenum type;
		GLint size;
		GLboolean normalized;
		GLsizei offset;
	};

	VertexLayout() : stride(0), attributes() {}
	VertexLayout(GLsizei byte_size, const std::vector<Attribute>& attributes)
		: stride(byte_size), attributes(attributes) {}
	VertexLayout(GLsizei byte_size, std::vector<Attribute>&& attributes)
		: stride(byte_size), attributes(attributes) {}

	GLsizei stride;
	std::vector<Attribute> attributes;
};

struct DrawElementsCommand
{
	GLuint cnt;
	GLuint instance_cnt;
	GLuint first_idx;
	GLuint base_vertex;
	GLuint base_instance;
};

#endif