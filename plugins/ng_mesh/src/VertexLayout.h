#ifndef VertexLayout_h
#define VertexLayout_h

#include "vislib/graphics/gl/IncludeAllGL.h"

#include <vector>

/**
 * Basic Vertex Layout descriptor taken over from glOwl.
 */
struct VertexLayout
{
	struct Attribute
	{
		Attribute(GLenum type, GLint size, GLboolean normalized, GLsizei offset)
			: size(size), type(type), normalized(normalized), offset(offset) {}

		GLint size;
		GLenum type;
		GLboolean normalized;
		GLsizei offset;
	};

	VertexLayout() : byte_size(0), attributes() {}
	VertexLayout(GLsizei byte_size, const std::vector<Attribute>& attributes)
		: byte_size(byte_size), attributes(attributes) {}
	VertexLayout(GLsizei byte_size, std::vector<Attribute>&& attributes)
		: byte_size(byte_size), attributes(attributes) {}

	GLsizei byte_size;
	std::vector<Attribute> attributes;
};

#endif // !VertexLayout_hpp