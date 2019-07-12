/*
* Mesh.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef GLOWL_MESH_H_INCLUDED
#define GLOWL_MESH_H_INCLUDED

#include "glowl/BufferObject.h"

#include "vislib/graphics/gl/IncludeAllGL.h"

namespace megamol {
	namespace mesh {

		/**
		* Basic Vertex Layout descriptor, courtesy of glOwl by Michael Becher.
		*/
		struct VertexLayout
		{
			struct Attribute
			{
				Attribute() {}
				Attribute(GLint size, GLenum type, GLboolean normalized, GLsizei offset)
					: size(size), type(type), normalized(normalized), offset(offset) {}

				inline bool operator==(Attribute const& other) {
					return ((type == other.type) && (size == other.size) && (normalized == other.normalized) && (offset == other.offset));
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
				bool retval = true;
                retval = retval && (stride == other.stride);
                retval = retval && (attributes.size() == other.attributes.size());

				if (retval){
                    for (int i = 0; i < attributes.size(); ++i) {
                        retval = retval && (attributes[i] == other.attributes[i]);
                    }
				}

				return retval;
			}

			GLsizei stride;
			std::vector<Attribute> attributes;
		};

		inline size_t computeByteSize(GLenum gl_type)
		{
			size_t retval = 0;
			switch (gl_type)
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

			return retval;
		}

		inline size_t computeAttributeByteSize(VertexLayout::Attribute attr)
		{
			size_t retval = computeByteSize(attr.type);
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
		 * OpenGL mesh class. Wraps all data and functionality required to store and use geometry with OpenGL. Courtesy of glOwl by Michael Becher.
		 */
		class Mesh
		{
		public:

			template<typename VertexContainer, typename IndexContainer>
			Mesh(VertexContainer const& vertices,
				IndexContainer const&	indices,
				VertexLayout const&		vertex_descriptor,
				GLenum					indices_type = GL_UNSIGNED_INT,
				GLenum					usage = GL_STATIC_DRAW,
				GLenum					primitive_type = GL_TRIANGLES)
				: m_ibo(GL_ELEMENT_ARRAY_BUFFER, indices, usage), //TODO ibo generation in constructor might fail? needs a bound vao?
				m_vertex_descriptor(vertex_descriptor),
				m_va_handle(0), m_indices_cnt(0), m_indices_type(indices_type), m_usage(usage), m_primitive_type(primitive_type)
			{
				m_vbos.emplace_back(std::make_unique<BufferObject>(GL_ARRAY_BUFFER, vertices, usage));

				glGenVertexArrays(1, &m_va_handle);

				// set attribute pointer and vao state
				glBindVertexArray(m_va_handle);
				m_ibo.bind();
				m_vbos.back()->bind();
				GLuint attrib_idx = 0;
				for (auto& attribute : vertex_descriptor.attributes)
				{
					glEnableVertexAttribArray(attrib_idx);
					glVertexAttribPointer(attrib_idx, attribute.size, attribute.type, attribute.normalized, vertex_descriptor.stride, (GLvoid*)attribute.offset);

					attrib_idx++;
				}
				glBindVertexArray(0);
				glBindBuffer(GL_ARRAY_BUFFER, 0);

				GLuint vi_size = static_cast<GLuint>(indices.size() * sizeof(IndexContainer::value_type));

				switch (m_indices_type)
				{
				case GL_UNSIGNED_INT:
					m_indices_cnt = static_cast<GLuint>(vi_size / 4);
					break;
				case GL_UNSIGNED_SHORT:
					m_indices_cnt = static_cast<GLuint>(vi_size / 2);
					break;
				case GL_UNSIGNED_BYTE:
					m_indices_cnt = static_cast<GLuint>(vi_size / 1);
					break;
				}
			}

			/**
			* C-interface Mesh constructor for interleaved data with a single vertex buffer object.
			*/
			Mesh(GLvoid const*		vertex_data,
				GLsizeiptr			vertex_data_byte_size,
				GLvoid const*		index_data,
				GLsizeiptr			index_data_byte_size,
				VertexLayout const& vertex_descriptor,
				GLenum				indices_type = GL_UNSIGNED_INT,
				GLenum				usage = GL_STATIC_DRAW,
				GLenum				primitive_type = GL_TRIANGLES)
				: m_ibo(GL_ELEMENT_ARRAY_BUFFER, index_data, index_data_byte_size, usage),
				m_vertex_descriptor(vertex_descriptor),
				m_va_handle(0), m_indices_cnt(0), m_indices_type(indices_type), m_usage(usage), m_primitive_type(primitive_type)
			{
				m_vbos.emplace_back(std::make_unique<BufferObject>(GL_ARRAY_BUFFER, vertex_data, vertex_data_byte_size, usage));

				glGenVertexArrays(1, &m_va_handle);

				// set attribute pointer and vao state
				glBindVertexArray(m_va_handle);
				m_vbos.back()->bind();

				// dirty hack to make ibo work as BufferObject
				//m_ibo = std::make_unique<BufferObject>(GL_ELEMENT_ARRAY_BUFFER, index_data, index_data_byte_size, usage);
				m_ibo.bind();

				GLuint attrib_idx = 0;
				for (auto& attribute : vertex_descriptor.attributes)
				{
					glEnableVertexAttribArray(attrib_idx);
					glVertexAttribPointer(attrib_idx, attribute.size, attribute.type, attribute.normalized, vertex_descriptor.stride, reinterpret_cast<GLvoid*>(attribute.offset));

					attrib_idx++;
				}
				glBindVertexArray(0);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

				switch (m_indices_type)
				{
				case GL_UNSIGNED_INT:
					m_indices_cnt = static_cast<GLuint>(index_data_byte_size / 4);
					break;
				case GL_UNSIGNED_SHORT:
					m_indices_cnt = static_cast<GLuint>(index_data_byte_size / 2);
					break;
				case GL_UNSIGNED_BYTE:
					m_indices_cnt = static_cast<GLuint>(index_data_byte_size / 1);
					break;
				}
			}

			/**
			 * Hybrid C-interface Mesh constructor for non-interleaved data with one vertex buffer object per vertex attribute.
			 */
			Mesh(std::vector<GLvoid*> const&	vertex_data,
				std::vector<size_t> const&		vertex_data_byte_sizes,
				GLvoid const*					index_data,
				GLsizeiptr						index_data_byte_size,
				VertexLayout const&				vertex_descriptor,
				GLenum							indices_type = GL_UNSIGNED_INT,
				GLenum							usage = GL_STATIC_DRAW,
				GLenum							primitive_type = GL_TRIANGLES)
				: m_ibo(GL_ELEMENT_ARRAY_BUFFER, index_data, index_data_byte_size, usage),
				m_vertex_descriptor(vertex_descriptor),
				m_va_handle(0), m_indices_cnt(0), m_indices_type(indices_type), m_usage(usage), m_primitive_type(primitive_type)
			{
				for (unsigned int i = 0; i < vertex_data.size(); ++i)
					m_vbos.emplace_back(std::make_unique<BufferObject>(GL_ARRAY_BUFFER, vertex_data[i], vertex_data_byte_sizes[i], usage));

				glGenVertexArrays(1, &m_va_handle);

				// set attribute pointer and vao state
				glBindVertexArray(m_va_handle);

				m_ibo.bind();

				// TODO check if vertex buffer count matches attribute count, throw exception if not?
				GLuint attrib_idx = 0;
				for (auto& attribute : vertex_descriptor.attributes)
				{
					m_vbos[attrib_idx]->bind();

					glEnableVertexAttribArray(attrib_idx);
					glVertexAttribPointer(attrib_idx, attribute.size, attribute.type, attribute.normalized, vertex_descriptor.stride, reinterpret_cast<GLvoid*>(attribute.offset));

					attrib_idx++;
				}
				glBindVertexArray(0);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

				switch (m_indices_type)
				{
				case GL_UNSIGNED_INT:
					m_indices_cnt = static_cast<GLuint>(index_data_byte_size / 4);
					break;
				case GL_UNSIGNED_SHORT:
					m_indices_cnt = static_cast<GLuint>(index_data_byte_size / 2);
					break;
				case GL_UNSIGNED_BYTE:
					m_indices_cnt = static_cast<GLuint>(index_data_byte_size / 1);
					break;
				}
			}

			//TODO iterator based constructor?

			~Mesh()
			{
				glDeleteVertexArrays(1, &m_va_handle);
			}

			template<typename VertexContainer>
			void loadVertexSubData(size_t idx, VertexContainer const& vertices, GLsizeiptr byte_offset) {
				if (idx < m_vbos.size())
				{
					m_vbos[idx]->loadSubData<VertexContainer>(vertices, byte_offset);
				}
			}

			void loadVertexSubData(size_t idx, GLvoid const* data, GLsizeiptr byte_size, GLsizeiptr byte_offset)
			{
				if (idx < m_vbos.size())
				{
					m_vbos[idx]->loadSubData(data, byte_size, byte_offset);
				}
			}

			void loadIndexSubData(GLvoid const* data, GLsizeiptr byte_size, GLsizeiptr byte_offset)
			{
				m_ibo.loadSubData(data, byte_size, byte_offset);
			}

			void bindVertexArray() const { glBindVertexArray(m_va_handle); }

			VertexLayout getVertexLayout() const { return m_vertex_descriptor; }

			GLenum getIndicesType() const { return m_indices_type; }

			GLenum getPrimitiveType() const { return m_primitive_type; }

			GLsizeiptr getVertexBufferByteSize(size_t idx) const {
				if (idx < m_vbos.size())
					return m_vbos[idx]->getByteSize();
				else
					return 0;
				// TODO: log some kind of error?
			}
			GLsizeiptr getIndexBufferByteSize() const { return m_ibo.getByteSize(); }

			Mesh(const Mesh &cpy) = delete;
			Mesh(Mesh&& other) = delete;
			Mesh& operator=(Mesh&& rhs) = delete;
			Mesh& operator=(const Mesh& rhs) = delete;

		private:
			std::vector<BufferObject::Ptr>	m_vbos;
			BufferObject					m_ibo;
			GLuint							m_va_handle;

			VertexLayout					m_vertex_descriptor;

			GLuint							m_indices_cnt;
			GLenum							m_indices_type;
			GLenum							m_usage;
			GLenum							m_primitive_type;
		};

	}
}

#endif // !GLOWL_MESH_H_INCLUDED
