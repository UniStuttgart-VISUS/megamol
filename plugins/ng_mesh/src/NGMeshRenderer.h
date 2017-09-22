/*
* NGMeshRenderer.h
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef NG_MESH_RENDERER_H_INCLUDED
#define NG_MESH_RENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/math/Matrix.h"

#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/CallRender3D.h"

#include "ng_mesh/CallNGMeshRenderBatches.h"

namespace megamol {
namespace ngmesh {

	class NGMeshRenderer : public megamol::core::view::Renderer3DModule
	{
	public:
		/**
		* Answer the name of this module.
		*
		* @return The name of this module.
		*/
		static const char *ClassName(void) {
			return "NGMeshRenderer";
		}

		/**
		* Answer a human readable description of this module.
		*
		* @return A human readable description of this module.
		*/
		static const char *Description(void) {
			return "Modern renderer for meshes. Works with render batches and indirect draw calls.";
		}

		/**
		* Answers whether this module is available on the current system.
		*
		* @return 'true' if the module is available, 'false' otherwise.
		*/
		static bool IsAvailable(void) {
#ifdef _WIN32
#if defined(DEBUG) || defined(_DEBUG)
			HDC dc = ::wglGetCurrentDC();
			HGLRC rc = ::wglGetCurrentContext();
			ASSERT(dc != NULL);
			ASSERT(rc != NULL);
#endif // DEBUG || _DEBUG
#endif // _WIN32
			return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable()
				&& isExtAvailable("GL_ARB_shader_draw_parameters")
				&& ogl_IsVersionGEQ(4, 3);
		}

		/** Ctor. */
		NGMeshRenderer();

		/** Dtor. */
		~NGMeshRenderer();

	protected:

		/**
		* Implementation of 'Create'.
		*
		* @return 'true' on success, 'false' otherwise.
		*/
		bool create();

		/**
		* Implementation of 'Release'.
		*/
		void release();

		/**
		* The get capabilities callback. The module should set the members
		* of 'call' to tell the caller its capabilities.
		*
		* @param call The calling call.
		*
		* @return The return value of the function.
		*/
		bool GetCapabilities(core::Call& call);

		/**
		* The get extents callback. The module should set the members of
		* 'call' to tell the caller the extents of its data (bounding boxes
		* and times).
		*
		* @param call The calling call.
		*
		* @return The return value of the function.
		*/
		bool GetExtents(core::Call& call);

		/**
		 * Access to render batches data via data call
		 */
		CallNGMeshRenderBatches* getData();

		/**
		 *
		 */
		void addRenderBatch(
			CallNGMeshRenderBatches::RenderBatchesData::ShaderPrgmData const&		shader_prgm_data,
			CallNGMeshRenderBatches::RenderBatchesData::MeshData const&				mesh_data,
			CallNGMeshRenderBatches::RenderBatchesData::DrawCommandData const&		draw_command_data,
			CallNGMeshRenderBatches::RenderBatchesData::MeshShaderParams const&		mesh_shader_params,
			CallNGMeshRenderBatches::RenderBatchesData::MaterialShaderParams const&	mtl_shader_params);

		/**
		 *
		 */
		void updateRenderBatch(
			size_t																	idx,
			CallNGMeshRenderBatches::RenderBatchesData::ShaderPrgmData const&		shader_prgm_data,
			CallNGMeshRenderBatches::RenderBatchesData::MeshData const&				mesh_data,
			CallNGMeshRenderBatches::RenderBatchesData::DrawCommandData const&		draw_command_data,
			CallNGMeshRenderBatches::RenderBatchesData::MeshShaderParams const&		mesh_shader_params,
			CallNGMeshRenderBatches::RenderBatchesData::MaterialShaderParams const&	mtl_shader_params,
			uint32_t																update_flags);

		/**
		* The render callback.
		*
		* @param call The calling call.
		*
		* @return The return value of the function.
		*/
		bool Render(core::Call& call);

	private:

		class BufferObject
		{
		public:
			template<typename Container>
			BufferObject(GLenum target, Container const& datastorage, GLenum usage = GL_DYNAMIC_DRAW)
				: m_target(target), m_handle(0), m_byte_size(static_cast<GLsizeiptr>(datastorage.size() * sizeof(Container::value_type))), m_usage(usage)
			{
				glGenBuffers(1, &m_handle);
				glBindBuffer(m_target, m_handle);
				glBufferData(m_target, m_byte_size, datastorage.data(), m_usage);
				glBindBuffer(m_target, 0);
			}

			BufferObject(GLenum target, GLvoid const* data, GLsizeiptr byte_size, GLenum usage = GL_DYNAMIC_DRAW)
				: m_target(target), m_handle(0), m_byte_size(byte_size), m_usage(usage)
			{
				glGenBuffers(1, &m_handle);
				glBindBuffer(m_target, m_handle);
				auto gl_err = glGetError();
				glBufferData(m_target, m_byte_size, data, m_usage);
				glBindBuffer(m_target, 0);
			}

			~BufferObject()
			{
				glDeleteBuffers(1, &m_handle);
			}

			BufferObject(const BufferObject& cpy) = delete;
			BufferObject(BufferObject&& other) = delete;
			BufferObject& operator=(BufferObject&& rhs) = delete;
			BufferObject& operator=(const BufferObject& rhs) = delete;

			template<typename Container>
			void loadSubData(Container const& datastorage, GLsizeiptr byte_offset = 0) const
			{
				// check if feasible
				if ((byte_offset + static_cast<GLsizeiptr>(datastorage.size() * sizeof(Container::value_type))) > m_byte_size)
				{
					// error message
					vislib::sys::Log::DefaultLog.WriteError("Invalid byte_offset or size for loadSubData");
					return;
				}

				glBindBuffer(m_target, m_handle);
				glBufferSubData(m_target, byte_offset, datastorage.size() * sizeof(Container::value_type), datastorage.data());
				glBindBuffer(m_target, 0);
			}

			void loadSubData(GLvoid const* data, GLsizeiptr byte_size, GLsizeiptr byte_offset = 0) const
			{
				// check if feasible
				if ((byte_offset + byte_size) > m_byte_size)
				{
					// error message
					vislib::sys::Log::DefaultLog.WriteError("Invalid byte_offset or size for loadSubData");
					return;
				}

				glBindBuffer(m_target, m_handle);
				glBufferSubData(m_target, byte_offset, byte_size, data);
				glBindBuffer(m_target, 0);
			}

			void bind() const
			{
				glBindBuffer(m_target, m_handle);
			}

			void bind(GLuint index) const
			{
				glBindBufferBase(m_target, index, m_handle);
			}

			GLenum getTarget() const
			{
				return m_target;
			}

			GLsizeiptr getByteSize() const
			{
				return m_byte_size;
			}

		private:
			GLenum		m_target;
			GLuint		m_handle;
			GLsizeiptr	m_byte_size;
			GLenum		m_usage;
		};

		class Mesh
		{
		public:

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

			template<typename VertexContainer, typename IndexContainer>
			Mesh(VertexContainer const& vertices,
				IndexContainer const&	indices,
				VertexLayout const&		vertex_descriptor,
				GLenum					indices_type = GL_UNSIGNED_INT,
				GLenum					usage = GL_STATIC_DRAW,
				GLenum					primitive_type = GL_TRIANGLES)
				: m_vbo<VertexContainer>(GL_ARRAY_BUFFER, vertices, usage),
				m_ibo<IndexContainer>(GL_ELEMENT_ARRAY_BUFFER, indices, usage), //TODO ibo generation in constructor might fail? needs a bound vao?
				m_va_handle(0), m_indices_cnt(0), m_indices_type(indices_type), m_usage(usage), m_primitive_type(primitive_type)
			{
				glGenVertexArrays(1, &m_va_handle);

				// set attribute pointer and vao state
				glBindVertexArray(m_va_handle);
				m_ibo.bind();
				m_vbo.bind();
				GLuint attrib_idx = 0;
				for (auto& attribute : vertex_descriptor.attributes)
				{
					glEnableVertexAttribArray(attrib_idx);
					glVertexAttribPointer(attrib_idx, attribute.size, attribute.type, attribute.normalized, vertex_descriptor.byte_size, (GLvoid*)attribute.offset);

					attrib_idx++;
				}
				glBindVertexArray(0);
				glBindBuffer(GL_ARRAY_BUFFER, 0);

				GLuint vi_size = static_cast<GLuint>(datastorage.size() * sizeof(Container::value_type));

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

			Mesh(GLvoid const*		vertex_data,
				GLsizeiptr			vertex_data_byte_size,
				GLvoid const*		index_data,
				GLsizeiptr			index_data_byte_size,
				VertexLayout const& vertex_descriptor,
				GLenum				indices_type = GL_UNSIGNED_INT,
				GLenum				usage = GL_STATIC_DRAW,
				GLenum				primitive_type = GL_TRIANGLES)
				: m_vbo(GL_ARRAY_BUFFER, vertex_data, vertex_data_byte_size, usage),
				m_ibo(GL_ELEMENT_ARRAY_BUFFER, index_data, index_data_byte_size, usage),
				m_va_handle(0), m_indices_cnt(0), m_indices_type(indices_type), m_usage(usage), m_primitive_type(primitive_type)
			{
				glGenVertexArrays(1, &m_va_handle);

				// set attribute pointer and vao state
				glBindVertexArray(m_va_handle);
				m_vbo.bind();

				// dirty hack to make ibo work as BufferObject
				//m_ibo = std::make_unique<BufferObject>(GL_ELEMENT_ARRAY_BUFFER, index_data, index_data_byte_size, usage);
				m_ibo.bind();

				GLuint attrib_idx = 0;
				for (auto& attribute : vertex_descriptor.attributes)
				{
					glEnableVertexAttribArray(attrib_idx);
					glVertexAttribPointer(attrib_idx, attribute.size, attribute.type, attribute.normalized, vertex_descriptor.byte_size, reinterpret_cast<GLvoid*>(attribute.offset));

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

			~Mesh()
			{
				glDeleteVertexArrays(1, &m_va_handle);
			}

			template<typename VertexContainer>
			void loadVertexSubData(VertexContainer const& vertices, GLsizeiptr byte_offset){
				m_vbo.loadSubData<VertexContainer>(vertices, byte_offset);
			}

			void bindVertexArray() const { glBindVertexArray(m_va_handle); }

			GLenum getIndicesType() const { return m_indices_type; }

			GLenum getPrimitiveType() const { return m_primitive_type; }

			GLsizeiptr getVertexBufferByteSize() const { return m_vbo.getByteSize(); }
			GLsizeiptr getIndexBufferByteSize() const { return m_ibo.getByteSize(); }

			Mesh(const Mesh &cpy) = delete;
			Mesh(Mesh&& other) = delete;
			Mesh& operator=(Mesh&& rhs) = delete;
			Mesh& operator=(const Mesh& rhs) = delete;

		private:
			BufferObject	m_vbo;
			BufferObject	m_ibo;
			GLuint			m_va_handle;

			VertexLayout	m_vertex_descriptor;

			GLuint			m_indices_cnt;
			GLenum			m_indices_type;
			GLenum			m_usage;
			GLenum			m_primitive_type;
		};

		typedef vislib::graphics::gl::GLSLShader GLSLShader;
		struct RenderBatch
		{
			GLsizei							draw_cnt;

			std::unique_ptr<GLSLShader>		shader_prgm;
			std::unique_ptr<Mesh>			mesh;
			std::unique_ptr<BufferObject>	draw_commands;
			std::unique_ptr<BufferObject>	mesh_shader_params;
			std::unique_ptr<BufferObject>	mtl_shader_params;
		};

		/** List of render batches ready for dispatching */
		std::vector<RenderBatch> m_render_batches;

		/** Render batches caller slot */
		megamol::core::CallerSlot m_renderBatches_callerSlot;
	};

}
}

#endif