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

//#include "ng_mesh/CallNGMeshRenderBatches.h"
#include "ng_mesh/BatchedMeshesDataCall.h"
#include "ng_mesh/MaterialDataStorage.h"
#include "ng_mesh/NGMeshStructs.h"
#include "ng_mesh/RenderTaskDataStorage.h"

#include "glowl/BufferObject.h"
#include "glowl/Mesh.h"

namespace megamol {
namespace ngmesh {


	/**
	 * Renderer module for rendering geometry with modern (OpenGL 4.3+) features.
	 * Objects for rendering are supplied in batches. Each  render batch can contain
	 * many objects that use the same shader program and also share the same geometry
	 * or at least the same vertex format.
	 * Per render batch, a single call of glMultiDrawElementsIndirect is made. The data
	 * for the indirect draw call is stored and accessed via SSBOs.
	 */
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
			return "Modern renderer for meshes. Objects are rendered in batches using indirect draw calls.";
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
		* The get extents callback. The module should set the members of
		* 'call' to tell the caller the extents of its data (bounding boxes
		* and times).
		*
		* @param call The calling call.
		*
		* @return The return value of the function.
		*/
		bool GetExtents(core::Call& call);

		//	/**
		//	 * Adds a new render batch by translating the given render batch data
		//	 * into a GPU RenderBatch ready for rendering.
		//	 *
		//	 * @param shader_prgm_data The data for loading a shader program, i.e. the shader filename
		//	 * @param mesh_data The data required for creating a mesh, i.e. vertex and index data, and vertex layout
		//	 * @param draw_command_data The data that describes the draw calls executed by glMultiDrawElementsIndirect
		//	 * @param obj_shader_params Additional data used by the shader during rendering that is specific to the rendered object
		//	 * @param mtl_shader_params Additional used by the shader during rendering that is specific to the material in use, i.e. texture handles
		//	 */
		//	void addRenderBatch(
		//		ShaderPrgmDataAccessor const&		shader_prgm_data,
		//		MeshDataAccessor const&				mesh_data,
		//		DrawCommandDataAccessor const&		draw_command_data,
		//		ObjectShaderParamsDataAccessor const&	obj_shader_params,
		//		MaterialShaderParamsDataAccessor const&	mtl_shader_params);
		//	
		//	/**
		//	* Selectively updates an existing render batch with new data.
		//	*
		//	* @param idx The index of the render batch that is updated
		//	* @param shader_prgm_data The data for loading a shader program, i.e. the shader filename
		//	* @param mesh_data The data required for creating a mesh, i.e. vertex and index data, and vertex layout
		//	* @param draw_command_data The data that describes the draw calls executed by glMultiDrawElementsIndirect
		//	* @param obj_shader_params Additional data used by the shader during rendering that is specific to the rendered object
		//	* @param mtl_shader_params Additional used by the shader during rendering that is specific to the material in use, i.e. texture handles
		//	* @param update_flags The bit flags that signal which parts of the render batch data needs to be updated
		//	*/
		//	void updateRenderBatch(
		//		size_t						idx,
		//		ShaderPrgmDataAccessor const&		shader_prgm_data,
		//		MeshDataAccessor const&				mesh_data,
		//		DrawCommandDataAccessor const&		draw_command_data,
		//		ObjectShaderParamsDataAccessor const&	obj_shader_params,
		//		MaterialShaderParamsDataAccessor const&	mtl_shader_params,
		//		uint32_t					update_flags);

		void updateMeshes(BatchedMeshesDataAccessor const& meshes, uint32_t update_flags);

		void updateMaterials(std::shared_ptr<MaterialsDataStorage> const& materials, uint32_t update_flags);

		void updateRenderTasks(std::shared_ptr<RenderTaskDataStorage> const& render_tasks, uint32_t update_flags);
	
		/**
		* The render callback.
		*
		* @param call The calling call.
		*
		* @return The return value of the function.
		*/
		bool Render(core::Call& call);

	private:

		/*
		 * OpenGL TextureLayout, Texture, Texture2D and Texture3D classes courtesy of glOwl by Michael Becher.
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

		class Texture
		{
		protected:
			std::string m_id; ///< Identifier set by application to help identifying textures

			GLuint		m_name; ///< OpenGL texture name given by glGenTextures
			GLuint64	m_texture_handle; ///< Actual OpenGL texture handle (used for bindless)

			GLenum		m_internal_format;
			GLenum		m_format;
			GLenum		m_type;

			GLsizei		m_levels;

			// TODO: Store texture parameters as well ?
		public:
			Texture(std::string id,
				GLint internal_format,
				GLenum format,
				GLenum type,
				GLsizei levels)
				: m_id(id),
				m_internal_format(internal_format),
				m_format(format),
				m_type(type),
				m_levels(levels) {}
			virtual ~Texture() { glDeleteTextures(1, &m_name); }
			Texture(const Texture &) = delete;

			virtual void bindTexture() const = 0;

			void bindImage(GLuint location, GLenum access) const
			{
				glBindImageTexture(location, m_name, 0, GL_TRUE, 0, access, m_internal_format);
			}

			void makeResident() { glMakeTextureHandleResidentARB(m_texture_handle); }
			void makeNonResident() { glMakeTextureHandleNonResidentARB(m_texture_handle); }

			virtual void updateMipmaps() = 0;

			virtual TextureLayout getTextureLayout() const = 0;

			std::string getId() const { return m_id; }

			GLuint getName() const { return m_name; }
			GLuint64 getTextureHandle() const { return m_texture_handle; }
			GLuint64 getImageHandle(GLint level, GLboolean layered, GLint layer) const {
				return glGetImageHandleARB(m_name, level, layered, layer, m_internal_format);
			}

			GLenum getInternalFormat() const { return m_internal_format; }
			GLenum getFormat() const { return m_format; }
			GLenum getType() const { return m_type; }
		};

		class Texture2D : public Texture
		{
		public:
			/**
			* \brief Constructor that creates and loads a 2D texture.
			*
			* \param id A identifier given to the texture object
			* \param layout A TextureLayout struct that specifies size, format and parameters for the texture
			* \param data Pointer to the actual texture data.
			* \param generateMipmap Specifies whether a mipmap will be created for the texture
			*/
			Texture2D(std::string id, TextureLayout const& layout, GLvoid * data, bool generateMipmap = false)
				:Texture(id, layout.internal_format, layout.format, layout.type, layout.levels), m_width(layout.width), m_height(layout.height)
			{
				glGenTextures(1, &m_name);

				glBindTexture(GL_TEXTURE_2D, m_name);

				for (auto& pname_pvalue : layout.int_parameters)
					glTexParameteri(GL_TEXTURE_2D, pname_pvalue.first, pname_pvalue.second);

				for (auto& pname_pvalue : layout.float_parameters)
					glTexParameterf(GL_TEXTURE_2D, pname_pvalue.first, pname_pvalue.second);

				GLsizei levels = 1;

				if (generateMipmap)
					levels = 1 + floor(log2(std::max(m_width, m_height)));

				glTexStorage2D(GL_TEXTURE_2D, levels, m_internal_format, m_width, m_height);

				if (data != nullptr)
					glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, m_format, m_type, data);

				if (generateMipmap)
					glGenerateMipmap(GL_TEXTURE_2D);

				glBindTexture(GL_TEXTURE_2D, 0);

				m_texture_handle = glGetTextureHandleARB(m_name);

				GLenum err = glGetError();
				if (err != GL_NO_ERROR)
				{
					// "Do something cop!"
					//std::cerr << "GL error during texture (id: " << id << ") creation: " << err << std::endl;
				}
			}
			Texture2D(const Texture2D&) = delete;
			Texture2D(Texture2D&& other) = delete;
			Texture2D& operator=(const Texture2D& rhs) = delete;
			Texture2D& operator=(Texture2D&& rhs) = delete;

			/**
			* \brief Bind the texture.
			*/
			void bindTexture() const
			{
				glBindTexture(GL_TEXTURE_2D, m_name);
			}

			void updateMipmaps()
			{
				glBindTexture(GL_TEXTURE_2D, m_name);
				glGenerateMipmap(GL_TEXTURE_2D);
				glBindTexture(GL_TEXTURE_2D, 0);
			}

			/**
			* \brief Reload the texture with any new format, type and size.
			*
			* \param layout A TextureLayout struct that specifies size, format and parameters for the texture
			* \param data Pointer to the actual texture data.
			* \param generateMipmap Specifies whether a mipmap will be created for the texture
			*/
			void reload(TextureLayout const& layout, GLvoid const * data, bool generateMipmap = false)
			{
				m_width = layout.width;
				m_height = layout.height;
				m_internal_format = layout.internal_format;
				m_format = layout.format;
				m_type = layout.type;

				glDeleteTextures(1, &m_name);

				glGenTextures(1, &m_name);

				glBindTexture(GL_TEXTURE_2D, m_name);

				for (auto& pname_pvalue : layout.int_parameters)
					glTexParameteri(GL_TEXTURE_2D, pname_pvalue.first, pname_pvalue.second);

				for (auto& pname_pvalue : layout.float_parameters)
					glTexParameterf(GL_TEXTURE_2D, pname_pvalue.first, pname_pvalue.second);

				GLsizei levels = 1;

				if (generateMipmap)
					levels = 1 + floor(log2(std::max(m_width, m_height)));

				glTexStorage2D(GL_TEXTURE_2D, levels, m_internal_format, m_width, m_height);

				if (data != nullptr)
					glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, m_format, m_type, data);

				if (generateMipmap)
					glGenerateMipmap(GL_TEXTURE_2D);

				glBindTexture(GL_TEXTURE_2D, 0);

				GLenum err = glGetError();
				if (err != GL_NO_ERROR)
				{
					// "Do something cop!"
					//std::cerr << "GL error during  (id: " << m_id << ") reload: " << err << std::endl;
				}
			}

			TextureLayout getTextureLayout() const
			{
				return TextureLayout(m_internal_format, m_width, m_height, 1, m_format, m_type, m_levels);
			}

			unsigned int getWidth() const
			{
				return m_width;
			}

			unsigned int getHeight() const
			{
				return m_height;
			}

		private:
			unsigned int m_width;
			unsigned int m_height;
		};

		typedef vislib::graphics::gl::GLSLShader GLSLShader;

		struct BatchedMeshes
		{
			std::shared_ptr<Mesh>            mesh; //< OpenGL Mesh object that stores the geometry of batched meshes
			std::vector<DrawElementsCommand> submesh_draw_commands; //< draw commands that identifiy the individual meshes batched together in a single OpenGL Mesh
		};

		struct Material
		{
			std::shared_ptr<GLSLShader>             shader;
			std::string                             btf_name;
			//std::vector<std::shared_ptr<Texture2D>> textures;
		};

		/**
		 * A collection of GPU Resources required for rendering geometry using glMultiDrawElementsIndirect.
		 */
		struct RenderBatch
		{
			GLsizei							draw_cnt;          //< draw count, i.e. numer of objects in batch

			std::shared_ptr<GLSLShader>		shader_prgm;       //< shader program used for drawing objects in batch
			std::shared_ptr<Mesh>			mesh;              //< mesh object that stores geometry of objects in batch
			std::shared_ptr<BufferObject>	draw_commands;     //< GPU buffer object that stores individual draw commands
			std::shared_ptr<BufferObject>	obj_shader_params; //< GPU buffer object that stores per object data, i.e. objects transform
			std::shared_ptr<BufferObject>	mtl_shader_params; //< GPU buffer object that stores per material data, i.e. texture handles
		};

		std::vector<BatchedMeshes>               m_meshes;
		std::vector<std::shared_ptr<GLSLShader>> m_shader_programs;
		std::vector<Material>                    m_materials;
		std::vector<RenderBatch>                 m_render_batches; //< List of render batches ready for dispatching
		std::unique_ptr<BufferObject>            per_frame_data; //< GPU buffer object that stores per frame data, i.e. camera parameters

		/** Render batches caller slot */
		//megamol::core::CallerSlot m_renderBatches_callerSlot;

		megamol::core::CallerSlot m_mesh_callerSlot;
		megamol::core::CallerSlot m_material_callerSlot;
		megamol::core::CallerSlot m_render_task_callerSlot;


		std::shared_ptr<Mesh> createMesh(BatchedMeshesDataAccessor const & meshes, size_t mesh_idx);
	};

}
}

#endif