/*
* RaycastVolumeRenderer.h
*
* Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef RAYCAST_VOLUME_RENDERER_H_INCLUDED
#define RAYCAST_VOLUME_RENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/graphics/gl/GLSLComputeShader.h"
#include "vislib/graphics/gl/OpenGLTexture2D.h"

#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol {
namespace stdplugin {
namespace volume {

	class RaycastVolumeRenderer : public megamol::core::view::Renderer3DModule
	{
	public:
		/**
		* Answer the name of this module.
		*
		* @return The name of this module.
		*/
		static const char *ClassName(void) {
			return "RaycastVolumeRenderer";
		}

		/**
		* Answer a human readable description of this module.
		*
		* @return A human readable description of this module.
		*/
		static const char *Description(void) {
			return "Modern compute-based raycast renderer for volumetric datasets.";
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
				&& ogl_IsVersionGEQ(4, 3);
		}

		RaycastVolumeRenderer();
		~RaycastVolumeRenderer();

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

		/**
		* The render callback.
		*
		* @param call The calling call.
		*
		* @return The return value of the function.
		*/
		bool Render(core::Call& call);

		bool updateVolumeData();

		bool updateTransferFunction();

	private:

		/* OpenGL TextureLayout, Texture and Texture2D classes courtesy of glOwl by Michael Becher. */

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

		class Texture3D : public Texture
		{
		public:
			Texture3D(std::string id, TextureLayout const& layout, GLvoid * data)
				: Texture(id, layout.internal_format, layout.format, layout.type, layout.levels), m_width(layout.width), m_height(layout.height), m_depth(layout.depth)
			{
				glGenTextures(1, &m_name);

				glBindTexture(GL_TEXTURE_3D, m_name);

				for (auto& pname_pvalue : layout.int_parameters)
					glTexParameteri(GL_TEXTURE_3D, pname_pvalue.first, pname_pvalue.second);

				for (auto& pname_pvalue : layout.float_parameters)
					glTexParameterf(GL_TEXTURE_3D, pname_pvalue.first, pname_pvalue.second);

				glTexStorage3D(GL_TEXTURE_3D, 1, m_internal_format, m_width, m_height, m_depth);

				if (data != nullptr)
					glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, m_width, m_height, m_depth, m_format, m_type, data);

				glBindTexture(GL_TEXTURE_3D, 0);

				m_texture_handle = glGetTextureHandleARB(m_name);

				GLenum err = glGetError();
				if (err != GL_NO_ERROR)
				{
					// "Do something cop!"
					//std::cerr << "GL error during 3D texture (id:" << id << ") creation: " << err << std::endl;
				}
			}

			Texture3D(const Texture3D&) = delete;
			Texture3D(Texture3D&& other) = delete;
			Texture3D& operator=(const Texture3D& rhs) = delete;
			Texture3D& operator=(Texture3D&& rhs) = delete;

			/**
			* \brief Bind the texture.
			*/
			void bindTexture() const
			{
				glBindTexture(GL_TEXTURE_3D, m_name);
			}

			void updateMipmaps()
			{
				glBindTexture(GL_TEXTURE_3D, m_name);
				glGenerateMipmap(GL_TEXTURE_3D);
				glBindTexture(GL_TEXTURE_3D, 0);
			}

			/**
			* \brief Reload the texture.
			* \param data Pointer to the new texture data.
			*/
			void reload(TextureLayout const& layout, GLvoid const * data)
			{
				m_width = layout.width;
				m_height = layout.height;
				m_depth = layout.depth;
				m_internal_format = layout.internal_format;
				m_format = layout.format;
				m_type = layout.type;

				glDeleteTextures(1, &m_name);

				glGenTextures(1, &m_name);

				glBindTexture(GL_TEXTURE_3D, m_name);

				for (auto& pname_pvalue : layout.int_parameters)
					glTexParameteri(GL_TEXTURE_3D, pname_pvalue.first, pname_pvalue.second);

				for (auto& pname_pvalue : layout.float_parameters)
					glTexParameterf(GL_TEXTURE_3D, pname_pvalue.first, pname_pvalue.second);

				glTexStorage3D(GL_TEXTURE_3D, 1, m_internal_format, m_width, m_height, m_depth);

				if (data != nullptr)
					glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, m_width, m_height, m_depth, m_format, m_type, data);

				glBindTexture(GL_TEXTURE_3D, 0);

				GLenum err = glGetError();
				if (err != GL_NO_ERROR)
				{
					// "Do something cop!"
					//std::cerr << "GL error during texture reloading: " << err << std::endl;
				}
			}

			TextureLayout getTextureLayout() const
			{
				return TextureLayout(m_internal_format, m_width, m_height, m_depth, m_format, m_type, m_levels);
			}

			unsigned int getWidth()
			{
				return m_width;
			}
			unsigned int getHeight()
			{
				return m_height;
			}
			unsigned int getDepth()
			{
				return m_depth;
			}

		private:
			unsigned int m_width;
			unsigned int m_height;
			unsigned int m_depth;
		};

		std::unique_ptr<vislib::graphics::gl::GLSLComputeShader> m_raycast_volume_compute_shdr;
		std::unique_ptr<vislib::graphics::gl::GLSLShader> m_render_to_framebuffer_shdr;

		std::unique_ptr<Texture2D> m_render_target;

		std::unique_ptr<Texture3D> m_volume_texture;

		std::unique_ptr<Texture2D> m_transfer_function;

		size_t m_volume_datahash;
		float m_volume_origin[3];
		float m_volume_extents[3];
        float m_volume_resolution[3];

		/** caller slot */
		megamol::core::CallerSlot m_volumetricData_callerSlot;
		megamol::core::CallerSlot m_transferFunction_callerSlot;

		core::param::ParamSlot m_ray_step_ratio_param;

	};

}
}
}

#endif