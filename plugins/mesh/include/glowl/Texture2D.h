/*
* Texture2D.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef TEXTURE2D_H_INCLUDED
#define TEXTURE2D_H_INCLUDED

#include <algorithm>
#include <cmath>

#include "Texture.h"

namespace megamol {
	namespace mesh {


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


	}
}

#endif // !TEXTURE2D_H_INCLUDED
