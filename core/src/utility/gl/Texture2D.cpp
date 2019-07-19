/*
 * Texture2D.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/utility/gl/Texture2D.h"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace megamol::core::utility::gl;

/*
 * Texture2D::Texture2D
 */
Texture2D::Texture2D(std::string id, TextureLayout const& layout, GLvoid* data, bool generateMipmap)
    : Texture(id, layout.internal_format, layout.format, layout.type, layout.levels)
    , m_width(layout.width)
    , m_height(layout.height) {
    glGenTextures(1, &m_name);

    glBindTexture(GL_TEXTURE_2D, m_name);

    for (auto& pname_pvalue : layout.int_parameters)
        glTexParameteri(GL_TEXTURE_2D, pname_pvalue.first, pname_pvalue.second);

    // for (auto& pname_pvalue : layout.float_parameters)
    //	glTexParameterf(GL_TEXTURE_2D, pname_pvalue.first, pname_pvalue.second);

    GLsizei levels = 1;

    if (generateMipmap) levels = static_cast<GLsizei>(1 + std::floor(std::log2(std::max(m_width, m_height))));

    glTexStorage2D(GL_TEXTURE_2D, levels, m_internal_format, m_width, m_height);

    if (data != nullptr) glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, m_format, m_type, data);

    if (generateMipmap) glGenerateMipmap(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, 0);

    m_texture_handle = glGetTextureHandleARB(m_name);

    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        // TODO proper error handling
    }
}

/*
 * Texture2D::bindTexture
 */
void Texture2D::bindTexture() const { glBindTexture(GL_TEXTURE_2D, m_name); }

/*
 * Texture2D::updateMipmaps
 */
void Texture2D::updateMipmaps() {
    glBindTexture(GL_TEXTURE_2D, m_name);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
}

/*
 * Texture2D::reload
 */
bool Texture2D::reload(TextureLayout const& layout, GLvoid* data, bool generateMipmap) {
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

    if (generateMipmap) levels = static_cast<GLsizei>(1 + std::floor(std::log2(std::max(m_width, m_height))));

    glTexStorage2D(GL_TEXTURE_2D, levels, m_internal_format, m_width, m_height);

    if (data != nullptr) glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, m_format, m_type, data);

    if (generateMipmap) glGenerateMipmap(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, 0);

    GLenum err = glGetError();
    return (err == GL_NO_ERROR);
}

/*
 * Texture2D::getTextureLayout
 */
TextureLayout Texture2D::getTextureLayout() const {
    return TextureLayout(m_internal_format, m_width, m_height, 1, m_format, m_type, m_levels);
}

/*
 * Texture2D::getWidth
 */
unsigned int Texture2D::getWidth() const { return m_width; }

/*
 * Texture2D::getHeight
 */
unsigned int Texture2D::getHeight() const { return m_height; }
