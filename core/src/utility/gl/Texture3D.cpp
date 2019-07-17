/*
 * Texture3D.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/utility/gl/Texture3D.h"

#include <cassert>
#include <iostream>

using namespace megamol::core::utility::gl;

/*
 * Texture3D::Texture3D
 */
Texture3D::Texture3D(std::string id, TextureLayout const& layout, GLvoid* data)
    : Texture(id, layout.internal_format, layout.format, layout.type, layout.levels)
    , m_width(layout.width)
    , m_height(layout.height)
    , m_depth(layout.depth) {
    glGenTextures(1, &m_name);

    glBindTexture(GL_TEXTURE_3D, m_name);

    for (auto& pname_pvalue : layout.int_parameters)
        glTexParameteri(GL_TEXTURE_3D, pname_pvalue.first, pname_pvalue.second);

    for (auto& pname_pvalue : layout.float_parameters)
        glTexParameterf(GL_TEXTURE_3D, pname_pvalue.first, pname_pvalue.second);

    glTexStorage3D(GL_TEXTURE_3D, 1, m_internal_format, m_width, m_height, m_depth);

    if (data != nullptr) glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, m_width, m_height, m_depth, m_format, m_type, data);

    glBindTexture(GL_TEXTURE_3D, 0);

    m_texture_handle = glGetTextureHandleARB(m_name);

    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        // TODO proper error handling;
    }
}

/*
 * Texture3D::bindTexture
 */
void Texture3D::bindTexture() const { glBindTexture(GL_TEXTURE_3D, m_name); }

/*
 * Texture3D::updateMipmaps
 */
void Texture3D::updateMipmaps() {
    glBindTexture(GL_TEXTURE_3D, m_name);
    glGenerateMipmap(GL_TEXTURE_3D);
    glBindTexture(GL_TEXTURE_3D, 0);
}

/*
 * Texture3D::reload
 */
bool Texture3D::reload(TextureLayout const& layout, GLvoid* data) {
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

    if (data != nullptr) glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, m_width, m_height, m_depth, m_format, m_type, data);

    glBindTexture(GL_TEXTURE_3D, 0);

    GLenum err = glGetError();
    return (err == GL_NO_ERROR);
}

/*
 * Texture3D::getTextureLayout
 */
TextureLayout Texture3D::getTextureLayout() const {
    return TextureLayout(m_internal_format, m_width, m_height, m_depth, m_format, m_type, m_levels);
}

/*
 * Texture3D::getWidth
 */
unsigned int Texture3D::getWidth() { return m_width; }

/*
 * Texture3D::getHeight
 */
unsigned int Texture3D::getHeight() { return m_height; }

/*
 * Texture3D::getDepth
 */
unsigned int Texture3D::getDepth() { return m_depth; }
