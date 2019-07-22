/*
 * Texture2DArray.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/utility/gl/Texture2DArray.h"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace megamol::core::utility::gl;

/*
 * Texture2DArray::Texture2DArray
 */
Texture2DArray::Texture2DArray(std::string id, TextureLayout const& layout, GLvoid* data, bool generateMipmap)
    : Texture(id, layout.internal_format, layout.format, layout.type, layout.levels)
    , m_width(layout.width)
    , m_height(layout.height)
    , m_layers(layout.depth) {
    glGenTextures(1, &m_name);

    glBindTexture(GL_TEXTURE_2D_ARRAY, m_name);

    for (auto& pname_pvalue : layout.int_parameters)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, pname_pvalue.first, pname_pvalue.second);

    for (auto& pname_pvalue : layout.float_parameters)
        glTexParameterf(GL_TEXTURE_2D_ARRAY, pname_pvalue.first, pname_pvalue.second);

    GLsizei levels = 1;

    levels = std::min(layout.levels, 1 + static_cast<GLsizei>(std::floor(std::log2(std::max(m_width, m_height)))));

    glTexStorage3D(GL_TEXTURE_2D_ARRAY, levels, m_internal_format, m_width, m_height, m_layers);

    if (data != nullptr)
        glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, m_width, m_height, m_layers, m_format, m_type, data);

    if (generateMipmap) glGenerateMipmap(GL_TEXTURE_2D_ARRAY);

    m_texture_handle = glGetTextureHandleARB(m_name);

    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        // TODO proper error handling
    }
}

/*
 * Texture2DArray::bindTexture
 */
void Texture2DArray::bindTexture() const { glBindTexture(GL_TEXTURE_2D_ARRAY, m_name); }

/*
 * Texture2DArray::updateMipmaps
 */
void Texture2DArray::updateMipmaps() {
    glBindTexture(GL_TEXTURE_2D_ARRAY, m_name);
    glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
}

/*
 * Texture2DArray::getTextureLayout
 */
TextureLayout Texture2DArray::getTextureLayout() const {
    return TextureLayout(m_internal_format, m_width, m_height, m_layers, m_format, m_type, m_levels);
}

/*
 * Texture2DArray::getWidth
 */
unsigned int Texture2DArray::getWidth() const { return m_width; }

/*
 * Texture2DArray::getHeigth
 */
unsigned int Texture2DArray::getHeigth() const { return m_height; }

/*
 * Texture2DArray::getLayers
 */
unsigned int Texture2DArray::getLayers() const { return m_layers; }
