/*
 * TextureCubemapArray.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/utility/gl/TextureCubemapArray.h"

#include <cassert>

using namespace megamol::core::utility::gl;

TextureCubemapArray::TextureCubemapArray(std::string id, GLint internal_format, unsigned int width, unsigned int height,
    unsigned int layers, GLenum format, GLenum type, GLsizei levels, GLvoid* data, bool generateMipmap)
    : Texture(id, internal_format, format, type, levels), m_width(width), m_height(height), m_layers(layers) {
    glGenTextures(1, &m_name);
    assert(m_name > 0);

    glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, m_name);

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    glTexStorage3D(GL_TEXTURE_CUBE_MAP_ARRAY, 1, m_internal_format, m_width, m_height, m_layers);

    if (data != nullptr)
        glTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, 0, 0, 0, 0, m_width, m_height, m_layers, m_format, m_type, data);

    if (generateMipmap) {
        glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP_ARRAY);
    }

    glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, 0);

    if (glGetError() == GL_NO_ERROR) {
        // "Do something cop!"
    }
}

bool TextureCubemapArray::reload(
    unsigned int width, unsigned int height, unsigned int layers, GLvoid* data, bool generateMipmap) {
    m_width = width;
    m_height = height;
    m_layers = layers;

    glDeleteTextures(1, &m_name);

    glGenTextures(1, &m_name);
    assert(m_name > 0);

    glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, m_name);

    glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    glTexStorage3D(GL_TEXTURE_CUBE_MAP_ARRAY, 1, m_internal_format, m_width, m_height, m_layers);

    if (data != nullptr)
        glTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, 0, 0, 0, 0, m_width, m_height, m_layers, m_format, m_type, data);

    if (generateMipmap) {
        glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP_ARRAY);
    }

    glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, 0);

    if (glGetError() == GL_NO_ERROR) {
        // "Do something cop!"
        return true;
    } else {
        return false;
    }
}

void TextureCubemapArray::bindTexture() const { glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, m_name); }

void TextureCubemapArray::updateMipmaps() {
    glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, m_name);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP_ARRAY);
    glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, 0);
}

void TextureCubemapArray::texParameteri(GLenum pname, GLenum param) {
    glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, m_name);
    glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, pname, param);
    glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, 0);
}

TextureLayout TextureCubemapArray::getTextureLayout() const {
    return TextureLayout(m_internal_format, m_width, m_height, m_layers, m_format, m_type, m_levels);
}

unsigned int TextureCubemapArray::getWidth() const { return m_width; }

unsigned int TextureCubemapArray::getHeight() const { return m_height; }

unsigned int TextureCubemapArray::getLayers() const { return m_layers; }
