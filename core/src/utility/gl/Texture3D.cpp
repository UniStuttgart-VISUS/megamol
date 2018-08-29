#include "stdafx.h"
#include "mmcore/utility/gl/Texture3D.h"

#include <cassert>
#include <iostream>

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
        // "Do something cop!"
        std::cerr << "GL error during 3D texture (id:" << id << ") creation: " << err << std::endl;
    }
}

void Texture3D::bindTexture() const { glBindTexture(GL_TEXTURE_3D, m_name); }

void Texture3D::updateMipmaps() {
    glBindTexture(GL_TEXTURE_3D, m_name);
    glGenerateMipmap(GL_TEXTURE_3D);
    glBindTexture(GL_TEXTURE_3D, 0);
}

void Texture3D::reload(TextureLayout const& layout, GLvoid* data) {
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
    if (err != GL_NO_ERROR) {
        // "Do something cop!"
        std::cerr << "GL error during texture reloading: " << err << std::endl;
    }
}

TextureLayout Texture3D::getTextureLayout() const {
    return TextureLayout(m_internal_format, m_width, m_height, m_depth, m_format, m_type, m_levels);
}

unsigned int Texture3D::getWidth() { return m_width; }

unsigned int Texture3D::getHeight() { return m_height; }

unsigned int Texture3D::getDepth() { return m_depth; }
