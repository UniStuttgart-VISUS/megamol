/*
 * ShaderStorageBufferObject.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/utility/gl/ShaderStorageBufferObject.h"

using namespace megamol::core::utility::gl;

/*
 * ShaderStorageBufferObject::ShaderStorageBufferObject
 */
ShaderStorageBufferObject::ShaderStorageBufferObject(unsigned int size, const GLvoid* data)
    : m_handle(0), m_size(size), m_written_size(0) {
    /* make clang++ compiler 'unused variable' warning go away */
    if (0 && m_written_size) {
    }

    glGenBuffers(1, &m_handle);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_handle);
    glBufferData(GL_SHADER_STORAGE_BUFFER, size, data, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

/*
 * ShaderStorageBufferObject::~ShaderStorageBufferObject
 */
ShaderStorageBufferObject::~ShaderStorageBufferObject() { glDeleteBuffers(1, &m_handle); }

/*
 * ShaderStorageBufferObject::reload
 */
bool ShaderStorageBufferObject::reload(unsigned int size, const GLvoid* data) {
    m_size = size;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_handle);
    glBufferData(GL_SHADER_STORAGE_BUFFER, size, data, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    auto err = glGetError();
    return (err == GL_NO_ERROR);
}

/*
 * ShaderStorageBufferObject::bind
 */
void ShaderStorageBufferObject::bind() { glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_handle); }

/*
 * ShaderStorageBufferObject::bind
 */
void ShaderStorageBufferObject::bind(GLuint index) { glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, m_handle); }

/*
 * ShaderStorageBufferObject::getSize
 */
GLuint ShaderStorageBufferObject::getSize() { return m_size; }
