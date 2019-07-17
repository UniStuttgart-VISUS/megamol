/*
 * Mesh.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/utility/gl/Mesh.h"

using namespace megamol::core::utility::gl;

/*
 * Mesh::Mesh
 */
Mesh::Mesh(GLvoid const* vertex_data, GLsizeiptr vertex_data_byte_size, GLvoid const* index_data,
    GLsizeiptr index_data_byte_size, VertexLayout const& vertex_descriptor, GLenum indices_type, GLenum usage,
    GLenum primitive_type)
    : m_ibo(GL_ELEMENT_ARRAY_BUFFER, index_data, index_data_byte_size, usage)
    , m_vertex_descriptor(vertex_descriptor)
    , m_va_handle(0)
    , m_indices_cnt(0)
    , m_indices_type(indices_type)
    , m_usage(usage)
    , m_primitive_type(primitive_type) {
    m_vbos.emplace_back(std::make_unique<BufferObject>(GL_ARRAY_BUFFER, vertex_data, vertex_data_byte_size, usage));

    glGenVertexArrays(1, &m_va_handle);

    // set attribute pointer and vao state
    glBindVertexArray(m_va_handle);
    m_vbos.back()->bind();

    // dirty hack to make ibo work as BufferObject
    // m_ibo = std::make_unique<BufferObject>(GL_ELEMENT_ARRAY_BUFFER, index_data, index_data_byte_size, usage);
    m_ibo.bind();

    GLuint attrib_idx = 0;
    for (auto& attribute : vertex_descriptor.attributes) {
        glEnableVertexAttribArray(attrib_idx);
        glVertexAttribPointer(attrib_idx, attribute.size, attribute.type, attribute.normalized,
            vertex_descriptor.byte_size, reinterpret_cast<GLvoid*>(attribute.offset));

        attrib_idx++;
    }
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    switch (m_indices_type) {
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

/*
 * Mesh::Mesh
 */
Mesh::Mesh(std::vector<uint8_t*> const& vertex_data, std::vector<size_t> const& vertex_data_byte_sizes,
    GLvoid const* index_data, GLsizeiptr index_data_byte_size, VertexLayout const& vertex_descriptor,
    GLenum indices_type, GLenum usage, GLenum primitive_type)
    : m_ibo(GL_ELEMENT_ARRAY_BUFFER, index_data, index_data_byte_size, usage)
    , m_vertex_descriptor(vertex_descriptor)
    , m_va_handle(0)
    , m_indices_cnt(0)
    , m_indices_type(indices_type)
    , m_usage(usage)
    , m_primitive_type(primitive_type) {
    for (unsigned int i = 0; i < vertex_data.size(); ++i)
        m_vbos.emplace_back(
            std::make_unique<BufferObject>(GL_ARRAY_BUFFER, vertex_data[i], vertex_data_byte_sizes[i], usage));

    glGenVertexArrays(1, &m_va_handle);

    // set attribute pointer and vao state
    glBindVertexArray(m_va_handle);

    m_ibo.bind();

    // TODO check if vertex buffer count matches attribute count, throw exception if not?
    GLuint attrib_idx = 0;
    for (auto& attribute : vertex_descriptor.attributes) {
        m_vbos[attrib_idx]->bind();

        glEnableVertexAttribArray(attrib_idx);
        glVertexAttribPointer(attrib_idx, attribute.size, attribute.type, attribute.normalized,
            vertex_descriptor.byte_size, reinterpret_cast<GLvoid*>(attribute.offset));

        attrib_idx++;
    }
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    switch (m_indices_type) {
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
