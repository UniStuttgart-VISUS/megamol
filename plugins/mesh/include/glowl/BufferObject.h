/*
 * Buffer.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef BUFFER_OBJECT_H_INCLUDED
#define BUFFER_OBJECT_H_INCLUDED

#include <memory>

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/sys/Log.h"

namespace megamol {
namespace mesh {

/**
 * Generic OpenGL buffer object.
 */
class BufferObject {
public:
    // BufferObject likely to be allocated into unique_ptr. Make usage less verbose.
    typedef std::unique_ptr<BufferObject> Ptr;

    template <typename Container>
    BufferObject(GLenum target, Container const& datastorage, GLenum usage = GL_DYNAMIC_DRAW)
        : m_target(target)
        , m_handle(0)
        , m_byte_size(static_cast<GLsizeiptr>(datastorage.size() * sizeof(Container::value_type)))
        , m_usage(usage) {
        glGenBuffers(1, &m_handle);
        glBindBuffer(m_target, m_handle);
        glBufferData(m_target, m_byte_size, datastorage.data(), m_usage);
        glBindBuffer(m_target, 0);
    }

    BufferObject(GLenum target, GLvoid const* data, GLsizeiptr byte_size, GLenum usage = GL_DYNAMIC_DRAW)
        : m_target(target), m_handle(0), m_byte_size(byte_size), m_usage(usage) {
        glGenBuffers(1, &m_handle);
        glBindBuffer(m_target, m_handle);
        glBufferData(m_target, m_byte_size, data, m_usage);
        glBindBuffer(m_target, 0);
    }

    ~BufferObject() { glDeleteBuffers(1, &m_handle); }

    BufferObject(const BufferObject& cpy) = delete;
    BufferObject(BufferObject&& other) = delete;
    BufferObject& operator=(BufferObject&& rhs) = delete;
    BufferObject& operator=(const BufferObject& rhs) = delete;

    template <typename Container> void loadSubData(Container const& datastorage, GLsizeiptr byte_offset = 0) const {
        // check if feasible
        if ((byte_offset + static_cast<GLsizeiptr>(datastorage.size() * sizeof(typename Container::value_type))) > m_byte_size) {
            // error message
            vislib::sys::Log::DefaultLog.WriteError("Invalid byte_offset or size for loadSubData");
            return;
        }

        glBindBuffer(m_target, m_handle);
        glBufferSubData(m_target, byte_offset, datastorage.size() * sizeof(typename Container::value_type), datastorage.data());
        glBindBuffer(m_target, 0);
    }

    void loadSubData(GLvoid const* data, GLsizeiptr byte_size, GLsizeiptr byte_offset = 0) const {
        // check if feasible
        if ((byte_offset + byte_size) > m_byte_size) {
            // error message
            vislib::sys::Log::DefaultLog.WriteError("Invalid byte_offset or size for loadSubData");
            return;
        }

        glBindBuffer(m_target, m_handle);
        glBufferSubData(m_target, byte_offset, byte_size, data);
        glBindBuffer(m_target, 0);
    }

    void bind() const { glBindBuffer(m_target, m_handle); }

    void bind(GLuint index) const { glBindBufferBase(m_target, index, m_handle); }

    void bindAs(GLenum target, GLuint index) const { glBindBufferBase(target, index, m_handle); }

    GLenum getTarget() const { return m_target; }

    GLsizeiptr getByteSize() const { return m_byte_size; }

    static void copy(BufferObject* src, BufferObject* tgt) {
        if (src->m_byte_size > tgt->m_byte_size) {
            // std::cerr << "Error: ShaderStorageBufferObject::copy - target buffer smaller than source." << std::endl;
            return;
        }

        glBindBuffer(GL_COPY_READ_BUFFER, src->m_handle);
        glBindBuffer(GL_COPY_WRITE_BUFFER, tgt->m_handle);

        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, src->m_byte_size);

        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    }

    static void copy(BufferObject* src, BufferObject* tgt, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size) {
        if ((readOffset + size) > src->m_byte_size) {
            // std::cerr << "Error: ShaderStorageBufferObject::copy - target buffer smaller than source." <<
            // std::endl;
            return;
        } else if ((writeOffset + size) > tgt->m_byte_size) {
            return;
        }

        glBindBuffer(GL_COPY_READ_BUFFER, src->m_handle);
        glBindBuffer(GL_COPY_WRITE_BUFFER, tgt->m_handle);

        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, readOffset, writeOffset, size);

        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    }

private:
    GLenum m_target;
    GLuint m_handle;
    GLsizeiptr m_byte_size;
    GLenum m_usage;
};

} // namespace mesh
} // namespace megamol

#endif // !BUFFER_H_INCLUDED