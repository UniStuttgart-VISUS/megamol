#ifndef MEGAMOLCORE_BUFFEROBJECT_H_INCLUDED
#define MEGAMOLCORE_BUFFEROBJECT_H_INCLUDED

#include <iostream>

#include "vislib/graphics/gl/IncludeAllGL.h"

namespace megamol {
namespace core {
namespace utility {
namespace gl {

/**
 * \class BufferObject
 *
 * \brief Generic OpenGL buffer object.
 *
 * \author Michael Becher
 */
class BufferObject {
public:
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

    template <typename Container> void bufferSubData(Container const& datastorage, GLsizeiptr byte_offset = 0) const {
        // check if feasible
        if ((byte_offset + static_cast<GLsizeiptr>(datastorage.size() * sizeof(Container::value_type))) > m_byte_size) {
            // error message
            std::cerr << "Error - BufferObject - bufferSubData: given data too large for buffer." << std::endl;
            return;
        }

        glBindBuffer(m_target, m_handle);
        glBufferSubData(m_target, byte_offset, datastorage.size() * sizeof(Container::value_type), datastorage.data());
        glBindBuffer(m_target, 0);
    }

    void bufferSubData(GLvoid const* data, GLsizeiptr byte_size, GLsizeiptr byte_offset = 0) const {
        // check if feasible
        if ((byte_offset + byte_size) > m_byte_size) {
            // error message
            std::cerr << "Error - BufferObject - bufferSubData: given data too large for buffer." << std::endl;
            return;
        }

        glBindBuffer(m_target, m_handle);
        glBufferSubData(m_target, byte_offset, byte_size, data);
        glBindBuffer(m_target, 0);
    }

    template <typename Container> void rebuffer(Container const& datastorage) {
        m_byte_size = static_cast<GLsizeiptr>(datastorage.size() * sizeof(Container::value_type));
        glBindBuffer(m_target, m_handle);
        glBufferData(m_target, m_byte_size, datastorage.data(), m_usage);
        glBindBuffer(m_target, 0);

        auto err = glGetError();
        if (err != GL_NO_ERROR) {
            std::cerr << "Error - BufferObject - rebuffer: " << err << std::endl;
        }
    }

    void rebuffer(GLvoid const* data, GLsizeiptr byte_size) {
        m_byte_size = byte_size;
        glBindBuffer(m_target, m_handle);
        glBufferData(m_target, m_byte_size, data, m_usage);
        glBindBuffer(m_target, 0);

        auto err = glGetError();
        if (err != GL_NO_ERROR) {
            std::cerr << "Error - BufferObject - rebuffer: " << err << std::endl;
        }
    }

    void bind() const { glBindBuffer(m_target, m_handle); }

    void bind(GLuint index) const { glBindBufferBase(m_target, index, m_handle); }

    void bindAs(GLenum target, GLuint index) const {
        glBindBufferBase(target, index, m_handle);
        auto err = glGetError();
        if (err != GL_NO_ERROR) {
            std::cerr << "Error - BufferObject - rebindAs: " << err << std::endl;
        }
    }

    static void copy(BufferObject* src, BufferObject* tgt) {
        if (src->m_byte_size > tgt->m_byte_size) {
            std::cerr << "Error: ShaderStorageBufferObject::copy - target buffer smaller than source." << std::endl;
            return;
        }

        glBindBuffer(GL_COPY_READ_BUFFER, src->m_handle);
        glBindBuffer(GL_COPY_WRITE_BUFFER, tgt->m_handle);

        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, src->m_byte_size);

        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    }

    GLenum getTarget() const { return m_target; }

    GLsizeiptr getByteSize() const { return m_byte_size; }

private:
    GLenum m_target;
    GLuint m_handle;
    GLsizeiptr m_byte_size;
    GLenum m_usage;
};

} // namespace gl
} // namespace utility
} // namespace core
} // namespace megamol

#endif // !MEGAMOLCORE_BUFFEROBJECT_H_INCLUDED
