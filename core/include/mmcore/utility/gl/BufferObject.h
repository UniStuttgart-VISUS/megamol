/*
 * BufferObject.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCORE_BUFFEROBJECT_H_INCLUDED
#define MEGAMOLCORE_BUFFEROBJECT_H_INCLUDED

#include "vislib/graphics/gl/IncludeAllGL.h"

namespace megamol {
namespace core {
namespace utility {
namespace gl {

/**
 * @class BufferObject
 *
 * @brief Generic OpenGL buffer object.
 *
 * @author Michael Becher
 */
class BufferObject {
public:
    /**
     * Constructor
     *
     * @param target Target definition of the buffer
     * @param datastorage Container holding the data to copy. This only works guaranteed with std::vector and std::array
     * @param usage Usage flag for the buffer
     */
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

    /**
     * Constructor
     *
     * @param target Target definition of the buffer
     * @param data Pointer to the data to copy to the buffer
     * @param byte_size Size of the buffer in bytes
     * @param usage Usage flag for the buffer
     */
    BufferObject(GLenum target, GLvoid const* data, GLsizeiptr byte_size, GLenum usage = GL_DYNAMIC_DRAW)
        : m_target(target), m_handle(0), m_byte_size(byte_size), m_usage(usage) {
        glGenBuffers(1, &m_handle);
        glBindBuffer(m_target, m_handle);
        glBufferData(m_target, m_byte_size, data, m_usage);
        glBindBuffer(m_target, 0);
    }

    /**
     * Destructor
     */
    virtual ~BufferObject() { glDeleteBuffers(1, &m_handle); }

    /* Deleted copy constructor (C++11). Don't wanna go around copying objects with OpenGL handles. */
    BufferObject(const BufferObject& cpy) = delete;

    /* Deleted move constructor (C++11). Don't wanna go around copying objects with OpenGL handles. */
    BufferObject(BufferObject&& other) = delete;

    /* Deleted move assignment (C++11). Don't wanna go around copying objects with OpenGL handles. */
    BufferObject& operator=(BufferObject&& rhs) = delete;

    /* Deleted assignment operator (C++11). Don't wanna go around copying objects with OpenGL handles. */
    BufferObject& operator=(const BufferObject& rhs) = delete;

    /**
     * Uploads data to a certain part of the buffer
     *
     * @param datastorage Container holding the data. This only works guaranteed with std::vector and std::array
     * @param byte_offset Offset of the data target from the buffer start in bytes
     * @return True on success, false otherwise
     */
    template <typename Container> bool bufferSubData(Container const& datastorage, GLsizeiptr byte_offset = 0) const {
        // check if feasible
        if ((byte_offset + static_cast<GLsizeiptr>(datastorage.size() * sizeof(Container::value_type))) > m_byte_size) {
            return false;
        }

        glBindBuffer(m_target, m_handle);
        glBufferSubData(m_target, byte_offset, datastorage.size() * sizeof(Container::value_type), datastorage.data());
        glBindBuffer(m_target, 0);
        return true;
    }

    /**
     * Uploads data to a certain part of the buffer
     *
     * @param data Pointer to the uploadable data
     * @param byte_size Size of the data in bytes
     * @param byte_offset Offset of the data target from the buffer start in bytes
     * @return True on success, false otherwise
     */
    bool bufferSubData(GLvoid const* data, GLsizeiptr byte_size, GLsizeiptr byte_offset = 0) const {
        // check if feasible
        if ((byte_offset + byte_size) > m_byte_size) {
            return false;
        }

        glBindBuffer(m_target, m_handle);
        glBufferSubData(m_target, byte_offset, byte_size, data);
        glBindBuffer(m_target, 0);
        return true;
    }

    /**
     * Uploads data to the buffer
     *
     * @param datastorage Container holding the data. This only works guaranteed with std::vector and std::array
     * @return True on success, false otherwise
     */
    template <typename Container> bool rebuffer(Container const& datastorage) {
        m_byte_size = static_cast<GLsizeiptr>(datastorage.size() * sizeof(Container::value_type));
        glBindBuffer(m_target, m_handle);
        glBufferData(m_target, m_byte_size, datastorage.data(), m_usage);
        glBindBuffer(m_target, 0);

        auto err = glGetError();
        return (err == GL_NO_ERROR);
    }

    /**
     * Uploads data to the buffer
     *
     * @param data Pointer to the data
     * @param byte_size Size of the buffer in bytes
     * @return True on success, false otherwise
     */
    bool rebuffer(GLvoid const* data, GLsizeiptr byte_size) {
        m_byte_size = byte_size;
        glBindBuffer(m_target, m_handle);
        glBufferData(m_target, m_byte_size, data, m_usage);
        glBindBuffer(m_target, 0);

        auto err = glGetError();
        return (err == GL_NO_ERROR);
    }

    /**
     * Binds the buffer as current target
     */
    void bind() const { glBindBuffer(m_target, m_handle); }

    /**
     * Binds the buffer as current target to a given index
     *
     * @param index The index to bind to
     */
    void bind(GLuint index) const { glBindBufferBase(m_target, index, m_handle); }

    /**
     * Binds the buffer to a new target with a given index
     *
     * @param target The new buffer target
     * @param index The index to bind to
     * @return True on success, false otherwise
     */
    bool bindAs(GLenum target, GLuint index) const {
        glBindBufferBase(target, index, m_handle);
        auto err = glGetError();
        return (err == GL_NO_ERROR);
    }

    /**
     * Copies the content of one buffer object into another
     *
     * @param src The source buffer for copy operation
     * @param tgt The target buffer for the copy operation
     * @return True on success, false otherwise
     */
    static bool copy(BufferObject* src, BufferObject* tgt) {
        if (src->m_byte_size > tgt->m_byte_size) {
            // std::cerr << "Error: ShaderStorageBufferObject::copy - target buffer smaller than source." << std::endl;
            // return;
            return false;
        }
        glBindBuffer(GL_COPY_READ_BUFFER, src->m_handle);
        glBindBuffer(GL_COPY_WRITE_BUFFER, tgt->m_handle);

        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, src->m_byte_size);

        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
        return true;
    }

    /**
     * Returns the target of the buffer object
     *
     * @return The target of the buffer object
     */
    GLenum getTarget() const { return m_target; }

    /**
     * Returns the size of the contained buffer in bytes.
     *
     * @return The size of the buffer in bytes.
     */
    GLsizeiptr getByteSize() const { return m_byte_size; }

private:
    /** The target of the buffer object */
    GLenum m_target;

    /** The handle of the buffer object */
    GLuint m_handle;

    /** The size of the buffer object in bytes */
    GLsizeiptr m_byte_size;

    /** The usage of the buffer object */
    GLenum m_usage;
};

} // namespace gl
} // namespace utility
} // namespace core
} // namespace megamol

#endif // !MEGAMOLCORE_BUFFEROBJECT_H_INCLUDED
