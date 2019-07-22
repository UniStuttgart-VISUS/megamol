/*
 * ShaderStorageBufferObject.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCORE_SHADERSTORAGEBUFFEROBJECT_H_INCLUDED
#define MEGAMOLCORE_SHADERSTORAGEBUFFEROBJECT_H_INCLUDED

#include "vislib/graphics/gl/IncludeAllGL.h"

namespace megamol {
namespace core {
namespace utility {
namespace gl {

/**
 * @class ShaderStorageBufferObject
 *
 * @brief Encapsulates SSBO functionality. Consider using the more generic BufferObject class.
 *
 * @author Michael Becher
 */
class ShaderStorageBufferObject {
public:
    /**
     * Constructor
     *
     * @param size Size of the SSBO in bytes
     * @param data Pointer to the data that should be stored into the SSBO
     */
    ShaderStorageBufferObject(unsigned int size, const GLvoid* data);

    /**
     * Constructor
     *
     * @param datastorage Container of the data that should be copied to the new SSBO
     */
    template <typename Container>
    ShaderStorageBufferObject(const Container& datastorage)
        : m_handle(0)
        , m_size(static_cast<GLuint>(datastorage.size() * sizeof(Container::value_type)))
        , m_written_size(0) {
        /* make clang++ compiler 'unused variable' warning go away */
        if (0 && m_written_size) {
        }

        glGenBuffers(1, &m_handle);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_handle);
        glBufferData(GL_SHADER_STORAGE_BUFFER, m_size, datastorage.data(), GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    /**
     * Destructor
     */
    virtual ~ShaderStorageBufferObject();

    /* Deleted copy constructor (C++11). No going around deleting copies of OpenGL Object with identical handles! */
    ShaderStorageBufferObject(const ShaderStorageBufferObject& cpy) = delete;

    /* Deleted move constructor (C++11). No going around deleting copies of OpenGL Object with identical handles! */
    ShaderStorageBufferObject(ShaderStorageBufferObject&& other) = delete;

    /* Deleted move operator (C++11). No going around deleting copies of OpenGL Object with identical handles! */
    ShaderStorageBufferObject& operator=(ShaderStorageBufferObject&& rhs) = delete;

    /* Deleted assignment operator (C++11). No going around deleting copies of OpenGL Object with identical handles! */
    ShaderStorageBufferObject& operator=(const ShaderStorageBufferObject& rhs) = delete;

    /**
     * Reloads the data in the ssbo
     *
     * @param size The size of the new data in bytes
     * @param data Pointer to the new data
     * @return True on success, false otherwise
     */
    bool reload(unsigned int size, const GLvoid* data);

    /**
     * Reloads the data in the SSBO
     *
     * @param datastorage Container storing the new data
     * @return True on success, false otherwise
     */
    template <typename Container> bool reload(const Container& datastorage) {
        m_size = static_cast<unsigned int>(datastorage.size() * sizeof(Container::value_type));

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_handle);
        glBufferData(GL_SHADER_STORAGE_BUFFER, m_size, datastorage.data(), GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        auto err = glGetError();
        return (err == GL_NO_ERROR);
    }

    /**
     * Changes the data in a part of the SSBO
     *
     * @param datastorage Container storing the new data
     * @param offset Offset of the target location from the start of the SSBO in bytes
     * @return True on success, false otherwise
     */
    template <typename Container> bool loadSubdata(const Container& datastorage, GLuint offset = 0) {
        // check if feasible
        if ((offset + datastorage.size() * sizeof(Container::value_type)) > m_size) {
            return false;
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_handle);
        glBufferSubData(
            GL_SHADER_STORAGE_BUFFER, offset, datastorage.size() * sizeof(Container::value_type), datastorage.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        return true;
    }

    /**
     * Binds SSBO to shader storage buffer target
     */
    void bind();

    /**
     * Binds SSBO to an indexed shader storage buffer target
     *
     * @param index The index of the buffer target
     */
    void bind(GLuint index);

    /**
     * Copies the data from one SSBO into another
     *
     * @param src The source buffer
     * @param tgt The buffer the data gets copied to
     */
    static bool copy(ShaderStorageBufferObject* src, ShaderStorageBufferObject* tgt) {
        if (src->m_size > tgt->m_size) {
            return false;
        }

        glBindBuffer(GL_COPY_READ_BUFFER, src->m_handle);
        glBindBuffer(GL_COPY_WRITE_BUFFER, tgt->m_handle);

        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, src->m_size);

        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
        return true;
    }

    /**
     * Get the size of the SSBO in bytes
     *
     * @return Size of the SSBO in bytes
     */
    GLuint getSize();

private:
    /**	OpenGL handle/id of the buffer object */
    GLuint m_handle;

    /**	Overall size of the buffer */
    GLuint m_size;

    /**
     *	Size of the data that has actually been written to the buffer.
     *	Note that this has to be set manually (usually from an atomic integer) after
     *	usage of the buffer!
     */
    GLuint m_written_size;
};

} // namespace gl
} // namespace utility
} // namespace core
} // namespace megamol

#endif // !MEGAMOLCORE_SHADERSTORAGEBUFFEROBJECT_H_INCLUDED
