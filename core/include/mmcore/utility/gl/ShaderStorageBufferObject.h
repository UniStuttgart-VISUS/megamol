#ifndef fetol_shaderStorageBufferObject_h
#define fetol_shaderStorageBufferObject_h

#include "vislib/graphics/gl/IncludeAllGL.h"

/*	std includes */
#include <iostream>

/**
 * \class ShaderStorageBufferObject
 *
 * \brief Encapsulates SSBO functionality. Consider using the more generic BufferObject class.
 *
 * \author Michael Becher
 */
class ShaderStorageBufferObject {
public:
    ShaderStorageBufferObject(unsigned int size, const GLvoid* data);

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

    ~ShaderStorageBufferObject();

    /* Deleted copy constructor (C++11). No going around deleting copies of OpenGL Object with identical handles! */
    ShaderStorageBufferObject(const ShaderStorageBufferObject& cpy) = delete;
    ShaderStorageBufferObject(ShaderStorageBufferObject&& other) = delete;
    ShaderStorageBufferObject& operator=(ShaderStorageBufferObject&& rhs) = delete;
    ShaderStorageBufferObject& operator=(const ShaderStorageBufferObject& rhs) = delete;

    void reload(unsigned int size, GLuint index, const GLvoid* data);

    template <typename Container> void reload(const Container& datastorage) {
        m_size = static_cast<unsigned int>(datastorage.size() * sizeof(Container::value_type));

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_handle);
        glBufferData(GL_SHADER_STORAGE_BUFFER, m_size, datastorage.data(), GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        auto err = glGetError();
        if (err != GL_NO_ERROR) {
            std::cerr << "Error - SSBO - reload: " << err << std::endl;
        }
    }

    template <typename Container> void loadSubdata(const Container& datastorage, GLuint offset = 0) {
        // check if feasible
        if ((offset + datastorage.size() * sizeof(Container::value_type)) > m_size) {
            // TODO error message
            return;
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_handle);
        glBufferSubData(
            GL_SHADER_STORAGE_BUFFER, offset, datastorage.size() * sizeof(Container::value_type), datastorage.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    /**
    /	Binds SSBO to shader storage buffer target
    */
    void bind();

    /**
    /	Binds SSBO to an indexed shader storage buffer target
    */
    void bind(GLuint index);

    static void copy(ShaderStorageBufferObject* src, ShaderStorageBufferObject* tgt) {
        if (src->m_size > tgt->m_size) {
            std::cerr << "Error: ShaderStorageBufferObject::copy - target buffer smaller than source." << std::endl;
            return;
        }

        glBindBuffer(GL_COPY_READ_BUFFER, src->m_handle);
        glBindBuffer(GL_COPY_WRITE_BUFFER, tgt->m_handle);

        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, src->m_size);

        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    }

    GLuint getSize();

private:
    /**	OpenGL handle/id of the buffer object */
    GLuint m_handle;

    /**	Overall size of the buffer */
    GLuint m_size;

    /**
    /	Size of the data that has actually been written to the buffer.
    /	Note that this has to be set manually (usually from an atomic integer) after
    /	usage of the buffer!
    */
    GLuint m_written_size;
};

#endif // !ShaderStorageBufferObject_hpp
