/*
 * Mesh.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCORE_MESH_H_INCLUDED
#define MEGAMOLCORE_MESH_H_INCLUDED

#include <memory>
#include <string>
#include <vector>
#include "vislib/graphics/gl/IncludeAllGL.h"

#include "mmcore/utility/gl/BufferObject.h"
#include "mmcore/utility/gl/VertexLayout.h"

namespace megamol {
namespace core {
namespace utility {
namespace gl {

/**
 * @class Mesh
 *
 * @brief Encapsulates mesh functionality.
 *
 * @author Michael Becher
 */
class Mesh {
public:
    /** The single buffer object holding the data for the mesh */
    typedef std::unique_ptr<BufferObject> BufferObjectPtr;

    /**
     * Constructor using container objects as input
     *
     * @param vertices Container storing the vertex coordinate data
     * @param indices Container storing the primitive index data
     * @param vertex_descriptor Descriptor vor the mesh layout
     * @param indices_type Data type of the indices
     * @param usage Usage of the meshes buffer data
     * @param primitive_type Type of the primitives that should be drawn
     */
    template <typename VertexContainer, typename IndexContainer>
    Mesh(VertexContainer const& vertices, IndexContainer const& indices, VertexLayout const& vertex_descriptor,
        GLenum indices_type = GL_UNSIGNED_INT, GLenum usage = GL_STATIC_DRAW, GLenum primitive_type = GL_TRIANGLES);

    /**
     * Constructor using container objects as input
     *
     * @param vertex_data Pointer to the vertex coordinate data
     * @aram vertex_data_byte_size Size of the vertex data in byte
     * @param index_data Pointer to the primitive index data
     * @param index_data_byte_size Size of the primitive index data in bytes
     * @param vertex_descriptor Descriptor vor the mesh layout
     * @param indices_type Data type of the indices
     * @param usage Usage of the meshes buffer data
     * @param primitive_type Type of the primitives that should be drawn
     */
    Mesh(GLvoid const* vertex_data, GLsizeiptr vertex_data_byte_size, GLvoid const* index_data,
        GLsizeiptr index_data_byte_size, VertexLayout const& vertex_descriptor, GLenum indices_type = GL_UNSIGNED_INT,
        GLenum usage = GL_STATIC_DRAW, GLenum primitive_type = GL_TRIANGLES);

    /**
     * Constructor for non-interleaved data (one vertex buffer object per attribute)
     *
     * @param vertex_data Vector storing the pointers to each attributes data storage
     * @param vertex_data_byte_sizes Vector containing the sizes of each attributes data in bytes
     * @param index_data Pointer to the primitive index data
     * @param index_data_byte_size Size of the primitive index data in bytes
     * @param vertex_descriptor Descriptor vor the mesh layout
     * @param indices_type Data type of the indices
     * @param usage Usage of the meshes buffer data
     * @param primitive_type Type of the primitives that should be drawn
     */
    Mesh(std::vector<uint8_t*> const& vertex_data, std::vector<size_t> const& vertex_data_byte_sizes,
        GLvoid const* index_data, GLsizeiptr index_data_byte_size, VertexLayout const& vertex_descriptor,
        GLenum indices_type = GL_UNSIGNED_INT, GLenum usage = GL_STATIC_DRAW, GLenum primitive_type = GL_TRIANGLES);

    /** Dtor */
    virtual ~Mesh() { glDeleteVertexArrays(1, &m_va_handle); }

    /** Deleted copy constructor */
    Mesh(const Mesh& cpy) = delete;

    /** Deleted move constructor */
    Mesh(Mesh&& other) = delete;

    /** Deleted move operator */
    Mesh& operator=(Mesh&& rhs) = delete;

    /** Delete assignment operator */
    Mesh& operator=(const Mesh& rhs) = delete;

    /**
     * Updates a part of the vertex data
     *
     * @param vbo_idx The index of the vertex buffer object to update
     * @param vertices Container containing the vertex data
     * @param byte_offset Offset from the start of the buffer in bytes
     */
    template <typename VertexContainer>
    void bufferVertexSubData(size_t vbo_idx, VertexContainer const& vertices, GLsizeiptr byte_offset);

    /**
     * Updates a part of the index data
     *
     * @param indices Container containing the index data
     * @param byte_offset Offset from the start of the buffer in bytes
     */
    template <typename IndexContainer> void bufferIndexSubData(IndexContainer const& indices, GLsizeiptr byte_offset);

    /**
     * Updates the data in the vertex buffer
     *
     * @param vbo_idx The index of the vertex buffer to update
     * @param vertices The container containing the new data
     */
    template <typename VertexContainer> bool rebufferVertexData(size_t vbo_idx, VertexContainer const& vertices);

    /**
     * Updates the data in the vertex buffer
     *
     * @param vbo_idx The index of the vertex buffer to update
     * @param vertex_data Pointer to the new data
     * @param vertex_data_byte_size Size of the new data in bytes
     * @return
     */
    bool rebufferVertexData(size_t vbo_idx, GLvoid const* vertex_data, GLsizeiptr vertex_data_byte_size) {
        if (vbo_idx < m_vbos.size()) {
            return m_vbos[vbo_idx]->rebuffer(vertex_data, vertex_data_byte_size);
        }
        return false;
    }

    /**
     * Updates the data in the vertex buffer
     *
     * @param vertices Container with the new vertex data
     * @param descriptor New vertex layout description
     */
    template <typename VertexContainer>
    bool rebufferVertexData(VertexContainer const& vertices, VertexLayout descriptor);

    /**
     *
     *
     * @param indices
     * @return
     */
    template <typename IndexContainer> bool rebufferIndexData(IndexContainer const& indices);

    /**
     * Updates the data in the index buffer
     *
     * @param index_data Pointer to the new index data
     * @param index_data_byte_sizes Size of the new data in bytes
     * @return
     */
    bool rebufferIndexData(GLvoid const* index_data, GLsizeiptr index_data_byte_size) {
        if (!m_ibo.rebuffer(index_data, index_data_byte_size)) return false;

        GLsizeiptr vi_size = index_data_byte_size;

        switch (m_indices_type) {
        case GL_UNSIGNED_INT:
            m_indices_cnt = static_cast<GLuint>(vi_size / 4);
            break;
        case GL_UNSIGNED_SHORT:
            m_indices_cnt = static_cast<GLuint>(vi_size / 2);
            break;
        case GL_UNSIGNED_BYTE:
            m_indices_cnt = static_cast<GLuint>(vi_size / 1);
            break;
        }
        return true;
    }

    /**
     * Updates the data in the index buffer
     *
     * @param indices Container with the new index data
     * @param indices_type The data type of the new indices
     * @param primitive type The new type of primitives to draw
     * @return
     */
    template <typename IndexContainer>
    bool rebufferIndexData(IndexContainer const& indices, GLenum indices_type, GLenum primitive_type);

    /**
     * Sets this vertex array as currently bound one
     */
    void bindVertexArray() const { glBindVertexArray(m_va_handle); }

    /**
     * Draw function for your conveniences.
     * If you need/want to work with sth. different from glDrawElementsInstanced,
     * use bindVertexArray() and do your own thing.
     *
     * @param instance_cnt Number of instances to draw
     */
    void draw(GLsizei instance_cnt = 1) {
        glBindVertexArray(m_va_handle);
        glDrawElementsInstanced(m_primitive_type, m_indices_cnt, m_indices_type, nullptr, instance_cnt);
        glBindVertexArray(0);
    }

    /**
     * Gets the current vertex layout
     *
     * @return The vertex layout
     */
    VertexLayout getVertexLayout() const { return m_vertex_descriptor; }

    /**
     * Gets the current count of stored indices
     *
     * @return The count of stored indices
     */
    GLuint getIndicesCount() const { return m_indices_cnt; }

    /**
     * Gets the current index data type
     *
     * @return The data type of the index data
     */
    GLenum getIndicesType() const { return m_indices_type; }

    /**
     * Gets the current type of the stored primitives
     *
     * @return  The type of the primitives
     */
    GLenum getPrimitiveType() const { return m_primitive_type; }

    /**
     * Gets the size of a vertex buffer in bytes
     *
     * @param vbo_idx The index of the vertex buffer to retrieve the size for
     * @return The size of the wanted vertex buffer in bytes
     */
    GLsizeiptr getVertexBufferByteSize(size_t vbo_idx) const {
        if (vbo_idx < m_vbos.size())
            return m_vbos[vbo_idx]->getByteSize();
        else
            return 0;
        // TODO: log some kind of error?
    }

    /**
     * Gets the size of the index buffer in bytes
     *
     * @return The size of the index buffer in bytes
     */
    GLsizeiptr getIndexBufferByteSize() const { return m_ibo.getByteSize(); }

    /**
     * Get all of the stored vertex buffers
     *
     * @return Vector containing references to all stored buffers
     */
    std::vector<BufferObjectPtr> const& getVbo() const { return m_vbos; }

    /**
     * Get the stored index buffer
     *
     * @return The stored index buffer
     */
    BufferObject const& getIbo() const { return m_ibo; }

private:
    /** Buffer objects for each attribute */
    std::vector<BufferObjectPtr> m_vbos;

    /** Buffer for the vertex indices */
    BufferObject m_ibo;

    /** Vertex array handle */
    GLuint m_va_handle;

    /** The vertex layout for the whole mesh */
    VertexLayout m_vertex_descriptor;

    /** Number of index values */
    GLuint m_indices_cnt;

    /** Data type of the index values */
    GLenum m_indices_type;

    /** Usage of the meshes data */
    GLenum m_usage;

    /** Type of the meshes primitives */
    GLenum m_primitive_type;
};
} // namespace gl
} // namespace utility
} // namespace core
} // namespace megamol

/*
 * megamol::core::utility::gl::Mesh::Mesh
 */
template <typename VertexContainer, typename IndexContainer>
megamol::core::utility::gl::Mesh::Mesh(VertexContainer const& vertices, IndexContainer const& indices,
    VertexLayout const& vertex_descriptor, GLenum indices_type, GLenum usage, GLenum primitive_type)
    : m_ibo(GL_ELEMENT_ARRAY_BUFFER, indices, usage)
    , // TODO ibo generation in constructor might fail? needs a bound vao?
    m_vertex_descriptor(vertex_descriptor)
    , m_va_handle(0)
    , m_indices_cnt(0)
    , m_indices_type(indices_type)
    , m_usage(usage)
    , m_primitive_type(primitive_type) {
    m_vbos.emplace_back(std::make_unique<BufferObject>(GL_ARRAY_BUFFER, vertices, m_usage));

    glGenVertexArrays(1, &m_va_handle);

    // set attribute pointer and vao state
    glBindVertexArray(m_va_handle);
    m_ibo.bind();
    m_vbos.back()->bind();
    GLuint attrib_idx = 0;
    for (auto& attribute : vertex_descriptor.attributes) {
        glEnableVertexAttribArray(attrib_idx);
        glVertexAttribPointer(attrib_idx, attribute.size, attribute.type, attribute.normalized,
            vertex_descriptor.byte_size, (GLvoid*)attribute.offset);

        attrib_idx++;
    }
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    GLuint vi_size = static_cast<GLuint>(indices.size() * sizeof(IndexContainer::value_type));

    switch (m_indices_type) {
    case GL_UNSIGNED_INT:
        m_indices_cnt = static_cast<GLuint>(vi_size / 4);
        break;
    case GL_UNSIGNED_SHORT:
        m_indices_cnt = static_cast<GLuint>(vi_size / 2);
        break;
    case GL_UNSIGNED_BYTE:
        m_indices_cnt = static_cast<GLuint>(vi_size / 1);
        break;
    }
}

/*
 * megamol::core::utility::gl::Mesh::bufferVertexSubData
 */
template <typename VertexContainer>
void megamol::core::utility::gl::Mesh::bufferVertexSubData(
    size_t vbo_idx, VertexContainer const& vertices, GLsizeiptr byte_offset) {
    if (vbo_idx < m_vbos.size()) m_vbos[vbo_idx]->bufferSubData<VertexContainer>(vertices, byte_offset);
}

/*
 * megamol::core::utility::gl::Mesh::bufferIndexSubData
 */
template <typename IndexContainer>
void megamol::core::utility::gl::Mesh::bufferIndexSubData(IndexContainer const& indices, GLsizeiptr byte_offset) {
    // TODO check type against current index type
    m_ibo.bufferSubData<IndexContainer>(indices, byte_offset);
}

/*
 * megamol::core::utility::gl::Mesh::rebufferVertexData
 */
template <typename VertexContainer>
bool megamol::core::utility::gl::Mesh::rebufferVertexData(size_t vbo_idx, VertexContainer const& vertices) {
    if (vbo_idx < m_vbos.size()) {
        return m_vbos[vbo_idx]->rebuffer(vertices);
    }
    return false;
}

/*
 * megamol::core::utility::gl::Mesh::rebufferVertexData
 */
template <typename VertexContainer>
bool megamol::core::utility::gl::Mesh::rebufferVertexData(VertexContainer const& vertices, VertexLayout descriptor) {
    m_vbos.clear();

    m_vbos.emplace_back(std::make_unique<BufferObject>(GL_ARRAY_BUFFER, vertices, m_usage));

    m_vertex_descriptor = descriptor;

    glBindVertexArray(m_va_handle);
    m_vbos.back()->bind();
    GLuint attrib_idx = 0;
    for (auto& attribute : m_vertex_descriptor.attributes) {
        glEnableVertexAttribArray(attrib_idx);
        glVertexAttribPointer(attrib_idx, attribute.size, attribute.type, attribute.normalized,
            m_vertex_descriptor.byte_size, reinterpret_cast<GLvoid*>(attribute.offset));

        attrib_idx++;
    }
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    auto err = glGetError();
    return (err == GL_NO_ERROR);
}

/*
 * megamol::core::utility::gl::Mesh::rebufferIndexData
 */
template <typename IndexContainer>
bool megamol::core::utility::gl::Mesh::rebufferIndexData(IndexContainer const& indices) {
    if (!m_ibo.rebuffer(indices)) return false;

    GLuint vi_size = static_cast<GLuint>(indices.size() * sizeof(IndexContainer::value_type));

    switch (m_indices_type) {
    case GL_UNSIGNED_INT:
        m_indices_cnt = static_cast<GLuint>(vi_size / 4);
        break;
    case GL_UNSIGNED_SHORT:
        m_indices_cnt = static_cast<GLuint>(vi_size / 2);
        break;
    case GL_UNSIGNED_BYTE:
        m_indices_cnt = static_cast<GLuint>(vi_size / 1);
        break;
    }
    return true;
}

/*
 * megamol::core::utility::gl::Mesh::rebufferIndexData
 */
template <typename IndexContainer>
bool megamol::core::utility::gl::Mesh::rebufferIndexData(
    IndexContainer const& indices, GLenum indices_type, GLenum primitive_type) {
    if (!m_ibo.rebuffer(indices)) return false;

    m_indices_type = indices_type;
    m_primitive_type = primitive_type;

    GLuint vi_size = static_cast<GLuint>(indices.size() * sizeof(IndexContainer::value_type));

    switch (m_indices_type) {
    case GL_UNSIGNED_INT:
        m_indices_cnt = static_cast<GLuint>(vi_size / 4);
        break;
    case GL_UNSIGNED_SHORT:
        m_indices_cnt = static_cast<GLuint>(vi_size / 2);
        break;
    case GL_UNSIGNED_BYTE:
        m_indices_cnt = static_cast<GLuint>(vi_size / 1);
        break;
    }

    auto err = glGetError();
    return (err == GL_NO_ERROR);
}

#endif // !MEGAMOLCORE_MESH_H_INCLUDED
