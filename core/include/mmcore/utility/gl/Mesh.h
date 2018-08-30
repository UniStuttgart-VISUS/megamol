#ifndef Mesh_hpp
#define Mesh_hpp

/*	Include system libraries */
#include <string>
#include <vector>
#include "vislib/graphics/gl/IncludeAllGL.h"
//#include <iostream>

#include "mmcore/utility/gl/BufferObject.h"
#include "mmcore/utility/gl/VertexLayout.h"

namespace megamol {
namespace core {
namespace utility {
namespace gl {

/**
 * \class Mesh
 *
 * \brief Encapsulates mesh functionality.
 *
 * \author Michael Becher
 */
class Mesh {
public:
    typedef std::unique_ptr<BufferObject> BufferObjectPtr;

    template <typename VertexContainer, typename IndexContainer>
    Mesh(VertexContainer const& vertices, IndexContainer const& indices, VertexLayout const& vertex_descriptor,
        GLenum indices_type = GL_UNSIGNED_INT, GLenum usage = GL_STATIC_DRAW, GLenum primitive_type = GL_TRIANGLES);

    Mesh(GLvoid const* vertex_data, GLsizeiptr vertex_data_byte_size, GLvoid const* index_data,
        GLsizeiptr index_data_byte_size, VertexLayout const& vertex_descriptor, GLenum indices_type = GL_UNSIGNED_INT,
        GLenum usage = GL_STATIC_DRAW, GLenum primitive_type = GL_TRIANGLES);

    /**
     * Hybrid C-interface Mesh constructor for non-interleaved data with one vertex buffer object per vertex attribute.
     */
    Mesh(std::vector<uint8_t*> const& vertex_data, std::vector<size_t> const& vertex_data_byte_sizes,
        GLvoid const* index_data, GLsizeiptr index_data_byte_size, VertexLayout const& vertex_descriptor,
        GLenum indices_type = GL_UNSIGNED_INT, GLenum usage = GL_STATIC_DRAW, GLenum primitive_type = GL_TRIANGLES);

    ~Mesh() { glDeleteVertexArrays(1, &m_va_handle); }

    Mesh(const Mesh& cpy) = delete;
    Mesh(Mesh&& other) = delete;
    Mesh& operator=(Mesh&& rhs) = delete;
    Mesh& operator=(const Mesh& rhs) = delete;

    template <typename VertexContainer>
    void bufferVertexSubData(size_t vbo_idx, VertexContainer const& vertices, GLsizeiptr byte_offset);

    template <typename IndexContainer> void bufferIndexSubData(IndexContainer const& indices, GLsizeiptr byte_offset);

    template <typename VertexContainer> void rebufferVertexData(size_t vbo_idx, VertexContainer const& vertices);

    void rebufferVertexData(size_t vbo_idx, GLvoid const* vertex_data, GLsizeiptr vertex_data_byte_size) {
        if (vbo_idx < m_vbos.size()) m_vbos[vbo_idx]->rebuffer(vertex_data, vertex_data_byte_size);
    }

    template <typename VertexContainer>
    void rebufferVertexData(VertexContainer const& vertices, VertexLayout descriptor);

    template <typename IndexContainer> void rebufferIndexData(IndexContainer const& indices);

    void rebufferIndexData(GLvoid const* index_data, GLsizeiptr index_data_byte_size) {
        m_ibo.rebuffer(index_data, index_data_byte_size);

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
    }

    template <typename IndexContainer>
    void rebufferIndexData(IndexContainer const& indices, GLenum indices_type, GLenum primitive_type);

    void bindVertexArray() const { glBindVertexArray(m_va_handle); }

    /**
     * Draw function for your conveniences.
     * If you need/want to work with sth. different from glDrawElementsInstanced,
     * use bindVertexArray() and do your own thing.
     */
    void draw(GLsizei instance_cnt = 1) {
        glBindVertexArray(m_va_handle);
        glDrawElementsInstanced(m_primitive_type, m_indices_cnt, m_indices_type, nullptr, instance_cnt);
        glBindVertexArray(0);
    }

    VertexLayout getVertexLayout() const { return m_vertex_descriptor; }

    GLuint getIndicesCount() const { return m_indices_cnt; }

    GLenum getIndicesType() const { return m_indices_type; }

    GLenum getPrimitiveType() const { return m_primitive_type; }

    GLsizeiptr getVertexBufferByteSize(size_t vbo_idx) const {
        if (vbo_idx < m_vbos.size())
            return m_vbos[vbo_idx]->getByteSize();
        else
            return 0;
        // TODO: log some kind of error?
    }
    GLsizeiptr getIndexBufferByteSize() const { return m_ibo.getByteSize(); }

    std::vector<BufferObjectPtr> const& getVbo() const { return m_vbos; }
    BufferObject const& getIbo() const { return m_ibo; }

private:
    std::vector<BufferObjectPtr> m_vbos;
    BufferObject m_ibo;
    GLuint m_va_handle;

    VertexLayout m_vertex_descriptor;

    GLuint m_indices_cnt;
    GLenum m_indices_type;
    GLenum m_usage;
    GLenum m_primitive_type;
};
} // namespace gl
} // namespace utility
} // namespace core
} // namespace megamol


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

template <typename VertexContainer>
void megamol::core::utility::gl::Mesh::bufferVertexSubData(
    size_t vbo_idx, VertexContainer const& vertices, GLsizeiptr byte_offset) {
    if (vbo_idx < m_vbos.size()) m_vbos[vbo_idx]->bufferSubData<VertexContainer>(vertices, byte_offset);
}

template <typename IndexContainer>
void megamol::core::utility::gl::Mesh::bufferIndexSubData(IndexContainer const& indices, GLsizeiptr byte_offset) {
    // TODO check type against current index type
    m_ibo.bufferSubData<IndexContainer>(indices, byte_offset);
}

template <typename VertexContainer>
void megamol::core::utility::gl::Mesh::rebufferVertexData(size_t vbo_idx, VertexContainer const& vertices) {
    if (vbo_idx < m_vbos.size()) m_vbos[vbo_idx]->rebuffer(vertices);
}

template <typename VertexContainer>
void megamol::core::utility::gl::Mesh::rebufferVertexData(VertexContainer const& vertices, VertexLayout descriptor) {
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
    if (err != GL_NO_ERROR) {
        std::cerr << "Error - Mesh - rebufferVertexData: " << err << std::endl;
    }
}

template <typename IndexContainer>
void megamol::core::utility::gl::Mesh::rebufferIndexData(IndexContainer const& indices) {
    m_ibo.rebuffer(indices);

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

template <typename IndexContainer>
void megamol::core::utility::gl::Mesh::rebufferIndexData(
    IndexContainer const& indices, GLenum indices_type, GLenum primitive_type) {
    m_ibo.rebuffer(indices);

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
    if (err != GL_NO_ERROR) {
        std::cerr << "Error - Mesh - rebufferIndexData: " << err << std::endl;
    }
}

#endif // !Mesh_hpp
