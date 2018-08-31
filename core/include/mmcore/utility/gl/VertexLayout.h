#ifndef MEGAMOLCORE_VERTEXLAYOUT_H_INCLUDED
#define MEGAMOLCORE_VERTEXLAYOUT_H_INCLUDED

#include "vislib/graphics/gl/IncludeAllGL.h"

namespace megamol {
namespace core {
namespace utility {
namespace gl {

/**
 * \struct VertexLayout
 *
 * \brief Container for vertex layout descritions. Used by Mesh class.
 *
 * \author Michael Becher
 */
struct VertexLayout {
    struct Attribute {
        Attribute(GLenum type, GLint size, GLboolean normalized, GLsizei offset)
            : size(size), type(type), normalized(normalized), offset(offset) {}

        GLint size;
        GLenum type;
        GLboolean normalized;
        GLsizei offset;
    };

    VertexLayout() : byte_size(0), attributes() {}
    VertexLayout(GLsizei byte_size, const std::vector<Attribute>& attributes)
        : byte_size(byte_size), attributes(attributes) {}
    VertexLayout(GLsizei byte_size, std::vector<Attribute>&& attributes)
        : byte_size(byte_size), attributes(attributes) {}

    GLsizei byte_size;
    std::vector<Attribute> attributes;
};

inline bool operator==(VertexLayout::Attribute const& lhs, VertexLayout::Attribute const& rhs) {
    return lhs.normalized == rhs.normalized && lhs.offset == rhs.offset && lhs.size == rhs.size && lhs.type == rhs.type;
}

inline bool operator==(VertexLayout const& lhs, VertexLayout const& rhs) {
    bool rtn = (lhs.byte_size == rhs.byte_size);

    if (lhs.attributes.size() == rhs.attributes.size()) {
        for (size_t i = 0; i < lhs.attributes.size(); ++i) {
            rtn &= (lhs.attributes == rhs.attributes);
        }
    } else {
        rtn = false;
    }

    return rtn;
}

} // namespace gl
} // namespace utility
} // namespace core
} // namespace megamol

#endif // !MEGAMOLCORE_VERTEXLAYOUT_H_INCLUDED
