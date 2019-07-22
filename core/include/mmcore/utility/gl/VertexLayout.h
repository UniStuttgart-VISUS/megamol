/*
 * VertexLayout.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCORE_VERTEXLAYOUT_H_INCLUDED
#define MEGAMOLCORE_VERTEXLAYOUT_H_INCLUDED

#include "vislib/graphics/gl/IncludeAllGL.h"

namespace megamol {
namespace core {
namespace utility {
namespace gl {

/**
 * @struct VertexLayout
 *
 * @brief Container for vertex layout descriptions. Used by Mesh class.
 *
 * @author Michael Becher
 */
struct VertexLayout {
    struct Attribute {
        /** Ctor */
        Attribute(GLenum type, GLint size, GLboolean normalized, GLsizei offset)
            : size(size), type(type), normalized(normalized), offset(offset) {}

        /** Size of the attribute in bytes */
        GLint size;

        /** Type of the attribute */
        GLenum type;

        /** Flag determining whether the attribute is normalized */
        GLboolean normalized;

        /** Offset of the attribute from the beginning */
        GLsizei offset;
    };

    /** Ctor */
    VertexLayout() : byte_size(0), attributes() {}

    /** Ctor */
    VertexLayout(GLsizei byte_size, const std::vector<Attribute>& attributes)
        : byte_size(byte_size), attributes(attributes) {}

    /** Ctor */
    VertexLayout(GLsizei byte_size, std::vector<Attribute>&& attributes)
        : byte_size(byte_size), attributes(attributes) {}

    /** Size of the layout in bytes */
    GLsizei byte_size;

    /** Vector storing all attributes of the layout */
    std::vector<Attribute> attributes;
};

/**
 * Equality operator overload
 *
 * @param lhs The left hand operand
 * @param rhs The right hand operand
 * @return True if both sides are equal, false otherwise
 */
inline bool operator==(VertexLayout::Attribute const& lhs, VertexLayout::Attribute const& rhs) {
    return lhs.normalized == rhs.normalized && lhs.offset == rhs.offset && lhs.size == rhs.size && lhs.type == rhs.type;
}

/**
 * Equality operator overload
 *
 * @param lhs The left hand operand
 * @param rhs The right hand operand
 * @return True if both sides are equal, false otherwise
 */
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
