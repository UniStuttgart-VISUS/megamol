/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <memory>

#include <glowl/glowl.h>

#include "mmcore/flags/FlagStorageTypes.h"

namespace glowl_experimental {
class ImmutableBufferObject {
private:
    GLuint m_name;
    GLsizeiptr m_byte_size;

public:
    template<typename Container>
    explicit ImmutableBufferObject(Container const& datastorage, GLbitfield flags = 0)
            : m_name(0)
            , m_byte_size(static_cast<GLsizeiptr>(datastorage.size() * sizeof(typename Container::value_type))) {
        glCreateBuffers(1, &m_name);
        glNamedBufferStorage(m_name, m_byte_size, datastorage.data(), flags);
    }

    ImmutableBufferObject(GLvoid const* data, GLsizeiptr byte_size, GLbitfield flags = 0)
            : m_name(0)
            , m_byte_size(byte_size) {
        glCreateBuffers(1, &m_name);
        glNamedBufferStorage(m_name, m_byte_size, data, flags);
    }

    inline ~ImmutableBufferObject() {
        glDeleteBuffers(1, &m_name);
    }

    inline GLuint getName() const {
        return m_name;
    }

    GLsizeiptr getByteSize() const {
        return m_byte_size;
    }

    inline void bind(GLenum target) const {
        glBindBuffer(target, m_name);
    }

    inline void bindBase(GLenum target, GLuint index) const {
        glBindBufferBase(target, index, m_name);
    }

    static inline void copy(ImmutableBufferObject& src, ImmutableBufferObject& tgt, GLintptr readOffset,
        GLintptr writeOffset, GLsizeiptr size) {
        glCopyNamedBufferSubData(src.m_name, tgt.m_name, readOffset, writeOffset, size);
    }
};
} // namespace glowl_experimental

namespace megamol::mmstd_gl {

class FlagCollection_GL {
public:
    std::shared_ptr<glowl_experimental::ImmutableBufferObject> flags;

    void validateFlagCount(core::FlagStorageTypes::index_type num) {
        constexpr auto defaultFlag = core::FlagStorageTypes::to_integral(core::FlagStorageTypes::flag_bits::ENABLED);

        if (flags == nullptr || flags->getByteSize() / sizeof(core::FlagStorageTypes::flag_item_type) < num) {
            core::FlagStorageTypes::flag_vector_type temp_data(num, defaultFlag);
            auto temp_buffer = std::make_shared<glowl_experimental::ImmutableBufferObject>(
                temp_data, GL_DYNAMIC_STORAGE_BIT | GL_CLIENT_STORAGE_BIT);

            if (flags != nullptr) {
                glowl_experimental::ImmutableBufferObject::copy(*flags, *temp_buffer, 0, 0, flags->getByteSize());
            }

            flags = temp_buffer;
        }
    }
};
} // namespace megamol::mmstd_gl
