/*
 * ImageWrapper_to_ByteArray.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "ImageWrapper.h"
#include <vector>

using byte = unsigned char;

namespace {
unsigned int to_uint(void* ptr) {
    return static_cast<unsigned int>(reinterpret_cast<long>(ptr));
}

std::vector<byte>* to_vector(void* ptr) {
    return static_cast<std::vector<byte>*>(ptr); //TODO
}
} // namespace

namespace megamol {
namespace frontend_resources {

    struct byte_texture {
        std::vector<byte> texture;
        std::vector<byte>* texture_ptr = nullptr;
        bool texture_owned = false;
        ImageWrapper* image_wrapper_ptr = nullptr;

        ImageWrapper::ImageSize size;

        byte_texture(ImageWrapper const& image);

        byte_texture& operator=(ImageWrapper const& image);

        std::vector<byte> const& as_byte_vector();

        private:
        void from_image(ImageWrapper const& image);
    };

} /* end namespace frontend_resources */
} /* end namespace megamol */
