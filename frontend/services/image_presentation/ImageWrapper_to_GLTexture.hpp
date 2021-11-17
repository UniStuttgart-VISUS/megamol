/*
 * ImageWrapper_to_GLTexture.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "ImageWrapper.h"

namespace megamol {
namespace frontend_resources {

    // RAII manager for on-the-fly gl texture handle
    struct gl_texture {
        unsigned int texture = 0;
        unsigned int texture_reference = 0;
        ImageWrapper* image_wrapper_ptr = nullptr;

        ImageWrapper::ImageSize size;

        gl_texture(ImageWrapper const& image);
        ~gl_texture(); // frees texture if owned
        // rule of five
        gl_texture(gl_texture const& other);
        gl_texture(gl_texture&& other) noexcept;
        gl_texture& operator=(gl_texture const& other);
        gl_texture& operator=(gl_texture&& other) noexcept;

        gl_texture& operator=(ImageWrapper const& image);

        unsigned int as_gl_handle();

        private:
        void assign(gl_texture const& other, bool take_ownership);
        void clear();
        void from_image(ImageWrapper const& image);
    };

} /* end namespace frontend_resources */
} /* end namespace megamol */
