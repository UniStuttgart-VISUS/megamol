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
        // rule of five
        gl_texture(gl_texture const& other);
        gl_texture(gl_texture&& other) noexcept;
        gl_texture& operator=(gl_texture const& other);
        gl_texture& operator=(gl_texture&& other) noexcept;

        gl_texture& operator=(ImageWrapper const& image);

        unsigned int as_gl_handle();


#ifdef WITH_GL
        ~gl_texture(); // frees texture if owned
    private:
        void assign(gl_texture const& other, bool take_ownership);
        void from_image(ImageWrapper const& image);
#else
        ~gl_texture() = default;
    private:
        void assign(gl_texture const& other, bool take_ownership) {}
        void from_image(ImageWrapper const& image) {}
#endif
        void clear() {
            this->image_wrapper_ptr = nullptr;
            this->texture_reference = 0;
            this->texture = 0;
            this->size = {0, 0};
        }

    };

    inline gl_texture& gl_texture::operator=(ImageWrapper const& image) {
        this->from_image(image);

        return *this;
    }

    inline gl_texture::gl_texture(gl_texture const& other) {
        this->assign(other, false);
    }

    inline gl_texture& gl_texture::operator=(gl_texture const& other) {
        this->assign(other, false);

        return *this;
    }

    inline gl_texture::gl_texture(gl_texture&& other) noexcept {
        this->assign(other, true);
        other.clear();
    }
    inline gl_texture& gl_texture::operator=(gl_texture&& other) noexcept {
        this->assign(other, true);
        other.clear();

        return *this;
    }

    inline gl_texture::gl_texture(ImageWrapper const& image) {
        this->from_image(image);
    }

    inline unsigned int gl_texture::as_gl_handle() {
        return this->texture_reference;
    }
} /* end namespace frontend_resources */
} /* end namespace megamol */
