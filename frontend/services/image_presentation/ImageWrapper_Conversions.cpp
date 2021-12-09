/*
 * ImageWrapper_Conversions.cpp
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */


#include <stdexcept>

#include "ImageWrapper_Conversion_Helpers.hpp"
#include "ImageWrapper_to_ByteArray.hpp"
#include "ImageWrapper_to_GLTexture.hpp"

#include <tuple>

using namespace megamol::frontend_resources;

// implemented in ImageWrapper_to_GLTexture.cpp
namespace gl_wrapper_impl {
void gl_download_texture_to_vector(
    unsigned int handle, ImageWrapper::ImageSize size, ImageWrapper::DataChannels channels, std::vector<byte>& target);
} // namespace gl_wrapper_impl

namespace megamol {
namespace frontend_resources {
namespace conversion {
unsigned int to_uint(void* ptr) {
    return static_cast<unsigned int>(reinterpret_cast<long>(ptr));
}

std::vector<byte>* to_vector(void* ptr) {
    return static_cast<std::vector<byte>*>(ptr);
}
} // namespace conversion
} // namespace frontend_resources
} // namespace megamol

gl_texture::gl_texture(ImageWrapper const& image) {
#ifdef WITH_GL
    this->from_image(image);
#else
    throw std::runtime_error("[ImageWrapper -> gl_texture] Trying to construct GL texture, but GL is not active.");
#endif
}

unsigned int gl_texture::as_gl_handle() {
    return this->texture_reference;
}

gl_texture::gl_texture(gl_texture const& other) {
    this->assign(other, false);
}

gl_texture& gl_texture::operator=(gl_texture const& other) {
    this->assign(other, false);

    return *this;
}

gl_texture::gl_texture(gl_texture&& other) noexcept {
    this->assign(other, true);
    other.clear();
}

gl_texture& gl_texture::operator=(gl_texture&& other) noexcept {
    this->assign(other, true);
    other.clear();

    return *this;
}

gl_texture& gl_texture::operator=(ImageWrapper const& image) {
#ifdef WITH_GL
    this->from_image(image);

    return *this;
#else
    throw std::runtime_error("[ImageWrapper -> gl_texture] Trying to construct GL texture, but GL is not active.");
#endif
}

void gl_texture::clear() {
    this->image_wrapper_ptr = nullptr;
    this->texture_reference = 0;
    this->texture = 0;
    this->size = {0, 0};
}

byte_texture::byte_texture(ImageWrapper const& image) {
    this->from_image(image);
}

byte_texture& byte_texture::operator=(ImageWrapper const& image) {
    this->from_image(image);

    return *this;
}

std::vector<byte> const& byte_texture::as_byte_vector() {
    return *this->texture_ptr;
}

void byte_texture::from_image(ImageWrapper const& image) {
    this->image_wrapper_ptr = const_cast<ImageWrapper*>(&image);
    this->size = image.size;

    switch (image.type) {
    case WrappedImageType::ByteArray:
        this->texture_owned = false;
        this->texture_ptr = conversion::to_vector(image.referenced_image_handle);
        break;
    case WrappedImageType::GLTexureHandle:
#ifdef WITH_GL
        this->texture_owned = true;
        auto gl_texture = conversion::to_uint(image.referenced_image_handle);
        gl_wrapper_impl::gl_download_texture_to_vector(gl_texture, image.size, image.channels, this->texture);
        this->texture_ptr = &this->texture;
#else
        throw std::runtime_error("[ImageWrapper -> byte_texture] Used GL texture as input, but GL is not active.");
#endif
        break;
    }
}
