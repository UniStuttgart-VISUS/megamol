/*
 * ImageWrapper_Conversions.cpp
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */


#include <stdexcept>

#include "ImageWrapper_to_GLTexture.hpp"
#include "ImageWrapper_to_ByteArray.hpp"
#include <tuple>

using namespace megamol::frontend_resources;
namespace gl_wrapper_impl {
void gl_download_texture_to_vector(
    unsigned int handle, ImageWrapper::ImageSize size, ImageWrapper::DataChannels channels, std::vector<byte>& target);
}

byte_texture::byte_texture(ImageWrapper const& image)
{
    this->from_image(image);
}

byte_texture& byte_texture::operator=(ImageWrapper const& image)
{
    this->from_image(image);

    return *this;
}

std::vector<byte> const& byte_texture::as_byte_vector()
{
    return *this->texture_ptr;
}

void byte_texture::from_image(ImageWrapper const& image)
{
    this->image_wrapper_ptr = const_cast<ImageWrapper*>(&image);
    this->size = image.size;

    switch (image.type) {
    case WrappedImageType::ByteArray:
        this->texture_owned = false;
        this->texture_ptr = to_vector(image.referenced_image_handle);
        break;
    case WrappedImageType::GLTexureHandle:
#ifdef WITH_GL
        this->texture_owned = true;
        auto gl_texture = to_uint(image.referenced_image_handle);
        gl_wrapper_impl::gl_download_texture_to_vector(gl_texture, image.size, image.channels, this->texture);
        this->texture_ptr = &this->texture;
#else
        throw std::runtime_error("[ImageWrapper] Used GL texture as input, but GL is not active.");
#endif
        break;
    }
}

