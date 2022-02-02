/*
 * ImageWrapper_to_GLTexture.cpp
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "ImageWrapper_to_GLTexture.hpp"
#include "ImageWrapper_Conversion_Helpers.hpp"

#include "glad/gl.h"

#include <stdexcept>
#include <tuple>

using namespace megamol::frontend_resources;

namespace gl_wrapper_impl {

void gl_init_texture(unsigned int& handle) {
    glCreateTextures(GL_TEXTURE_2D, 1, &handle);
}

std::tuple<int, int, int> getInternalformatFormatType(ImageWrapper::DataChannels channels) {
    if (channels != ImageWrapper::DataChannels::RGBA8 && channels != ImageWrapper::DataChannels::RGB8) {
        throw std::runtime_error(
            "[ImageWrapper_to_GLTexture.cpp] Only image with RGBA8 or RGA8 channels supported for now...");
    }

    const auto internalformat = channels == ImageWrapper::DataChannels::RGB8 ? GL_RGB8 : GL_RGBA8;
    const auto format = channels == ImageWrapper::DataChannels::RGB8 ? GL_RGB : GL_RGBA;
    const auto type = GL_UNSIGNED_BYTE;

    return {internalformat, format, type};
}

void gl_set_and_resize_texture(unsigned int& handle, ImageWrapper::ImageSize size, ImageWrapper::DataChannels channels,
    const void* data = nullptr) {
    if (!handle)
        gl_init_texture(handle);

    int old_handle = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &old_handle);

    glBindTexture(GL_TEXTURE_2D, handle);

    glTextureParameteri(handle, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(handle, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(handle, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(handle, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    const auto [internalformat, format, type] = getInternalformatFormatType(channels);

    glTexImage2D(GL_TEXTURE_2D, 0, internalformat, size.width, size.height, 0, format, type, data);

    glBindTexture(GL_TEXTURE_2D, old_handle);
}

void gl_copy_texture(unsigned int from_handle, unsigned int to_handle, ImageWrapper::ImageSize size) {
    glCopyImageSubData(
        from_handle, GL_TEXTURE_2D, 0, 0, 0, 0, to_handle, GL_TEXTURE_2D, 0, 0, 0, 0, size.width, size.height, 1);
}

void gl_delete_texture(unsigned int& handle) {
    glDeleteTextures(1, &handle);
    handle = 0;
}

void gl_download_texture_to_vector(
    unsigned int handle, ImageWrapper::ImageSize size, ImageWrapper::DataChannels channels, std::vector<byte>& target) {
    target.resize(size.width * size.height * 4); // see below

    const auto [internalformat, format, type] = getInternalformatFormatType(channels);

    // returns RGBA8 and fills unused components with defaults
    // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glGetTextureSubImage.xhtml
    // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glGetTexImage.xhtml
    glGetTextureSubImage(handle, 0, 0, 0, 0, size.width, size.height, 1, format, type, target.size(), target.data());
    //glGetTextureImage(handle, 0, format, type, target.size(), target.data());
}

void gl_upload_texture_from_vector(unsigned int& handle, ImageWrapper::ImageSize size,
    ImageWrapper::DataChannels channels, std::vector<byte> const& source) {
    gl_set_and_resize_texture(handle, size, channels, source.data());
}
} // namespace gl_wrapper_impl


gl_texture::~gl_texture() {
    if (this->texture != 0) {
        gl_wrapper_impl::gl_delete_texture(this->texture);
    }
    this->clear();
}

void gl_texture::from_image(ImageWrapper const& image) {
    this->image_wrapper_ptr = const_cast<ImageWrapper*>(&image);
    this->size = image.size;

    switch (image.type) {
    case WrappedImageType::ByteArray:
        gl_wrapper_impl::gl_set_and_resize_texture(
            this->texture, image.size, image.channels, conversion::to_vector(image.referenced_image_handle)->data());
        this->texture_reference = this->texture;
        break;
    case WrappedImageType::GLTexureHandle:
        this->texture_reference = conversion::to_uint(image.referenced_image_handle);
        break;
    }
}

void gl_texture::assign(gl_texture const& other, bool take_ownership) {
    this->image_wrapper_ptr = other.image_wrapper_ptr;
    this->size = other.size;

    if (take_ownership) {
        // move texture data
        this->texture_reference = other.texture_reference;
        this->texture = other.texture;
        // other gets cleared after this assign
    } else {
        // actually copy data into own texture
        auto copy = [](unsigned int from, unsigned int& to, auto& size, auto& channels) {
            gl_wrapper_impl::gl_set_and_resize_texture(to, size, channels);
            gl_wrapper_impl::gl_copy_texture(from, to, size);
        };

        auto& image = *this->image_wrapper_ptr;

        copy(other.texture_reference, this->texture, image.size, image.channels);
        this->texture_reference = this->texture;
    }
}
