/*
 * ImageWrapper_Conversions.cpp
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */


#include "ImageWrapper_to_ByteArray.h"
#include "ImageWrapper_to_GLTexture.h"

#include "glad/glad.h"
#include <tuple>
//#include <cstring>

using namespace megamol::frontend_resources;

namespace /*gl texture handling*/ {
    void gl_init_texture(unsigned int& handle) {
        glCreateTextures(GL_TEXTURE_2D, 1, &handle);
    }

    std::tuple<int, int, int> getInternalformatFormatType(ImageWrapper::DataChannels channels) {
        const auto internalformat = channels == ImageWrapper::DataChannels::RGB8 ? GL_RGB8 : GL_RGBA8;
        const auto format = channels == ImageWrapper::DataChannels::RGB8 ? GL_RGB : GL_RGBA;
        const auto type = GL_UNSIGNED_BYTE;

        return {internalformat, format, type};
    }

    void gl_set_and_resize_texture(unsigned int& handle, ImageWrapper::ImageSize size, ImageWrapper::DataChannels channels, const void* data = nullptr) {
        if (!handle)
            gl_init_texture(handle);

        int old_handle = 0;
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &old_handle);

        glBindTexture(GL_TEXTURE_2D, handle);

        glTextureParameteri(handle, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTextureParameteri(handle, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        glTextureParameteri(handle, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(handle, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        const auto [internalformat, format, type] = getInternalformatFormatType(channels);

        glTexImage2D(GL_TEXTURE_2D, 0, internalformat, size.width, size.height, 0, format, type, data);

        glBindTexture(GL_TEXTURE_2D, old_handle);
    }

    void gl_copy_texture(unsigned int from_handle, unsigned int to_handle, ImageWrapper::ImageSize size) {
        glCopyImageSubData(from_handle, GL_TEXTURE_2D, 0, 0, 0, 0, to_handle, GL_TEXTURE_2D, 0, 0, 0, 0, size.width, size.height, 1);
    }

    void gl_delete_texture(unsigned int& handle) {
        glDeleteTextures(1, &handle);
        handle = 0;
    }

    void gl_download_texture_to_vector(unsigned int handle, ImageWrapper::ImageSize size, ImageWrapper::DataChannels channels, std::vector<byte>& target) {
        target.resize(size.width * size.height * 4); // see below

        const auto [internalformat, format, type] = getInternalformatFormatType(channels);

        // returns RGBA8 and fills unused components with defaults
        // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glGetTextureSubImage.xhtml
        // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glGetTexImage.xhtml
        glGetTextureSubImage(handle, 0, 0, 0, 0, size.width, size.height, 1, format, type, target.size(), target.data());
        //glGetTextureImage(handle, 0, format, type, target.size(), target.data());
    }

    void gl_upload_texture_from_vector(unsigned int& handle, ImageWrapper::ImageSize size, ImageWrapper::DataChannels channels, std::vector<byte> const& source) {
        gl_set_and_resize_texture(handle, size, channels, source.data());
    }
}

gl_texture::gl_texture(ImageWrapper const& image)
{
    this->from_image(image);
}

unsigned int gl_texture::as_gl_handle()
{
    return this->texture;
}

gl_texture::~gl_texture()
{
    this->clear();
}

gl_texture::gl_texture(gl_texture const& other)
{
    this->assign(other);

    if (other.texture_owned) {
        this->texture = 0;
        auto& image = *this->image_wrapper_ptr;
        gl_set_and_resize_texture(this->texture, image.size(), image.channels);
        gl_copy_texture(other.texture, this->texture, image.size());
    }
}
gl_texture& gl_texture::operator=(gl_texture const& other)
{
    this->clear();
    this->assign(other);

    if (other.texture_owned) {
        this->texture = 0;
        auto& image = *this->image_wrapper_ptr;
        gl_set_and_resize_texture(this->texture, image.size(), image.channels);
        gl_copy_texture(other.texture, this->texture, image.size());
    }

    return *this;
}

gl_texture::gl_texture(gl_texture&& other) noexcept
{
    this->assign(other);
    other.texture_owned = false;
    other.clear();
}
gl_texture& gl_texture::operator=(gl_texture&& other) noexcept
{
    this->clear();
    this->assign(other);
    other.texture_owned = false;
    other.clear();

    return *this;
}

void gl_texture::from_image(ImageWrapper const& image)
{
    this->image_wrapper_ptr = const_cast<ImageWrapper*>(&image);

    switch (image.image_type) {
    case WrappedImageType::ByteArray:
        this->texture_owned = true;
        gl_set_and_resize_texture(this->texture, image.size(), image.channels, static_cast<std::vector<byte> const*>(image.borrowed_image_handle)->data());
        break;
    case WrappedImageType::GLTexureHandle:
        // avoid dangling gl texture handle if we owned one before
        if (this->texture != 0 || this->texture_owned) {
            gl_delete_texture(this->texture);
        }
        this->texture_owned = false;
        this->texture = static_cast<unsigned int>(reinterpret_cast<long>(image.borrowed_image_handle));
        break;
    }
}

gl_texture& gl_texture::operator=(ImageWrapper const& image)
{
    this->from_image(image);

    return *this;
}

void gl_texture::assign(gl_texture const& other) {
    this->image_wrapper_ptr = other.image_wrapper_ptr;
    this->texture_owned     = other.texture_owned;
    this->texture           = other.texture;
}
void gl_texture::clear() {
    if (this->texture_owned) {
        gl_delete_texture(this->texture);
    }
    this->image_wrapper_ptr = nullptr;
    this->texture_owned     = false;
    this->texture           = 0;
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

    switch (image.image_type) {
    case WrappedImageType::ByteArray:
        this->texture_owned = false;
        this->texture_ptr = static_cast<std::vector<byte>*>(image.borrowed_image_handle);
        break;
    case WrappedImageType::GLTexureHandle:
        this->texture_owned = true;
        auto gl_texture = static_cast<unsigned int>(reinterpret_cast<long>(image.borrowed_image_handle));
        gl_download_texture_to_vector(gl_texture, image.size(), image.channels, this->texture);
        this->texture_ptr = &this->texture;
        break;
    }
}

