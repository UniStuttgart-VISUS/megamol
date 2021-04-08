/*
 * ImageWrapper.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <vector>
#include <memory>
#include <string>
#include <optional>
#include <functional>

namespace megamol {
namespace frontend_resources {

enum class WrappedImageType {
    GLTexureHandle, // data array holds a GL texture handle
    ByteArray       // data array holds image data as raw bytes
};
static const WrappedImageType GLTexureHandle = WrappedImageType::GLTexureHandle;
static const WrappedImageType ByteArray = WrappedImageType::ByteArray;

// the idea is that each AbstractView (via the concrece View implementation)
// fills a ImageWrapper for the frontend to use.
// depending on the WrappedImageType and DataChannels the frontend knows
// what it can do with each wrapped image
//   - either use the contained texture directly for GL rendering
//   - or forward the contained byte data to interested sources: show in window, write to screenshot file, send via network...
// the ImageWrapper does not own the wrapped image, it acts more like a generic reference wrapper for GL or byte array images
struct ImageWrapper {

    enum class DataChannels {
        // for texture and byte array, tells us how many channels there are
        RGB8,
        RGBA8,
    };
    struct ImageSize {
        size_t width = 0;
        size_t height = 0;

        bool operator!=(ImageSize const& other) { return width != other.width || height != other.height; }
    };

    ImageWrapper(ImageSize size, DataChannels channels, WrappedImageType type, const void* data);
    ImageWrapper() = default;

    WrappedImageType image_type{};
    ImageSize image_size{};
    DataChannels channels{};

    void* borrowed_image_handle = nullptr;

    size_t channels_count() const;
    const ImageSize& size() const { return image_size; }
};

template <WrappedImageType>
ImageWrapper wrap_image(ImageWrapper::ImageSize size, const void* data = nullptr, ImageWrapper::DataChannels channels = ImageWrapper::DataChannels::RGBA8);

template <>
ImageWrapper wrap_image<WrappedImageType::GLTexureHandle>(ImageWrapper::ImageSize size, const void* data, ImageWrapper::DataChannels channels);
template <>
ImageWrapper wrap_image<WrappedImageType::ByteArray>(ImageWrapper::ImageSize size, const void* data, ImageWrapper::DataChannels channels);

ImageWrapper wrap_image(ImageWrapper::ImageSize size, unsigned int gl_texture_handle, ImageWrapper::DataChannels channels);

ImageWrapper wrap_image(ImageWrapper::ImageSize size, std::vector<unsigned char> const& byte_texture, ImageWrapper::DataChannels channels);

size_t channels_count(ImageWrapper::DataChannels channels);

} /* end namespace frontend_resources */
} /* end namespace megamol */
