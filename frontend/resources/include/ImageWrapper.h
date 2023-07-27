/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using byte = unsigned char;

namespace megamol::frontend_resources {

enum class WrappedImageType {
    GLTexureHandle, // void* holds a GL texture handle
    ByteArray       // void* holds a pointer to std::vector<byte>
};

// the idea is that each AbstractView (via the concrete View implementation)
// fills an ImageWrapper for the frontend to use.
// depending on the WrappedImageType and DataChannels the frontend knows
// what it can do with each wrapped image
//   - either use the contained texture directly for GL rendering
//   - or forward the contained byte data to interested sources: show in window, write to screenshot file, send via network...
// the ImageWrapper does not own the wrapped image, it acts more like a generic reference wrapper for GL or byte array images
//
// we expect the lifetime of an ImageWrapper instance to span from beeing returned by a view.Render() call to being presented to the user
// thus, an ImageWrapper does not reference its wrapped texture data for longer than a frame
struct ImageWrapper {

    enum class DataChannels {
        // for texture and byte array, tells us how many channels there are
        RGB8,
        RGBA8,
        RGBAF,
        R8,
        RF
    };
    struct ImageSize {
        size_t width = 0;
        size_t height = 0;
    };

    ImageWrapper(ImageSize size, DataChannels channels, WrappedImageType type, const void* data);
    ImageWrapper(std::string const& name);
    ImageWrapper() = default;

    WrappedImageType type = WrappedImageType::ByteArray;
    ImageSize size = {0, 0};
    DataChannels channels = DataChannels::RGBA8;

    void* referenced_image_handle = nullptr;

    size_t channels_count() const;

    std::string name = "";
};

template<WrappedImageType>
ImageWrapper wrap_image(ImageWrapper::ImageSize size, const void* data = nullptr,
    ImageWrapper::DataChannels channels = ImageWrapper::DataChannels::RGBA8);

template<>
ImageWrapper wrap_image<WrappedImageType::GLTexureHandle>(
    ImageWrapper::ImageSize size, const void* data, ImageWrapper::DataChannels channels);
template<>
ImageWrapper wrap_image<WrappedImageType::ByteArray>(
    ImageWrapper::ImageSize size, const void* data, ImageWrapper::DataChannels channels);

ImageWrapper wrap_image(
    ImageWrapper::ImageSize size, unsigned int gl_texture_handle, ImageWrapper::DataChannels channels);

ImageWrapper wrap_image(
    ImageWrapper::ImageSize size, std::vector<byte> const& byte_texture, ImageWrapper::DataChannels channels);

ImageWrapper wrap_image(
    ImageWrapper::ImageSize size, std::vector<uint32_t> const& byte_texture, ImageWrapper::DataChannels channels);

ImageWrapper wrap_image(
    ImageWrapper::ImageSize size, std::vector<float> const& byte_texture, ImageWrapper::DataChannels channels);

size_t channels_count(ImageWrapper::DataChannels channels);

} // namespace megamol::frontend_resources
