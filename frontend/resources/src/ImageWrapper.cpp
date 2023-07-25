/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "ImageWrapper.h"

#include <algorithm>
#include <list>

using namespace megamol::frontend_resources;

ImageWrapper::ImageWrapper(ImageSize size, DataChannels channels, WrappedImageType type, const void* data)
        : size{size}
        , channels{channels}
        , type{type} {
    referenced_image_handle = const_cast<void*>(data);
}

ImageWrapper::ImageWrapper(std::string const& name) : name{name} {}

size_t ImageWrapper::channels_count() const {
    return megamol::frontend_resources::channels_count(channels);
}

template<>
ImageWrapper megamol::frontend_resources::wrap_image<WrappedImageType::GLTexureHandle>(
    ImageWrapper::ImageSize size, const void* data, ImageWrapper::DataChannels channels) {
    return ImageWrapper(size, channels, WrappedImageType::GLTexureHandle, data);
}

template<>
ImageWrapper megamol::frontend_resources::wrap_image<WrappedImageType::ByteArray>(
    ImageWrapper::ImageSize size, const void* data, ImageWrapper::DataChannels channels) {
    return ImageWrapper(size, channels, WrappedImageType::ByteArray, data);
}

ImageWrapper megamol::frontend_resources::wrap_image(
    ImageWrapper::ImageSize size, unsigned int gl_texture_handle, ImageWrapper::DataChannels channels) {
    return wrap_image<WrappedImageType::GLTexureHandle>(size, reinterpret_cast<void*>(gl_texture_handle), channels);
}

ImageWrapper megamol::frontend_resources::wrap_image(
    ImageWrapper::ImageSize size, std::vector<byte> const& byte_texture, ImageWrapper::DataChannels channels) {
    return wrap_image<WrappedImageType::ByteArray>(size, &byte_texture, channels);
}

ImageWrapper megamol::frontend_resources::wrap_image(
    ImageWrapper::ImageSize size, std::vector<uint32_t> const& byte_texture, ImageWrapper::DataChannels channels) {
    return wrap_image(size, reinterpret_cast<std::vector<byte> const&>(byte_texture), channels);
}

ImageWrapper megamol::frontend_resources::wrap_image(
    ImageWrapper::ImageSize size, std::vector<float> const& byte_texture, ImageWrapper::DataChannels channels) {
    return wrap_image(size, reinterpret_cast<std::vector<byte> const&>(byte_texture), channels);
}

size_t megamol::frontend_resources::channels_count(ImageWrapper::DataChannels channels) {
    switch (channels) {
    case ImageWrapper::DataChannels::RGB8:
        return 3;
        break;
    case ImageWrapper::DataChannels::RGBA8:
        return 4;
        break;
    default:
        return 0;
    }
}
