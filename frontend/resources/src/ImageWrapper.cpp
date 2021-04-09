/*
 * ImageWrapper.cpp
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */


#include "ImageWrapper.h"

#include <algorithm>
#include <list>

using namespace megamol::frontend_resources;

ImageWrapper::ImageWrapper(ImageSize size, DataChannels channels, WrappedImageType type, const void* data)
    : image_size{size}
    , channels{channels}
    , image_type{type}
{
    borrowed_image_handle = const_cast<void*>(data);
}

size_t ImageWrapper::channels_count() const {
    return megamol::frontend_resources::channels_count(channels);
}

template <>
ImageWrapper megamol::frontend_resources::wrap_image<WrappedImageType::GLTexureHandle>(
    ImageWrapper::ImageSize size,
    const void* data,
    ImageWrapper::DataChannels channels)
{
    return ImageWrapper(size, channels, WrappedImageType::GLTexureHandle, data);
}

template <>
ImageWrapper megamol::frontend_resources::wrap_image<WrappedImageType::ByteArray>(
    ImageWrapper::ImageSize size,
    const void* data,
    ImageWrapper::DataChannels channels)
{
    return ImageWrapper(size, channels, WrappedImageType::ByteArray, data);
}

ImageWrapper megamol::frontend_resources::wrap_image(
    ImageWrapper::ImageSize size,
    unsigned int gl_texture_handle,
    ImageWrapper::DataChannels channels)
{
    return wrap_image<WrappedImageType::GLTexureHandle>(size, reinterpret_cast<void*>(gl_texture_handle), channels);
}

ImageWrapper megamol::frontend_resources::wrap_image(
    ImageWrapper::ImageSize size,
    std::vector<unsigned char> const& byte_texture,
    ImageWrapper::DataChannels channels)
{
    return wrap_image<WrappedImageType::ByteArray>(size, &byte_texture, channels);
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

#define images \
    (*(static_cast< std::list<std::pair<std::string, megamol::frontend_resources::ImageWrapper>>* >(pimpl)))

megamol::frontend_resources::ImageWrapper& megamol::frontend_resources::ImageRegistry::make(std::string const& name) {
    images.push_back({name, megamol::frontend_resources::ImageWrapper{}});
    updates = true;
    return images.back().second;
}
bool megamol::frontend_resources::ImageRegistry::rename(std::string const& old_name, std::string const& new_name) {
    auto find_it = std::find_if(images.begin(), images.end(), [&](auto const& elem) { return elem.first == old_name; });
    if (find_it == images.end())
        return false;

    find_it->first = new_name;
    updates = true;
    return true;
}
bool megamol::frontend_resources::ImageRegistry::remove(std::string const& name) {
    auto find_it = std::find_if(images.begin(), images.end(), [&](auto const& elem) { return elem.first == name; });
    if (find_it == images.end())
        return false;

    images.erase(find_it);
    updates = true;
    return true;
}
std::optional<std::reference_wrapper<megamol::frontend_resources::ImageWrapper const>>
megamol::frontend_resources::ImageRegistry::find(std::string const& name) const {
    auto find_it = std::find_if(images.begin(), images.end(), [&](auto const& elem) { return elem.first == name; });
    if (find_it == images.end())
        return std::nullopt;

    return std::make_optional(std::reference_wrapper<megamol::frontend_resources::ImageWrapper const>{find_it->second});
}
void megamol::frontend_resources::ImageRegistry::iterate_over_entries(
    std::function<void(std::string /*name*/, ImageWrapper const& /*image*/)> const& callback) const
{
    std::for_each(images.begin(), images.end(), [&](auto const& elem) { callback(elem.first, elem.second); });
}
#undef images

