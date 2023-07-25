/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "ImageWrapper.h"
#include "ImageWrapper_Conversion_Helpers.hpp"

#include <vector>

namespace megamol::frontend_resources {

struct byte_texture {
    std::vector<byte> texture;
    std::vector<byte>* texture_ptr = nullptr;
    bool texture_owned = false;
    ImageWrapper* image_wrapper_ptr = nullptr;

    ImageWrapper::ImageSize size;

    byte_texture(ImageWrapper const& image);

    byte_texture& operator=(ImageWrapper const& image);

    std::vector<byte> const& as_byte_vector();

private:
    void from_image(ImageWrapper const& image);
};

} // namespace megamol::frontend_resources
