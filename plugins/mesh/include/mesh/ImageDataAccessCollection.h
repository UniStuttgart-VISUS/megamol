/*
 * ImageDataAccessCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#ifndef IMAGE_DATA_ACCESS_COLLECTION_H_INCLUDED
#define IMAGE_DATA_ACCESS_COLLECTION_H_INCLUDED

#include <vector>
#include "mesh.h"

namespace megamol {
namespace mesh {

class MESH_API ImageDataAccessCollection {
public:
    enum TextureFormat { R32F, RGB32F, RGBA32F, R8, RGB8, RGBA8 };

    static constexpr unsigned int convertToGLInternalFormat(TextureFormat format) {
        unsigned int retval = 0;

        switch (format) {
        case R32F:
            retval = 0x822E;
            break;
        case RGB32F:
            retval = 0x8815;
            break;
        case RGBA32F:
            retval = 0x8814;
            break;
        case R8:
            retval = 0x8229;
            break;
        case RGB8:
            retval = 0x8051;
            break;
        case RGBA8:
            retval = 0x8058;
            break;
        default:
            break;
        }

        return retval;
    }

    static constexpr TextureFormat covertToTextureFormat(unsigned int gl_internal_format) {
        TextureFormat retval = RGB8; // TODO default to something more reasonable

        switch (gl_internal_format) {
        case 0x822E:
            retval = R32F;
            break;
        case 0x8815:
            retval = RGB32F;
            break;
        case 0x8814:
            retval = RGBA32F;
            break;
        case 0x8229:
            retval = R8;
            break;
        case 0x1907:
            retval = RGB8;
            break;
        case 0x1908:
            retval = RGBA8;
            break;
        default:
            break;
        }

        return retval;
    }

    struct Image {

        uint8_t* data;
        size_t byte_size;

        TextureFormat format;
        int width;
        int height;

        //int levels;
    };

    ImageDataAccessCollection() = default;
    ~ImageDataAccessCollection() = default;

    /**
    * Add an accessor for an image. Caution! Takes no ownership of image data, simply provides acces via raw pointer.
    */
    void addImage(TextureFormat format, int width, int height, uint8_t* data, size_t byte_size);

    // TODO delete functionality

    std::vector<Image>& accessImages() { return m_images; }

private:

    std::vector<Image> m_images;
};


inline void ImageDataAccessCollection::addImage(
    TextureFormat format,
    int width,
    int height,
    uint8_t* data,
    size_t byte_size)
{
    Image img;
    img.format = format;
    img.width = width;
    img.height = height;
    img.data = data;
    img.byte_size = byte_size;

    m_images.emplace_back(std::move(img));
}

}
}

#endif // !IMAGE_DATA_ACCESS_COLLECTION_H_INCLUDED
