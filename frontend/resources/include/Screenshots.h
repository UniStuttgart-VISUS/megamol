/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <filesystem>
#include <vector>

#include "GL_STUB.h"

namespace megamol::frontend_resources {

struct ScreenshotImageData {
    struct Pixel {
        std::uint8_t r = 255;
        std::uint8_t g = 0;
        std::uint8_t b = 0;
        std::uint8_t a = 255;
    };

    size_t width = 0;
    size_t height = 0;

    // row-major image starting at bottom-left pixel
    // as in https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glReadPixels.xhtml
    std::vector<Pixel> image;

    // to write PNGs we need to provide rows
    std::vector<Pixel*> rows;
    std::vector<Pixel*> flipped_rows;

    void resize(const size_t width, const size_t height) {
        this->width = width;
        this->height = height;

        image.resize(width * height);
        rows.resize(height);
        flipped_rows.resize(height);

        for (size_t i = 0; i < height; i++) {
            const auto row_address = image.data() + i * width;
            rows[i] = row_address;
            flipped_rows[height - (1 + i)] = row_address;
        }
    }
};

class IScreenshotSource {
public:
    virtual ScreenshotImageData const& take_screenshot() const = 0;

    ~IScreenshotSource() = default;
};

class IImageDataWriter {
public:
    bool write_screenshot(IScreenshotSource const& image_source, std::filesystem::path const& filename) const {
        return this->write_image(image_source.take_screenshot(), filename);
    }

    virtual bool write_image(ScreenshotImageData const& image, std::filesystem::path const& filename) const = 0;

    ~IImageDataWriter() = default;
};

struct ImageWrapper;
class ImageWrapperScreenshotSource : public IScreenshotSource {
public:
    ImageWrapperScreenshotSource() = default;
    ImageWrapperScreenshotSource(ImageWrapper const& image);

    ScreenshotImageData const& take_screenshot() const override;

private:
    ImageWrapper* m_image = nullptr;
};

class GLScreenshotSource : public IScreenshotSource {
public:
    enum ReadBuffer {
        FRONT = 0x0404,                              // GL_FRONT
        BACK = 0x0405,                               // GL_BACK
        COLOR_ATTACHMENT0 = 0x8CE0,                  // GL_COLOR_ATTACHMENT0
        COLOR_ATTACHMENT1 = (COLOR_ATTACHMENT0 + 1), // GL_COLOR_ATTACHMENT1
        COLOR_ATTACHMENT2 = (COLOR_ATTACHMENT0 + 2), // GL_COLOR_ATTACHMENT2
        COLOR_ATTACHMENT3 = (COLOR_ATTACHMENT0 + 3), // GL_COLOR_ATTACHMENT3
    };

    void set_read_buffer(ReadBuffer buffer) GL_VSTUB();

    ScreenshotImageData const& take_screenshot() const override GL_STUB({});

private:
    ReadBuffer m_read_buffer = FRONT;
};

class ScreenshotImageDataToPNGWriter : public IImageDataWriter {
public:
    bool write_image(ScreenshotImageData const& image, std::filesystem::path const& filename) const override;
};


} // namespace megamol::frontend_resources
