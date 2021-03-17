/*
 * Screenshots.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <vector>
#include <string>

namespace megamol {
namespace frontend_resources {

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
    virtual ScreenshotImageData take_screenshot() const = 0;

    ~IScreenshotSource() = default;
};

class IImageDataWriter {
public:
    bool write_screenshot(IScreenshotSource const& image_source, std::string const& filename) const {
        return this->write_image(std::move(image_source.take_screenshot()), filename);
    }

    virtual bool write_image(ScreenshotImageData image, std::string const& filename) const = 0;

    ~IImageDataWriter() = default;
};

class GLScreenshotSource : public IScreenshotSource {
public:
    enum ReadBuffer { FRONT, BACK, COLOR_ATT0, COLOR_ATT1, COLOR_ATT2, COLOR_ATT3};

    void set_read_buffer(ReadBuffer buffer);

    ScreenshotImageData take_screenshot() const override;

private:
    ReadBuffer m_read_buffer = FRONT;
};

class ScreenshotImageDataToPNGWriter : public IImageDataWriter {
public:
    bool write_image(ScreenshotImageData image, std::string const& filename) const override;
};


} /* end namespace frontend_resources */
} /* end namespace megamol */
