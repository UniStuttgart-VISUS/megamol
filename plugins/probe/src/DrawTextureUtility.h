/*
 * DrawTextureUtility.h
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <cstdint>
#include <tuple>
#include <memory>
#include <vector>
#include "blend2d.h"

namespace megamol {
namespace probe {

class DrawTextureUtility {

public:
    enum GraphType {
        PLOT,
        GLYPH
    };

    DrawTextureUtility() = default;

    ~DrawTextureUtility() = default;

    inline void setResolution(uint32_t width, uint32_t height) {
        this->_pixel_width = width;
        this->_pixel_height = height;
    }

    inline std::pair<uint32_t, uint32_t> getResolution() {
        return std::make_pair(this->_pixel_width, this->_pixel_height);
    }

    inline uint32_t getPixelWidth() const { return this->_pixel_width; }

    inline uint32_t getPixelHeight() const { return this->_pixel_height; }

    inline void setGraphType(GraphType gt) { this->_graph_type = gt; }

    inline GraphType getGraphType() const { return this->_graph_type; }

    template <typename T>
    uint8_t* draw(std::vector<T>& data, T min, T max);

private:

    uint32_t _pixel_width = 0;
    uint32_t _pixel_height = 0;
    GraphType _graph_type = PLOT;
    BLImageData _img_data;
    BLImage _img;

    uint8_t* _pixel_data;

};


template <typename T>
uint8_t* DrawTextureUtility::draw(std::vector<T>& data, T min, T max) {
    _img = BLImage(this->_pixel_width, this->_pixel_height, BL_FORMAT_PRGB32);
    BLContext ctx(_img);

    uint32_t width_halo = this->_pixel_width * 0.1f;
    uint32_t height_halo = this->_pixel_height * 0.1f;

    // fill with red 
    ctx.setFillStyle(BLRgba32(0x88000000));
    ctx.fillAll();

    // Draw sampe data

    // calc clamp on texture
    auto num_data = data.size();

    // first and last positions should be empty
    float step_width = static_cast<T>(this->_pixel_width - 2*width_halo) / static_cast<T>(num_data);

    if (max - min > static_cast<T>(0.0000001)) {

        BLPath path;
        for (uint32_t i = 0; i < num_data - 1; i++) {
            BLLine line(width_halo + step_width * (i + 1),
                height_halo + ((data[i] - min) / (max - min)) * (this->_pixel_height - 2*height_halo),
                width_halo + step_width * (i + 2),
                height_halo + ((data[i + 1] - min) / (max - min)) * (this->_pixel_height - 2*height_halo));
            path.addLine(line);
        }
        // Draw axis
        BLPath yaxis;
        BLLine l1(width_halo, height_halo, width_halo, this->_pixel_height - 2 * height_halo);
        BLLine l2(width_halo, this->_pixel_height - 2 * height_halo, 2 * width_halo, this->_pixel_height - 3 * height_halo);
        yaxis.addLine(l1);
        yaxis.addLine(l2);

        ctx.setCompOp(BL_COMP_OP_SRC_OVER);
        ctx.setStrokeStyle(BLRgba32(0xFF00FF00));
        ctx.setStrokeWidth(5);
        ctx.strokePath(yaxis);



        // add path
        ctx.setCompOp(BL_COMP_OP_SRC_OVER);
        ctx.setStrokeStyle(BLRgba32(0xFF0000FF));
        ctx.setStrokeWidth(5);
        //ctx.setStrokeStartCap(BL_STROKE_CAP_ROUND);
        //ctx.setStrokeEndCap(BL_STROKE_CAP_BUTT);
        ctx.strokePath(path);
    }
    ctx.end();

    if (_img.getData(&this->_img_data) != BL_SUCCESS) {
        vislib::sys::Log::DefaultLog.WriteError("[DrawTextureUtility] Could not receive image data from blend2d.");
    }

    this->_pixel_data = reinterpret_cast<uint8_t*>(this->_img_data.pixelData);

    return this->_pixel_data;
}


} // namespace probe
} // namespace megamol
