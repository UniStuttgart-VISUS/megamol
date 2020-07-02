/*
 * DrawTextureUtility.h
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>
#include "blend2d.h"


namespace megamol {
namespace probe {

class DrawTextureUtility {

public:
    enum GraphType { PLOT, GLYPH, LINEAR, RADARGLYPH };

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

    template <typename T> uint8_t* draw(std::vector<T>& data, T min, T max);

    template <typename T> uint8_t* draw(std::vector<T>& data, std::array<float,3> probe_direction);

private:
    template <typename T> void drawPlot(std::vector<T>& data, T min, T max);
    template <typename T> void drawStar(std::vector<T>& data, T min, T max);
    template <typename T> void drawLinear(std::vector<T>& data, T min, T max);
    template <typename T> void drawRadarGlyph(std::vector<T>& data, std::array<float, 3> probe_direction);

    uint32_t _pixel_width = 0;
    uint32_t _pixel_height = 0;
    GraphType _graph_type = PLOT;
    BLImageData _img_data;
    BLImage _img;
    BLContext _ctx;

    uint8_t* _pixel_data;
};


template <typename T> uint8_t* DrawTextureUtility::draw(std::vector<T>& data, T min, T max) {
    _img = BLImage(this->_pixel_width, this->_pixel_height, BL_FORMAT_PRGB32);
    _ctx = BLContext(_img);


    // fill with BG color
    _ctx.setCompOp(BL_COMP_OP_SRC_COPY);
    _ctx.setFillStyle(BLRgba32(0x00000000));
    _ctx.fillAll();

    // Draw sampe data
    if (max - min > static_cast<T>(0.0000001)) {
        if (this->_graph_type == PLOT) {
            this->drawPlot(data, min, max);
        } else if (this->_graph_type == GLYPH) {
            this->drawStar(data, min, max);
        } else if (this->_graph_type == LINEAR) {
            this->drawLinear(data,min,max);
        }

    }
    _ctx.end();

    // extract image
    if (_img.getData(&this->_img_data) != BL_SUCCESS) {
        vislib::sys::Log::DefaultLog.WriteError("[DrawTextureUtility] Could not receive image data from blend2d.");
    }

    this->_pixel_data = reinterpret_cast<uint8_t*>(this->_img_data.pixelData);

    return this->_pixel_data;
}

template <typename T>
inline uint8_t* DrawTextureUtility::draw(std::vector<T>& data, std::array<float, 3> probe_direction) {

    _img = BLImage(this->_pixel_width, this->_pixel_height, BL_FORMAT_PRGB32);
    _ctx = BLContext(_img);


    // fill with BG color
    _ctx.setCompOp(BL_COMP_OP_SRC_COPY);
    _ctx.setFillStyle(BLRgba32(0x00000000));
    _ctx.fillAll();

    // Draw sample data
    
    if (this->_graph_type == RADARGLYPH) {
        this->drawRadarGlyph(data, probe_direction);
    }
    
    _ctx.end();

    // extract image
    if (_img.getData(&this->_img_data) != BL_SUCCESS) {
        vislib::sys::Log::DefaultLog.WriteError("[DrawTextureUtility] Could not receive image data from blend2d.");
    }

    this->_pixel_data = reinterpret_cast<uint8_t*>(this->_img_data.pixelData);

    return this->_pixel_data;

    return NULL;
}

template <typename T> void DrawTextureUtility::drawPlot(std::vector<T>& data, T min, T max) {

    uint32_t width_halo = this->_pixel_width * 0.1f;
    uint32_t height_halo = this->_pixel_height * 0.1f;
    // calc clamp on texture
    auto num_data = data.size();
    // first and last positions should be empty
    float step_width = static_cast<T>(this->_pixel_width - 2 * width_halo) / static_cast<T>(num_data);

    BLPath path;
    for (uint32_t i = 0; i < num_data - 1; i++) {
        BLLine line(width_halo + step_width * (i + 1),
            height_halo + ((data[i] - min) / (max - min)) * (this->_pixel_height - 2 * height_halo),
            width_halo + step_width * (i + 2),
            height_halo + ((data[i + 1] - min) / (max - min)) * (this->_pixel_height - 2 * height_halo));
        path.addLine(line);
    }
    // Draw axis
    BLPath yaxis;
    BLLine l1(width_halo, height_halo, width_halo, this->_pixel_height - 2 * height_halo);
    BLLine l2(width_halo, this->_pixel_height - 2 * height_halo, 2 * width_halo, this->_pixel_height - 3 * height_halo);
    yaxis.addLine(l1);
    yaxis.addLine(l2);

    _ctx.setCompOp(BL_COMP_OP_SRC_COPY);
    _ctx.setStrokeStyle(BLRgba32(0xFF00FF00));
    _ctx.setStrokeWidth(5);
    _ctx.strokePath(yaxis);


    // add path
    _ctx.setCompOp(BL_COMP_OP_SRC_COPY);
    _ctx.setStrokeStyle(BLRgba32(0xFF0000FF));
    _ctx.setStrokeWidth(5);
    // ctx.setStrokeStartCap(BL_STROKE_CAP_ROUND);
    // ctx.setStrokeEndCap(BL_STROKE_CAP_BUTT);
    _ctx.strokePath(path);
}

template <typename T> void DrawTextureUtility::drawStar(std::vector<T>& data, T min, T max) {

    uint32_t width_halo = this->_pixel_width * 0.1f;
    uint32_t height_halo = this->_pixel_height * 0.1f;
    std::array<uint32_t, 2> center = {this->_pixel_width / 2, this->_pixel_height / 2};
    // calc clamp on texture
    auto num_data = data.size();

    auto angle_step = 2 * 3.14159265359 / num_data;
    auto angle_offset = angle_step * 0.5 - 3.14159265359 * 0.5;

    auto max_radius = std::min(this->_pixel_width, this->_pixel_height) / 2 - std::min(width_halo, height_halo);
    auto axis_radius = std::min(this->_pixel_width, this->_pixel_height) / 2;


    BLPath axis;
    BLPath max_polygon;
    BLPath data_polygon;
    for (uint32_t i = 0; i < num_data; i++) {

        if (i == 0) {
            data_polygon.moveTo(center[0] + ((data[i] - min) / (max - min)) * max_radius * std::cos(i * angle_step + angle_offset),
                center[1] + ((data[i] - min) / (max - min)) * max_radius * std::sin(i * angle_step + angle_offset));
            max_polygon.moveTo(center[0] + max_radius * std::cos(i * angle_step + angle_offset),
                center[1] + max_radius* std::sin(i * angle_step + angle_offset));
        } else {
            data_polygon.lineTo(center[0] + ((data[i] - min) / (max - min)) * max_radius * std::cos(i * angle_step + angle_offset),
                center[1] + ((data[i] - min) / (max - min)) * max_radius * std::sin(i * angle_step + angle_offset));
            max_polygon.lineTo(
                center[0] + max_radius * std::cos(i * angle_step + angle_offset), center[1] + max_radius * std::sin(i * angle_step + angle_offset));
            }

        // Draw axes
        BLLine axis_line(center[0], center[1], center[0] + axis_radius * std::cos(i * angle_step + angle_offset),
            center[1] + axis_radius * std::sin(i * angle_step + angle_offset));
        axis.addLine(axis_line);
    }
    //max_polygon.lineTo(center[0] + max_radius, center[1]); 

    BLPath cut;
    cut.moveTo(center[0], center[1]);
    cut.lineTo(center[0] + axis_radius * std::cos(angle_offset),
        center[1] + axis_radius * std::sin(angle_offset));

    cut.lineTo(center[0] + axis_radius * std::cos((num_data - 1) * angle_step + angle_offset),
        center[1] + axis_radius * std::sin((num_data - 1) * angle_step + angle_offset));

    // Max Polygon Filling
    _ctx.setCompOp(BL_COMP_OP_SRC_COPY);
    //_ctx.setFillStyle(BLRgba32(0x66F3DF92));
    _ctx.setFillStyle(BLRgba32(0xCC2C0D0E));
    //_ctx.setFillStyle(BLRgba32(0xBBDABABE));
    _ctx.fillPath(max_polygon);

    // Star Axis
    _ctx.setStrokeStyle(BLRgba32(0xFF000000));
    _ctx.setStrokeWidth(2);
    _ctx.strokePath(axis);

    // Max Polygon Stroke
    _ctx.setStrokeStyle(BLRgba32(0xFFF3DF92));
    _ctx.setStrokeWidth(2);
    //_ctx.strokePath(max_polygon);

    // Data Stroke
    _ctx.setStrokeStyle(BLRgba32(0xFF0000FF));
    _ctx.setStrokeWidth(7);
    //_ctx.strokePath(data_polygon);

    // Data Filling
    //_ctx.setFillStyle(BLRgba32(0x660000FF));
    _ctx.setFillStyle(BLRgba32(0xDD00B9FF));
    //_ctx.setFillStyle(BLRgba32(0xBBd3b180));
    _ctx.fillPath(data_polygon);

    // Solid Cut
    //_ctx.setFillStyle(BLRgba32(0xFFF3DF92));
    _ctx.setFillStyle(BLRgba32(0xFF2A5F3B));
    _ctx.fillPath(cut);

}

template <typename T> void DrawTextureUtility::drawLinear(std::vector<T>& data, T min, T max) {
    
    uint32_t width_halo = this->_pixel_width * 0.2f;
    uint32_t height_halo = this->_pixel_height * 0.1f;
    std::array<uint32_t, 2> start = {this->_pixel_width / 2, height_halo};
    // calc clamp on texture
    auto num_data = data.size();

    auto linear_step = (this->_pixel_height - 2*height_halo) / num_data;

    auto max_linear = std::min(this->_pixel_width, this->_pixel_height) / 2 - std::min(width_halo, height_halo);

    BLPath axis;
    
    BLPath data_polygon;
    for (uint32_t i = 0; i < num_data; i++) {

        if (i == 0) {
            data_polygon.moveTo(start[0] + ((data[i] - min) / (max - min)) * max_linear,
                start[1]);
        } else {
            data_polygon.lineTo(start[0] + ((data[i] - min) / (max - min)) * max_linear,
                start[1] + i * linear_step );
        }

        // Draw axes
        BLLine axis_line(start[0] - max_linear, start[1] + i * linear_step, start[0] + max_linear,
            start[1] + i * linear_step);
        axis.addLine(axis_line);
    }
    // max_polygon.lineTo(center[0] + max_radius, center[1]);

    //BLPath max_polygon;

    //// Max Polygon Filling
    //_ctx.setCompOp(BL_COMP_OP_SRC_COPY);
    //_ctx.setFillStyle(BLRgba32(0x66F3DF92));
    //_ctx.fillPath(max_polygon);

    BLLine y(start[0], start[1], start[0], start[1] + num_data * linear_step);
    axis.addLine(y);

    // Star Axis
    _ctx.setStrokeStyle(BLRgba32(0xFF000000));
    _ctx.setStrokeWidth(2);
    _ctx.strokePath(axis);

    // Data Stroke
    _ctx.setStrokeStyle(BLRgba32(0xFF0000FF));
    _ctx.setStrokeWidth(7);
    _ctx.strokePath(data_polygon);

    // Data Filling
    _ctx.setFillStyle(BLRgba32(0x660000FF));
    _ctx.fillPath(data_polygon);

}

template <typename T>
inline void DrawTextureUtility::drawRadarGlyph(std::vector<T>& data, std::array<float, 3> probe_direction) {

    uint32_t width_halo = this->_pixel_width * 0.1f;
    uint32_t height_halo = this->_pixel_height * 0.1f;
    std::array<uint32_t, 2> center = {this->_pixel_width / 2, this->_pixel_height / 2};
    // calc clamp on texture
    auto num_data = data.size();

    auto max_radius = std::min(this->_pixel_width, this->_pixel_height) / 2 - std::min(width_halo, height_halo);
    auto radius_steps = max_radius / num_data;

    BLCircle background(center[0], center[1], max_radius);
    // Max Polygon Filling
    _ctx.setCompOp(BL_COMP_OP_SRC_COPY);
    //_ctx.setFillStyle(BLRgba32(0x66F3DF92));
    //_ctx.setFillStyle(BLRgba32(0xCC2C0D0E));
    //_ctx.setFillStyle(BLRgba32(0xBBDABABE));
    _ctx.setFillStyle(BLRgba32(0x00FFFFFF));
    _ctx.fillCircle(background); 

    BLArc border(center[0], center[1], max_radius, max_radius, 0.0, 2.0 * 3.14159);
    //_ctx.setStrokeStyle(BLRgba32(0xFFFF00FF));
    _ctx.setStrokeStyle(BLRgba32(0xFFFFFFFF));
    _ctx.setStrokeWidth(2);
    //_ctx.strokeArc(border);

    auto radius = 0.0f;
    for (uint32_t i = 0; i < num_data; i++) {
        radius += radius_steps;

        std::array<float, 3> direction = {std::get<0>(data[i]), std::get<1>(data[i]), std::get<2>(data[i])};
        float l = std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
        direction[0] /= l;
        direction[1] /= l;
        direction[2] /= l;
        float length = std::get<3>(data[i]);


        // project data direction onto plane given by probe direction?
        //float dot_product =
        //    probe_direction[0] * direction[0] + probe_direction[1] * direction[1] + probe_direction[2] * direction[2];

        // for now, try projection onto z-plane for billboards
        std::array<float, 2> proj_dir = {direction[0], direction[1]};
        float proj_length = std::sqrt(proj_dir[0] * proj_dir[0] + proj_dir[1] * proj_dir[1]);
        proj_dir[0] /= proj_length;
        proj_dir[1] /= proj_length;

        float angle = proj_dir[1] > 0.0f ? acos(proj_dir[0]) : -acos(proj_dir[0]);

        double arc_length = std::max(direction[2] * 2.0f * 3.14159f, (5.0f / 360.0f) * (2.0f * 3.14159f) );

        //arc_length = 50.0 * arc_length / (2.0f * 3.14159f * radius);
        BLArc arc(center[0], center[1], radius, radius, angle - arc_length / 2.0f, arc_length);

        //_ctx.setStrokeStyle(BLRgba32(0xFFFF00FF));
        _ctx.setStrokeStyle(BLRgba32(length*255,0,0));
        _ctx.setStrokeWidth( std::max(2.0 , radius_steps * (1.0 - (arc_length / (2.0 * 3.14159) ) ) ) );
        _ctx.strokeArc(arc);
    }

    //BLCircle background(150, 150, 120);

    //  // Data Stroke
    //  _ctx.setStrokeStyle(BLRgba32(0xFF0000FF));
    //  _ctx.setStrokeWidth(7);
    //  //_ctx.strokePath(data_polygon);
    //  
    //  // Data Filling
    //  //_ctx.setFillStyle(BLRgba32(0x660000FF));
    //  _ctx.setFillStyle(BLRgba32(0xDD00B9FF));
    //  //_ctx.setFillStyle(BLRgba32(0xBBd3b180));
    //  _ctx.fillPath(data_polygon);
    //  
    //  // Solid Cut
    //  //_ctx.setFillStyle(BLRgba32(0xFFF3DF92));
    //  _ctx.setFillStyle(BLRgba32(0xFF2A5F3B));
    //  _ctx.fillPath(cut);
}



} // namespace probe
} // namespace megamol
