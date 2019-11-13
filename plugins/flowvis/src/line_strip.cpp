#include "stdafx.h"
#include "line_strip.h"

#include "glyph_data_call.h"

#include "tsp/tsp.h"

#include "mmcore/Call.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/utility/DataHash.h"

#include "vislib/sys/Log.h"

#include "Eigen/Dense"

#include <memory>
#include <utility>
#include <vector>

namespace megamol {
namespace flowvis {

line_strip::line_strip()
    : line_strip_slot("line_strip", "Line strip connecting the input points")
    , points_slot("points", "Input points")
    , method("method", "Method for connecting the points")
    , points_hash(-1)
    , points_changed(false)
    , line_strip_hash(-1) {

    // Connect output
    this->line_strip_slot.SetCallback(
        glyph_data_call::ClassName(), glyph_data_call::FunctionName(0), &line_strip::get_lines_data);
    this->line_strip_slot.SetCallback(
        glyph_data_call::ClassName(), glyph_data_call::FunctionName(1), &line_strip::get_lines_extent);
    this->MakeSlotAvailable(&this->line_strip_slot);

    // Connect input
    this->points_slot.SetCompatibleCall<glyph_data_call::glyph_data_description>();
    this->MakeSlotAvailable(&this->points_slot);

    // Create parameters
    this->method << new core::param::EnumParam(0);
    this->method.Param<core::param::EnumParam>()->SetTypePair(0, "Point order");
    this->method.Param<core::param::EnumParam>()->SetTypePair(1, "Smallest distance (approx. TSP)");
    this->MakeSlotAvailable(&this->method);
}

line_strip::~line_strip() { this->Release(); }

bool line_strip::create() { return true; }

void line_strip::release() {}

bool line_strip::get_input_data() {
    auto pc_ptr = this->points_slot.CallAs<glyph_data_call>();

    if (pc_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("No input connection set");

        return false;
    }

    auto& pc = *pc_ptr;

    if (!pc(0)) {
        vislib::sys::Log::DefaultLog.WriteError("Error getting input points");

        return false;
    }

    if (pc.DataHash() != this->points_hash) {
        this->points = pc.get_point_vertices();

        this->points_hash = pc.DataHash();
        this->points_changed = true;
    }

    return true;
}

bool line_strip::get_input_extent() {
    auto pc_ptr = this->points_slot.CallAs<glyph_data_call>();

    if (pc_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("No input conection set");

        return false;
    }

    auto& pc = *pc_ptr;

    if (!pc(1)) {
        vislib::sys::Log::DefaultLog.WriteError("Error getting extents for the points");

        return false;
    }

    this->bounding_rectangle = pc.get_bounding_rectangle();

    return true;
}

void line_strip::create_lines_input_order(const std::vector<Eigen::Vector2f>& points) {
    for (std::size_t point_index = 0; point_index < points.size() - 1; ++point_index) {
        this->lines.push_back(std::make_pair(static_cast<float>(point_index),
            std::vector<Eigen::Vector2f>{points[point_index], points[point_index + 1]}));
    }
}

void line_strip::create_lines_tsp(const std::vector<Eigen::Vector2f>& points) {
    // Create seed lines between points, approximating the travelling salesman problem
    const auto polygon_order =
        thirdparty::tsp::Genetic(std::make_shared<thirdparty::tsp::Graph>(points), 10, 1000, 5).run();

    for (std::size_t point_index = 0; point_index < polygon_order.size() - 1; ++point_index) {
        this->lines.push_back(std::make_pair(static_cast<float>(point_index),
            std::vector<Eigen::Vector2f>{points[polygon_order[point_index]], points[polygon_order[point_index + 1]]}));
    }
}

bool line_strip::get_lines_data(core::Call& call) {
    auto& gdc = static_cast<glyph_data_call&>(call);

    if (!get_input_data()) {
        return false;
    }

    if (this->points_changed || this->method.IsDirty()) {
        this->method.ResetDirty();

        // Get points
        std::vector<Eigen::Vector2f> points;

        for (std::size_t cp_index = 0; cp_index < this->points->size(); cp_index += 2) {
            points.push_back(Eigen::Vector2f((*this->points)[cp_index], (*this->points)[cp_index + 1]));
        }

        // Connect points
        this->lines.clear();

        if (points.size() > 1) {
            this->lines.reserve(points.size() - 1);

            switch (this->method.Param<core::param::EnumParam>()->Value()) {
            case 0:
                create_lines_input_order(points);
                break;
            case 1:
                create_lines_tsp(points);
            }
        }

        // Set new hash
        this->line_strip_hash = core::utility::DataHash(
            this->line_strip_hash, this->points_hash, this->method.Param<core::param::EnumParam>()->Value());
    }

    this->points_changed = false;

    if (gdc.DataHash() != this->line_strip_hash) {
        gdc.clear();

        for (const auto& line : this->lines) {
            gdc.add_line(line.second, line.first);
        }

        gdc.SetDataHash(this->line_strip_hash);
    }

    return true;
}

bool line_strip::get_lines_extent(core::Call& call) {
    auto& gdc = static_cast<glyph_data_call&>(call);

    if (!get_input_extent()) {
        return false;
    }

    gdc.set_bounding_rectangle(this->bounding_rectangle);

    return true;
}

} // namespace flowvis
} // namespace megamol
