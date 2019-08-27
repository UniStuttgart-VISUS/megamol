#pragma once

#include <cmath>
#include <memory>


namespace megamol {
namespace remote {

template <int DIM> struct vec {
    static_assert(DIM > 0, "Zero dimensional vector not allowed");
    std::unique_ptr<float[]> coord_;
    vec(void) : coord_{new float[DIM]} {};
    vec(float coord[DIM]) : vec{} { std::copy(coord, coord + DIM, coord_.get()); }
    vec(vec const& rhs) : vec{} { std::copy(rhs.coord_.get(), rhs.coord_.get() + DIM, coord_.get()); }
    vec& operator=(vec const& rhs) {
        std::copy(rhs.coord_.get(), rhs.coord_.get() + DIM, coord_.get());
        return *this;
    }
    vec& operator=(float coord[DIM]) {
        std::copy(coord, coord + DIM, coord_.get());
        return *this;
    }
    float& operator[](size_t idx) { return coord_[idx]; }
};

template <int DIM> using vec_t = vec<DIM>;

template <int DIM> struct box {
    vec_t<DIM> lower_;
    vec_t<DIM> upper_;
    box(void) : lower_{}, upper_{} {};

    box(float lower[DIM], float upper[DIM]) : lower_{}, upper_{} {
        lower_ = lower;
        upper_ = upper;
    }
    box& operator=(float rhs[2 * DIM]) {
        for (int d = 0; d < DIM; ++d) {
            lower_[d] = rhs[d];
        }
        for (int d = 0; d < DIM; ++d) {
            upper_[d] = rhs[d + DIM];
        }
        return *this;
    }
    float volume() {
        float ret{0.0f};
        for (int i = 0; i < DIM; ++i) {
            ret *= upper_[i] - lower_[i];
        }
        return ret;
    }
    box& unite(box& rhs) {
        for (int d = 0; d < DIM; ++d) {
            lower_[d] = fmin(lower_[d], rhs.lower_[d]);
            upper_[d] = fmax(upper_[d], rhs.upper_[d]);
        }
        return *this;
    }
    float& operator[](int idx) {
        if (idx < DIM) {
            return lower_[idx];
        } else {
            return upper_[idx - DIM];
        }
    }
};

using bbox_t = box<3>;
using viewp_t = box<2>;

enum fbo_color_type : unsigned int { RGBAf, RGBAu8, RGBf, RGBu8 };

enum fbo_depth_type : unsigned int { Df, Du16, Du24, Du32 };

using data_ptr = char*;

using id_t = unsigned int;

struct fbo_msg_header {
    // node id
    id_t node_id;
    // frame id
    id_t frame_id;
    // obbox
    float os_bbox[6];
    // cbbox
    float cs_bbox[6];
    // frame_times
    float frame_times[2]; /// [0] requested time, [1] time frames count
    // cam_params
    float cam_params[9];  /// [0]-[2] position, [3]-[5] up, [6]-[8] lookat
    // viewport
    int screen_area[4];
    // updated viewport
    int updated_area[4];
    // fbo color type
    fbo_color_type color_type;
    // fbo depth type
    fbo_depth_type depth_type;
    // color buf size
    size_t color_buf_size;
    // depth buf size
    size_t depth_buf_size;
};

using fbo_msg_header_t = fbo_msg_header;

struct fbo_msg {
    fbo_msg() = default;

    explicit fbo_msg(fbo_msg_header_t&& header, std::vector<char>&& col, std::vector<char>&& depth)
        : fbo_msg_header{std::forward<fbo_msg_header_t>(header)}
        , color_buf{std::forward<std::vector<char>>(col)}
        , depth_buf{std::forward<std::vector<char>>(depth)} {}

    fbo_msg_header_t fbo_msg_header;
    std::vector<char> color_buf;
    std::vector<char> depth_buf;
};

using fbo_msg_t = fbo_msg;

} // end namespace remote
} // end namespace megamol
