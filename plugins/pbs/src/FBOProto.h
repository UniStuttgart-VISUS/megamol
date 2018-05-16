#pragma once

namespace megamol {
namespace pbs {

template <int DIM> struct vec {
    float coord_[DIM];
    vec(void) = default;
    vec(float coord[DIM]) {
        std::copy(coord, coord + DIM, coord_);
    }
    vec& operator=(vec const& rhs) {
        std::copy(rhs.coord_, rhs.coord_ + DIM, coord_);
        return *this;
    }
    vec& operator=(float coord[DIM]) {
        std::copy(coord, coord + DIM, coord_);
        return *this;
    }
};

template <int DIM> using vec_t = vec<DIM>;

template <int DIM>
struct box {
    vec_t<DIM> lower_;
    vec_t<DIM> upper_;
    box(void) = default;
    box(float lower[DIM], float upper[DIM]) {
        lower_ = lower;
        upper_ = upper;
    }
};

using bbox_t = box<3>;
using viewp_t = box<2>;

enum fbo_color_type : unsigned int {
    RGBAf,
    RGBAu8,
    RGBf,
    RGBu8
};

enum fbo_depth_type : unsigned int {
    Df,
    Du16,
    Du24,
    Du32
};

using data_ptr = char*;

using id_t = unsigned int;

struct fbo_msg_header {
    // node id
    id_t node_id;
    // frame id
    id_t frame_id;
    // obbox
    bbox_t os_bbox;
    // cbbox
    bbox_t cs_bbox;
    // viewport
    viewp_t screen_area;
    // updated viewport
    viewp_t updated_area;
    // fbo color type
    fbo_color_type color_type;
    // fbo depth type
    fbo_depth_type depth_type;
};

using fbo_msg_header_t = fbo_msg_header;

} // end namespace pbs
} // end namespace megamol