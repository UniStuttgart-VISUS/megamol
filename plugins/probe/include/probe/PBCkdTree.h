
#pragma once

namespace megamol::probe {


class PBC_adaptor {
private:
    std::vector<std::array<float,2>> dat;
    std::array<float,4> bounds;
    bool cycleX = false;
    bool cycleY = false;

    enum applyPBC { X = (1 << 0), Y = (1 << 1)};

public:
    typedef float coord_t;

    PBC_adaptor(std::vector<std::array<float, 2>>& dat, std::array<float, 4> const& bounds)
            : dat(dat)
            , bounds(bounds) {}

    PBC_adaptor(std::vector<std::array<float, 2>> const& dat, std::array<float, 4> const& bounds, bool pbcX, bool pbcY)
            : dat(dat)
            , bounds(bounds)
            , cycleX(pbcX)
            , cycleY(pbcY) {}

    ~PBC_adaptor() {}

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return dat.size();
    }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
    inline coord_t kdtree_distance(const coord_t* p1, const size_t idx_p2, size_t /*size*/) const {
        coord_t const* p2 = get_position(idx_p2);

        coord_t d0 = p1[0] - p2[0];
        coord_t d1 = p1[1] - p2[1];

        const coord_t bboxW = bounds[2] - bounds[0];
        const coord_t bboxH = bounds[3] - bounds[1];


        bool a = std::abs(d0) > 0.5f * bboxW;
        bool b = std::abs(d1) > 0.5f * bboxH;


        const uint32_t flags_dist = ((std::abs(d0) > 0.5f * bboxW) << 0) | ((std::abs(d1) > 0.5f * bboxH) << 1);
        const uint32_t flags_p2 = ((p2[0] > 0.5f * bboxW) << 0) | ((p2[1] > 0.5f * bboxH) << 1);


        if (flags_dist & applyPBC::X && cycleX) {
            const coord_t movedir = (flags_p2 & applyPBC::X) ? -1.0f : 1.0f;
            d0 = p1[0] - (p2[0] + movedir*bboxW);
        }

        if (flags_dist & applyPBC::Y && cycleY) {
            const coord_t movedir = (flags_p2 & applyPBC::Y) ? -1.0f : 1.0f;
            d1 = p1[1] - (p2[1] + movedir * bboxW);
        }

        return d0 * d0 + d1 * d1;
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline coord_t kdtree_get_pt(const size_t idx, int dim) const {
        assert((dim >= 0) && (dim < 2));
        return get_position(idx)[dim];
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo
    //   it again. Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template<class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const {
        return false;
    }

    inline const coord_t* get_position(size_t index) const {

        return dat[index].data();
    }

};

}
