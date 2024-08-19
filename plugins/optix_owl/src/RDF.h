#pragma once

#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>

#include <nanoflann.hpp>

#include <owl/common/math/vec.h>

namespace megamol::optix_owl {
    using namespace owl::common;
class OWLPointcloud {
private:
    std::vector<vec3f>* data;
    vec3f lower, upper;

public:
    typedef float coord_t;

    OWLPointcloud(std::vector<vec3f>* data) : data(data) {
        // intentionally empty
        auto xfit = std::minmax_element(
            data->begin(), data->end(), [](auto const& lhs, auto const& rhs) { return lhs[0] < rhs[0]; });
        auto yfit = std::minmax_element(
            data->begin(), data->end(), [](auto const& lhs, auto const& rhs) { return lhs[1] < rhs[1]; });
        auto zfit = std::minmax_element(
            data->begin(), data->end(), [](auto const& lhs, auto const& rhs) { return lhs[2] < rhs[2]; });

        lower.x = xfit.first->x;
        lower.y = yfit.first->y;
        lower.z = zfit.first->z;

        upper.x = xfit.second->x;
        upper.y = yfit.second->y;
        upper.z = zfit.second->z;
    }
    ~OWLPointcloud() {
        // intentionally empty
    }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return data->size();
    }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
    inline coord_t kdtree_distance(const coord_t* p1, const size_t idx_p2, size_t /*size*/) const {
        auto const& p2 = data->operator[](idx_p2);

        const coord_t d0 = p1[0] - p2[0];
        const coord_t d1 = p1[1] - p2[1];
        const coord_t d2 = p1[2] - p2[2];

        return d0 * d0 + d1 * d1 + d2 * d2;
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline coord_t kdtree_get_pt(const size_t idx, int dim) const {
        return data->operator[](idx)[dim];
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template<class BBOX>
    bool kdtree_get_bbox(BBOX& bb) const {
        //return false;

        bb[0].low = lower.x;
        bb[0].high = upper.x;
        bb[1].low = lower.y;
        bb[1].high = upper.y;
        bb[2].low = lower.z;
        bb[2].high = upper.z;
        return true;
    }
};

class RDF {
public:
    using kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, OWLPointcloud>,
        OWLPointcloud, 3, size_t>;

    RDF(std::shared_ptr<std::vector<vec3f>> org_data, std::shared_ptr<std::vector<vec3f>> new_data);
    ~RDF() = default;

    std::tuple<std::vector<float>, std::vector<float>> BuildHistogram(float cut_off, unsigned int num_bins);

private:
    std::shared_ptr<std::vector<vec3f>> org_data_;
    std::shared_ptr<OWLPointcloud> org_Pts_;
    std::shared_ptr<kd_tree_t> org_particleTree_;

    std::shared_ptr<std::vector<vec3f>> new_data_;
    std::shared_ptr<OWLPointcloud> new_Pts_;
    std::shared_ptr<kd_tree_t> new_particleTree_;
};
} // namespace megamol::optix_owl
