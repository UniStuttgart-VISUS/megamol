#pragma once

#include "ANN/ANN.h"

namespace megamol {
namespace datatools {
namespace clustering {


class ann_instance {
public:
    static ann_instance& get_instance() {
        static ann_instance instance;
        return instance;
    }

    ANNkd_tree get_kdtree(ANNpointArray pa, int n, int d) { return ANNkd_tree(pa, n, d); }

private:
    ann_instance() = default;
    ~ann_instance() { annClose(); };
    ann_instance(ann_instance const& rhs) = delete;
    ann_instance& operator=(ann_instance const& rhs) = delete;
};

class ann_point {
public:
    ann_point(int dim) : ptr_(annAllocPt(dim)) {}

    template <typename T> ann_point(int dim, T* data) : ptr_(annAllocPt(dim)) {
        for (int d = 0; d < dim; ++d) {
            ptr_[d] = data[d];
        }
    }

    ~ann_point() { annDeallocPt(ptr_); }

    operator ANNpoint() { return ptr_; }

private:
    ANNpoint ptr_;
};

class ann_points {
public:
    ann_points(int n, int dim) : ptr_(annAllocPts(n, dim)) {
        for (int idx = 0; idx < n; ++idx) {
            for (int d = 0; d < dim; ++d) {
                ptr_[idx][d] = static_cast<ANNcoord>(0);
            }
        }
    }

    template <typename T> ann_points(int n, int dim, T* data) : ptr_(annAllocPts(n, dim)) {
        for (int idx = 0; idx < n; ++idx) {
            for (int d = 0; d < dim; ++d) {
                ptr_[idx][d] = static_cast<ANNcoord>(data[idx * dim + d]);
            }
        }
    }

    ~ann_points() { annDeallocPts(ptr_); }

    operator ANNpointArray() { return ptr_; }

private:
    ANNpointArray ptr_;
};

} // namespace clustering
} // namespace datatools
} // namespace megamol
