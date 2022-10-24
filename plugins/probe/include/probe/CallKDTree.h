/*
 * CallKDTree.h
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once
#include "mmstd/generic/CallGeneric.h"
#include <nanoflann.hpp>

namespace megamol {
namespace probe {

template<typename Derived>
struct kd_adaptor {
    typedef typename Derived::value_type::value_type coord_t;

    const Derived& obj; //!< A const ref to the data set origin

    /// The constructor that sets the data set source
    kd_adaptor(const Derived& obj_) : obj(obj_) {}

    /// CRTP helper method
    inline const Derived& derived() const {
        return obj;
    }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return derived().size();
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline coord_t kdtree_get_pt(const size_t idx, const size_t dim) const {
        return derived()[idx][dim];
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo
    //   it again. Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template<class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const {
        return false;
    }

};

typedef kd_adaptor<std::vector<std::array<float, 3>>> data2KD;
typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, data2KD>, data2KD, 3 /* dim */>
my_kd_tree_t;

class CallKDTree
             : public core::GenericVersionedCall<std::shared_ptr<my_kd_tree_t>, core::Spatial3DMetaData> {
     public:
    CallKDTree()
                 : core::GenericVersionedCall<std::shared_ptr<my_kd_tree_t>, core::Spatial3DMetaData>() {}
    ~CallKDTree(){};

    static const char* ClassName(void) {
        return "CallKDTree";
    }
    static const char* Description(void) {
        return "Call that gives access to kd-tree data.";
    }
};

typedef megamol::core::factories::CallAutoDescription<CallKDTree> CallKDTreeDescription;

} // namespace probe
} // namespace megamol
