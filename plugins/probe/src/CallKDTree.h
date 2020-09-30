/*
 * CallKDTree.h
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once
#include "mmcore/CallGeneric.h"
#include "ExtractMesh.h"

namespace megamol {
namespace probe {
class CallKDTree
    : public core::GenericVersionedCall<std::shared_ptr<pcl::KdTreeFLANN<pcl::PointXYZ>>, core::Spatial3DMetaData> {
public:
    CallKDTree()
        : core::GenericVersionedCall<std::shared_ptr<pcl::KdTreeFLANN<pcl::PointXYZ>>, core::Spatial3DMetaData>() {}
    ~CallKDTree(){};

    static const char* ClassName(void) { return "CallKDTree"; }
    static const char* Description(void) { return "Call that gives access to kd-tree data."; }

};

typedef megamol::core::factories::CallAutoDescription<CallKDTree> CallKDTreeDescription;

} // namespace probe
} // namespace megamol
