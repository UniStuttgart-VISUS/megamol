/*
 * ConstructKDTree.h
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "concave_hull.h"
#include "poisson.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mesh/MeshCalls.h"

namespace megamol {
namespace probe {

class ConstructKDTree : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "ConstructKDTree"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Constructs a KD tree."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    ConstructKDTree(void);

    /** Dtor. */
    virtual ~ConstructKDTree(void);

protected:
    virtual bool create();
    virtual void release();

    core::CallerSlot _getDataCall;
    core::CalleeSlot _deployFullDataTree;
    core::param::ParamSlot _xSlot;
    core::param::ParamSlot _ySlot;
    core::param::ParamSlot _zSlot;
    core::param::ParamSlot _xyzSlot;
    core::param::ParamSlot _formatSlot;


private:
    bool InterfaceIsDirty();
    bool createPointCloud(std::vector<std::string>& vars);
    bool getMetaData(core::Call& call);
    bool getData(core::Call& call);
    bool toggleFormat(core::param::ParamSlot& p);
    
    // PCL stuff
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr _inputCloud; 
    pcl::PointCloud<pcl::PointXYZ> _cloud;
    std::shared_ptr<pcl::PointCloud<pcl::PointNormal>> _resultNormalCloud;
    std::shared_ptr<pcl::KdTreeFLANN<pcl::PointXYZ>> _full_data_tree;

    size_t _old_datahash = 0;
    uint32_t _version = 0;
    core::BoundingBoxes_2 _bbox;
};

} // namespace probe
} // namespace megamol
