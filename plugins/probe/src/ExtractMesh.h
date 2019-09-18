/*
 * ExtractMesh.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "concave_hull.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mesh/MeshCalls.h"

namespace megamol {
namespace probe {

class ExtractMesh : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "ExtractMesh"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Extracts a mesh from point data."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    ExtractMesh(void);

    /** Dtor. */
    virtual ~ExtractMesh(void);

protected:
    virtual bool create();
    virtual void release();


    core::CallerSlot _getDataCall;
    core::CalleeSlot _deployMeshCall;
    core::param::ParamSlot _algorithmSlot;
    core::param::ParamSlot _xSlot;
    core::param::ParamSlot _ySlot;
    core::param::ParamSlot _zSlot;
    core::param::ParamSlot _xyzSlot;
    core::param::ParamSlot _formatSlot;
    core::param::ParamSlot _alphaSlot;


private:
    bool InterfaceIsDirty();
    // virtual void readParams();
    void calculateAlphaShape();

    bool createPointCloud(std::vector<std::string>& vars);
    void convertToMesh();
    bool getData(core::Call& call);

    bool getMetaData(core::Call& call);

    bool toggleFormat(core::param::ParamSlot& p);

  // PCL stuff
    pcl::PointCloud<pcl::PointXYZ> _cloud;
    std::vector<pcl::Vertices> _polygons;
    std::vector<pcl::PointIndices> _indices;

    // CallMesh stuff
    std::vector<mesh::MeshDataAccessCollection::VertexAttribute> _mesh_attribs;
    mesh::MeshDataAccessCollection::IndexData _mesh_indices;
    core::BoundingBoxes _bbox;

    size_t _old_datahash;

};

} // namespace probe
} // namespace megamol
