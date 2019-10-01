/*
 * ExtractMesh.h
 * Copyright (C) 2009-2017 by MegaMol Team
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
#include <cstdlib>

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
    core::CalleeSlot _deploySpheresCall;
    core::CalleeSlot _deployLineCall;
    core::param::ParamSlot _algorithmSlot;
    core::param::ParamSlot _xSlot;
    core::param::ParamSlot _ySlot;
    core::param::ParamSlot _zSlot;
    core::param::ParamSlot _xyzSlot;
    core::param::ParamSlot _formatSlot;
    core::param::ParamSlot _alphaSlot;


private:
    bool InterfaceIsDirty();
    bool flipNormalsWithCenterLine(pcl::PointCloud<pcl::PointNormal>& point_cloud);
    bool extractCenterLine(pcl::PointCloud<pcl::PointNormal>& point_cloud);
    // virtual void readParams();
    void calculateAlphaShape();

    bool createPointCloud(std::vector<std::string>& vars);
    void convertToMesh();
    bool getData(core::Call& call);

    bool getMetaData(core::Call& call);
    bool getParticleData(core::Call& call);
    bool getParticleMetaData(core::Call& call);
    bool getCenterlineData(core::Call& call);
     
    

    bool toggleFormat(core::param::ParamSlot& p);

    bool alphaChanged(core::param::ParamSlot& p);

    bool filterResult();
    bool filterByIndex();

    bool usePoisson = true;
    std::vector<float> vertex_data;

    // PCL stuff
    pcl::PointCloud<pcl::PointXYZ> _cloud;
    std::vector<pcl::Vertices> _polygons;
    std::vector<pcl::PointIndices> _indices;
    pcl::PointCloud<pcl::PointXYZ> _resultCloud;
    pcl::PointCloud<pcl::PointNormal> _resultSurface;
    std::shared_ptr<pcl::PointCloud<pcl::PointNormal>> _resultNormalCloud;

    // CallMesh stuff
    std::vector<mesh::MeshDataAccessCollection::VertexAttribute> _mesh_attribs;
    mesh::MeshDataAccessCollection::IndexData _mesh_indices;
    core::BoundingBoxes _bbox;

    // CallCenterline stuff
    std::vector<mesh::MeshDataAccessCollection::VertexAttribute> _line_attribs;
    mesh::MeshDataAccessCollection::IndexData _line_indices;
    std::vector<std::array<float, 4>> _centerline;
    std::vector<std::vector<uint32_t>> _cl_indices_per_slice;
    std::vector<uint32_t> _cl_indices;

    size_t _old_datahash = 0;
    size_t _recalc_hash = 0;

};

} // namespace probe
} // namespace megamol
