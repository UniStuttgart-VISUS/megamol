/*
 * ReconstructSurface.h
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "nanoflann.hpp"
#include <CGAL/Surface_mesh/Surface_mesh.h>
#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <glm/glm.hpp>

#include "mesh/MeshCalls.h"
#include "mmadios/CallADIOSData.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace probe {

inline std::array<glm::mat3, 3> get_rot_mx(float angle) {
    std::array<glm::mat3, 3> ret_val;
    ret_val[0] = glm::mat3(0, 0, 1, 0, glm::cos(angle), glm::sin(angle), 0, -glm::sin(angle), glm::cos(angle));
    ret_val[1] = glm::mat3(glm::cos(angle), 0, -glm::sin(angle), 0, 1, 0, glm::sin(angle), 0, glm::cos(angle));
    ret_val[2] = glm::mat3(glm::cos(angle), glm::sin(angle), 0, -glm::sin(angle), glm::cos(angle), 0, 0, 0, 1);
    return ret_val;
}

// default triangulation for Surface_mesher
typedef CGAL::Surface_mesh_default_triangulation_3 Tr;
typedef Tr::Geom_traits GT;
typedef GT::Point_3 Point;
typedef CGAL::Surface_mesh<Point> Surface_mesh;

class ReconstructSurface : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "ReconstructSurface";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Extracts a surface mesh from volume data.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor. */
    ReconstructSurface(void);

    /** Dtor. */
    virtual ~ReconstructSurface(void);

protected:
    virtual bool create();
    virtual void release();


    core::CallerSlot _getDataCall;
    core::CalleeSlot _deployMeshCall;
    core::CalleeSlot _deployNormalsCall;
    core::CalleeSlot _meshToDiscCall;
    core::CallerSlot _meshFromDiscCall;
    core::CallerSlot _getHullCall;

    core::param::ParamSlot _xSlot;
    core::param::ParamSlot _ySlot;
    core::param::ParamSlot _zSlot;
    core::param::ParamSlot _xyzSlot;
    core::param::ParamSlot _formatSlot;
    core::param::ParamSlot _showShellSlot;
    core::param::ParamSlot _numShellsSlot;
    core::param::ParamSlot _meshOutputSlot;
    core::param::ParamSlot _shellSplitsAxis;
    core::param::ParamSlot _shellSplitsAngle;

private:
    bool InterfaceIsDirty();

    void do_remeshing(Surface_mesh& mesh, float spacing_ = 0.0f);
    void generateNormals(Surface_mesh& mesh);
    void generateNormals_2(Surface_mesh& mesh, std::vector<std::array<float, 3>>& normals);
    void onionize();
    void cut();
    void remove_self_intersections(Surface_mesh& mesh);
    void do_smoothing(Surface_mesh& mesh);
    void generateBox();

    void compute();
    bool processRawData(adios::CallADIOSData* call, bool& something_changed);

    bool getMetaData(core::Call& call);
    bool getData(core::Call& call);

    bool getADIOSMetaData(core::Call& call);
    bool getADIOSData(core::Call& call);
    bool readMeshElementsFromFile();

    bool parameterChanged(core::param::ParamSlot& p);
    bool shellToShowChanged(core::param::ParamSlot& p);

    bool getNormalData(core::Call& call);

    bool getNormalMetaData(core::Call& call);

    void activateMesh(const Surface_mesh& shell, std::vector<std::array<float, 3>>& vertices,
        std::vector<std::array<uint32_t, 3>>& indices);

    // CallMesh stuff
    typedef std::pair<mesh::MeshDataAccessCollection::IndexData,
        std::vector<mesh::MeshDataAccessCollection::VertexAttribute>>
        Mesh;
    Mesh _mesh;
    uint32_t _version = 0;

    size_t _old_datahash;
    bool _recalc = false;

    std::vector<float> _raw_positions;

    bool _useBBoxAsHull = false;

    int _main_axis;
    std::array<int, 2> _off_axes;
    glm::vec3 _data_origin;
    std::vector<std::array<float, 4>> _ellipsoid_backup;

    Surface_mesh _sm;
    std::vector<Surface_mesh> _scaledHulls;
    std::vector<Surface_mesh> _shells;
    std::vector<std::vector<Surface_mesh>> _shellElements;
    std::vector<std::vector<std::vector<std::array<float, 3>>>> _shellElementsVertices;
    std::vector<std::vector<std::vector<std::array<float, 3>>>> _shellElementsNormals;
    std::vector<std::vector<std::vector<std::array<uint32_t, 3>>>> _shellElementsTriangles;
    std::vector<core::BoundingBoxes_2> _shellBBoxes;
    bool _shellToShowChanged = false;
    ;
    std::vector<std::vector<Mesh>> _elementMesh;

    std::shared_ptr<mesh::MeshDataAccessCollection> _mesh_for_call;

    // store surface
    std::vector<std::array<float, 3>> _vertices;
    std::vector<std::array<float, 3>> _normals;
    std::vector<std::array<uint32_t, 3>> _triangles;

    // store bounding box
    megamol::core::BoundingBoxes_2 _bbox;

    adios::adiosDataMap _dataMap;
};

} // namespace probe
} // namespace megamol
