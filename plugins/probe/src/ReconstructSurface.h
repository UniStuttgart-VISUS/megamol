/*
 * ReconstructSurface.h
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mesh/MeshCalls.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "nanoflann.hpp"
#include <CGAL/Surface_mesh/Surface_mesh.h>
#include <CGAL/Surface_mesh_default_triangulation_3.h>

namespace megamol {
namespace probe {
    // default triangulation for Surface_mesher
    typedef CGAL::Surface_mesh_default_triangulation_3 Tr;
    typedef Tr::Geom_traits GT;
    typedef GT::Point_3 Point;
    typedef CGAL::Surface_mesh<Point> Surface_mesh;

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
            if (dim == 0)
                return derived()[idx][0];
            else if (dim == 1)
                return derived()[idx][1];
            else
                return derived()[idx][2];
        }

        // Optional bounding-box computation: return false to default to a standard bbox computation loop.
        //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo
        //   it again. Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
        template<class BBOX>
        bool kdtree_get_bbox(BBOX& /*bb*/) const {
            return false;
        }

    }; // end of PointCloudAdaptor


class ReconstructSurface : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "ReconstructSurface"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Extracts a surface mesh from volume data."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

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

    core::param::ParamSlot _numSlices;
    core::param::ParamSlot _isoValue;
    core::param::ParamSlot _meshResolution;
    core::param::ParamSlot _faceTypeSlot;

    core::param::ParamSlot _xSlot;
    core::param::ParamSlot _ySlot;
    core::param::ParamSlot _zSlot;
    core::param::ParamSlot _xyzSlot;
    core::param::ParamSlot _formatSlot;


private:
    bool InterfaceIsDirty();

    void sliceData();
    void generateEllipsoid();
    void generateEllipsoid_2();
    void tighten();
    void generateNormals();
    void onionize();

    bool getMetaData(core::Call& call);
    bool getData(core::Call& call);

    bool parameterChanged(core::param::ParamSlot& p);

    bool getNormalData(core::Call& call);

    bool getNormalMetaData(core::Call& call);

    // CallMesh stuff
    std::vector<mesh::MeshDataAccessCollection::VertexAttribute> _mesh_attribs;
    mesh::MeshDataAccessCollection::IndexData _mesh_indices;
    mesh::MeshDataAccessCollection::PrimitiveType _mesh_type;
    uint32_t _version = 0;

    size_t _old_datahash;
    bool _recalc = false;

    std::vector<float> _raw_positions;

    std::vector<std::vector<uint32_t>> _slice_data;
    std::vector<std::vector<uint32_t>> _slice_ellipsoid;
    std::vector<std::vector<std::array<float,3>>> _sliced_positions;
    std::vector<std::vector<std::array<float, 3>>> _sliced_positions_whalo;
    std::vector<std::vector<std::array<float, 3>>> _sliced_vertices;
    std::vector<glm::vec3> _slice_data_center_of_mass;
    std::vector<glm::vec3> _slice_ellipsoid_center_of_mass;
    int _main_axis;
    glm::vec3 _data_origin;

    Surface_mesh _sm;


    // store surface
    std::vector<std::array<float, 4>> _vertices;
    std::vector<std::array<float, 3>> _normals;
    std::vector<std::array<uint32_t, 4>> _faces;
    std::vector<std::array<uint32_t, 3>> _triangles;

    // store bounding box
    megamol::core::BoundingBoxes_2 _bbox;
    typedef kd_adaptor<std::vector<std::array<float, 3>>> data2KD;
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, data2KD>, data2KD, 3 /* dim */
        >my_kd_tree_t;
    std::vector<std::shared_ptr<my_kd_tree_t>> _kd_indices;
    std::vector<std::shared_ptr<const data2KD>> _data2kd;

};

} // namespace probe
} // namespace megamol
