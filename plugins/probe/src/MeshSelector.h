/*
 * MeshSelector.h
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mesh/MeshCalls.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "probe/MeshUtilities.h"

namespace megamol {
namespace probe {

class MeshSelector : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "MeshSelector";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Manipulates Mesh data using libigl";
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
    MeshSelector(void);

    /** Dtor. */
    virtual ~MeshSelector(void);

protected:
    virtual bool create();
    virtual void release();


    core::CallerSlot _getDataSlot;
    core::CalleeSlot _deployMeshSlot;

    core::param::ParamSlot _meshNumberSlot;
    core::param::ParamSlot _splitMeshSlot;


private:
    bool InterfaceIsDirty();
    void connectedMesh(const uint32_t idx);
    bool splitMesh(const mesh::MeshDataAccessCollection::Mesh& mesh);

    bool getMetaData(core::Call& call);
    bool getData(core::Call& call);

    bool parameterChanged(core::param::ParamSlot& p);

    // CallMesh stuff
    std::vector<float> _mesh_vertices;
    std::vector<uint32_t> _mesh_faces;
    std::vector<mesh::MeshDataAccessCollection::VertexAttribute> _mesh_attribs;
    std::vector<mesh::MeshDataAccessCollection::IndexData> _mesh_indices;
    uint32_t _version = 0;

    bool _recalc = true;
    core::BoundingBoxes_2 _bbox;
    std::shared_ptr<MeshUtility> _mu;
    uint32_t _numFaces;

    mesh::MeshDataAccessCollection::VertexAttribute _pos_attribute;

    std::map<uint32_t, uint32_t> _face_to_mesh_map;
    uint32_t _current_mesh_id;
    std::map<uint32_t, std::vector<uint32_t>> _neighbor_map;
    std::shared_ptr<mesh::MeshDataAccessCollection> _split_mesh;
    //std::vector<std::vector<std::array<float, 3>>> _grouped_vertices;
    //std::vector<std::map<uint32_t, uint32_t>> _new_indice_map;
    std::vector<std::vector<std::array<uint32_t, 3>>> _grouped_indices;

    int _selected_mesh;
};

} // namespace probe
} // namespace megamol
