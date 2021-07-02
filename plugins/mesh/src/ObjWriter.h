/*
 * ObjWriter.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include "mmcore/AbstractDataWriter.h"
#include "mesh/MeshCalls.h"
#include "mesh/MeshDataAccessCollection.h"

#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"

#include "obj_io/obj_io.h"

namespace megamol {
namespace mesh {

class ObjWriter : public core::AbstractDataWriter {
    public:

    struct Vec2 {
        float x;
        float y;
    };

    struct Vec3 {
        float x;
        float y;
        float z;
    };

    struct Vertex {
        Vec3 position;
        Vec2 tex_coord;
        Vec3 normal;
    };

    struct ObjMesh {
        std::vector<Vertex> vertices;
        std::vector<std::uint32_t> indices;
    };

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "ObjWriter"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Data source for simply loading a wavefront obj file from disk"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    ObjWriter();
    ~ObjWriter();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create(void);

    /**
     * Implementation of 'Release'.
     */
    void release();

    virtual bool run();

    /**
     * Function querying the writers capabilities
     *
     * @param call The call to receive the capabilities
     *
     * @return True on success
     */
    virtual bool getCapabilities(core::DataWriterCtrlCall& call);

private:

    void WriteMesh(const std::string&, const ObjMesh&);

    uint32_t _version;

    /**
     * Meta data for communicating data updates, as well as data size
     */
    core::Spatial3DMetaData _meta_data;

    /** The gltf file name */
    core::param::ParamSlot _filename_slot;

    core::CallerSlot _rhs_mesh_slot;

};

} // namespace mesh
} // namespace megamol

