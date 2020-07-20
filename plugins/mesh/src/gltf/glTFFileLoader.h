/*
 * glTFFileLoader.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef GLTF_FILE_LOADER_H_INCLUDED
#define GLTF_FILE_LOADER_H_INCLUDED

#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"

#include "mesh/MeshCalls.h"
#include "mesh/mesh.h"
#include "mesh/MeshDataAccessCollection.h"

namespace megamol {
namespace mesh {

class MESH_API GlTFFileLoader : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "GlTFFileLoader"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Data source for simply loading a glTF file from disk"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    GlTFFileLoader();
    ~GlTFFileLoader();

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

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getGltfDataCallback(core::Call& caller);

    bool getGltfMetaDataCallback(core::Call& caller);

    bool getMeshDataCallback(core::Call& caller);

    bool getMeshMetaDataCallback(core::Call& caller);

    bool checkAndLoadGltfModel();

private:
    std::shared_ptr<tinygltf::Model> m_gltf_model;

    std::shared_ptr<MeshDataAccessCollection> m_mesh_collection;

    uint32_t m_version;

    /** The gltf file name */
    core::param::ParamSlot m_glTFFilename_slot;

    /** The slot for providing the complete gltf model */
    megamol::core::CalleeSlot m_gltf_slot;
    //size_t m_gltf_cached_hash;

    /** The slot for providing access to internal mesh data */
    megamol::core::CalleeSlot m_mesh_slot;
    //size_t m_mesh_cached_hash;
};

} // namespace mesh
} // namespace megamol


#endif // !GLTF_FILE_LOADER_H_INCLUDED
