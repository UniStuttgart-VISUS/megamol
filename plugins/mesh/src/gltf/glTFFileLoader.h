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

#include "mesh/AbstractMeshDataSource.h"
#include "mesh/MeshCalls.h"
#include "mesh/MeshDataAccessCollection.h"

namespace megamol {
namespace mesh {

class GlTFFileLoader : public AbstractMeshDataSource {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "GlTFFileLoader";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Data source for simply loading a glTF file from disk";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    GlTFFileLoader();
    ~GlTFFileLoader() override;

protected:
    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getGltfDataCallback(core::Call& caller);

    bool getGltfMetaDataCallback(core::Call& caller);

    bool getMeshDataCallback(core::Call& caller) override;

    bool getMeshMetaDataCallback(core::Call& caller) override;

    bool disconnectGltfCallback(core::Call& caller);

    bool checkAndLoadGltfModel();

private:
    std::shared_ptr<tinygltf::Model> m_gltf_model;

    uint32_t m_version;

    /** The gltf file name */
    core::param::ParamSlot m_glTFFilename_slot;

    /** The slot for providing the complete gltf model */
    megamol::core::CalleeSlot m_gltf_slot;
};

} // namespace mesh
} // namespace megamol


#endif // !GLTF_FILE_LOADER_H_INCLUDED
