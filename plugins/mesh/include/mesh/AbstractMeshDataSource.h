/*
 * AbstractMeshDataSource.h
 *
 * MegaMol
 * Copyright(c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mesh/MeshCalls.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"


namespace megamol::mesh {

class AbstractMeshDataSource : public core::Module {
public:
    AbstractMeshDataSource();
    ~AbstractMeshDataSource() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    virtual bool getMeshDataCallback(core::Call& caller) = 0;

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    virtual bool getMeshMetaDataCallback(core::Call& caller);

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * Clears all mesh access entries made by this module from the used material collection.
     */
    void clearMeshAccessCollection();

    /**
     * Local mesh collection that is used with a list of identifier strings of meshs accesses that this module added to
     * the mesh collection.
     * Append this each frame to the mesh collection passing trough the calls.
     */
    std::pair<std::shared_ptr<MeshDataAccessCollection>, std::vector<std::string>> m_mesh_access_collection;

    /** The slot for querying additional mesh data, i.e. a rhs chaining connection */
    megamol::core::CallerSlot m_mesh_rhs_slot;

    /** The slot for requesting data */
    megamol::core::CalleeSlot m_mesh_lhs_slot;
};

} // namespace megamol::mesh
