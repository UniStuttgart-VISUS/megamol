/*
 * AbstractMeshDataSource.h
 *
 * MegaMol
 * Copyright(c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef ABSTRACT_MESH_DATA_SOURCE_H_INCLUDED
#define ABSTRACT_MESH_DATA_SOURCE_H_INCLUDED

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mesh/MeshCalls.h"


namespace megamol {
namespace mesh {

    class AbstractMeshDataSource : public core::Module {
    public:
        AbstractMeshDataSource();
        virtual ~AbstractMeshDataSource();
    
    protected:
        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);
    
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
        virtual bool getMeshMetaDataCallback(core::Call& caller) = 0;
    
        /**
         * Implementation of 'Release'.
         */
        virtual void release();
    
        void syncMeshAccessCollection(CallMesh* lhs_call, CallMesh* rhs_call);
    
        /**
         * Mesh collection that is used with a list of identifier strings of meshs accesses that this module added to the mesh collection.
         * Needed to delete/update submeshes if the collection is shared across a chain of data sources modules.
         */
        std::pair<std::shared_ptr<MeshDataAccessCollection>, std::vector<std::string>> m_mesh_access_collection;
    
        /** The slot for querying additional mesh data, i.e. a rhs chaining connection */
        megamol::core::CallerSlot m_mesh_rhs_slot;
    
        /** The slot for requesting data */
        megamol::core::CalleeSlot m_mesh_lhs_slot;
    };

} // namespace mesh
} // namespace megamol


#endif // !ABSTRACT_MESH_DATA_SOURCE_H_INCLUDED
