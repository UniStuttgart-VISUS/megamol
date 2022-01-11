#pragma once
#ifndef OMNI_USD_READER_H_INCLUDED
#define OMNI_USD_READER_H_INCLUDED

#include "mesh/AbstractMeshDataSource.h"
#include "mesh/MeshCalls.h"
#include "mesh/MeshDataAccessCollection.h"

#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol {
namespace mesh {

class OmniUsdReader : public AbstractMeshDataSource {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "OmniUsdReader";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Data source for simply loading a usd file from an omniverse server";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    OmniUsdReader();
    ~OmniUsdReader();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create(void);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getMeshDataCallback(core::Call& caller);

    bool getMeshMetaDataCallback(core::Call& caller);

    /**
     * Implementation of 'Release'.
     */
    void release();

private:
    /** The gltf file name */
    core::param::ParamSlot m_filename_slot;
};

} // namespace mesh
} // namespace megamol


#endif // !#ifndef OMNI_USD_READER_H_INCLUDED
