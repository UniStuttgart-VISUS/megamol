#pragma once
#ifndef OMNI_USD_READER_H_INCLUDED
#define OMNI_USD_READER_H_INCLUDED

#include "mesh/AbstractMeshDataSource.h"
#include "mesh/MeshCalls.h"
#include "mesh/MeshDataAccessCollection.h"

#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"

#define TBB_USE_DEBUG 0
#include "pxr/usd/usd/stage.h"



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

    uint32_t m_version;

    /** The usd file name */
    core::param::ParamSlot m_filename_slot;

    /**
     * Representation of usd stage as loaded by OmniUsdLib
     */
    pxr::UsdStageRefPtr m_usd_stage;


    /**
     * Internal storage for unpacked positions, i.e one position per vertex, three vertices per triangle
     */
    std::vector<std::vector<float>> m_positions;

    /**
     * Internal storage for unpacked normals, i.e one normal per vertex, three vertices per triangle
     */
    std::vector<std::vector<float>> m_normals;

    /**
     * Internal storage for unpacked texcoords, i.e one texcoord per vertex, three vertices per triangle
     */
    std::vector<std::vector<float>> m_texcoords;

    /**
     *
     */
    std::vector<std::vector<int>> m_indices;

    /**
     * Meta data for communicating data updates, as well as data size
     */
    core::Spatial3DMetaData m_meta_data;
};

} // namespace mesh
} // namespace megamol


#endif // !#ifndef OMNI_USD_READER_H_INCLUDED
