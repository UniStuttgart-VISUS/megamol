#ifndef MEGAMOL_DATATOOLS_FLOATTABLETOPARTICLES_H_INCLUDED
#define MEGAMOL_DATATOOLS_FLOATTABLETOPARTICLES_H_INCLUDED
#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "datatools/table/TableDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include <array>
#include <map>

namespace megamol::datatools {

/**
 * This module converts from a generic table to the MultiParticleDataCall.
 */
class TableToParticles : public megamol::core::Module {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() {
        return "TableToParticles";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() {
        return "Converts generic tables to Particles.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable() {
        return true;
    }

    /**
     * Initialises a new instance.
     */
    TableToParticles();

    /**
     * Finalises an instance.
     */
    ~TableToParticles() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    bool getMultiParticleData(core::Call& call);

    bool getMultiparticleExtent(core::Call& call);

    /**
     * Implementation of 'Release'.
     */
    void release() override;

private:
    bool assertData(table::TableDataCall* ft, unsigned int frameID = 0);

    bool anythingDirty();

    void resetAllDirty();

    std::string cleanUpColumnHeader(const std::string& header) const;
    std::string cleanUpColumnHeader(const vislib::TString& header) const;

    bool pushColumnIndex(std::vector<uint32_t>& cols, const vislib::TString& colName);

    /** Minimum coordinates of the bounding box. */
    float bboxMin[3];

    /** Maximum coordinates of the bounding box. */
    float bboxMax[3];

    float iMin, iMax;

    /** The slot for retrieving the data as multi particle data. */
    core::CalleeSlot slotCallMultiPart;

    /** The data callee slot. */
    core::CallerSlot slotCallTable;

    /** The name of the float column holding the red colour channel. */
    core::param::ParamSlot slotColumnB;

    /** The name of the float column holding the green colour channel. */
    core::param::ParamSlot slotColumnG;

    /** The name of the float column holding the blue colour channel. */
    core::param::ParamSlot slotColumnR;

    /** The name of the float column holding the intensity channel. */
    core::param::ParamSlot slotColumnI;

    /**
     * The constant color of spheres if the data set does not provide
     * one.
     */
    core::param::ParamSlot slotGlobalColor;

    /** The color mode: explicit rgb, intensity or constant */
    core::param::ParamSlot slotColorMode;

    /** The name of the float column holding the particle radius. */
    core::param::ParamSlot slotColumnRadius;

    /**
     * The constant radius of spheres if the data set does not provide
     * one.
     */
    core::param::ParamSlot slotGlobalRadius;

    /** The color mode: explicit rgb, intensity or constant */
    core::param::ParamSlot slotRadiusMode;

    /** The name of the float column holding the x-coordinate. */
    core::param::ParamSlot slotColumnX;

    /** The name of the float column holding the y-coordinate. */
    core::param::ParamSlot slotColumnY;

    /** The name of the float column holding the z-coordinate. */
    core::param::ParamSlot slotColumnZ;

    /** The name of the int column holding the particle id. */
    core::param::ParamSlot slotColumnID;

    /** The name of the float column holding the vx-coordinate. */
    core::param::ParamSlot slotColumnVX;

    /** The name of the float column holding the vy-coordinate. */
    core::param::ParamSlot slotColumnVY;

    /** The name of the float column holding the vz-coordinate. */
    core::param::ParamSlot slotColumnVZ;

    /** given a tensor interpretable as an orthonormal coordinate system {X,Y,Z}^T, its 9 values {xx, xy, xz, ...} */
    std::array<core::param::ParamSlot, 9> slotTensorColumn;

    /** if the tensor contains normalized vectors, you can also supply a magnitude */
    std::array<core::param::ParamSlot, 3> slotTensorMagnitudeColumn;

    /** The lastly calculated time step */
    unsigned int lastTimeStep;

    std::vector<float> everything;

    bool haveIDs = false;
    bool haveVelocities = false;
    bool haveTensor = false;
    bool haveTensorMagnitudes = false;

    SIZE_T inputHash;
    SIZE_T myHash;
    SIZE_T myTime = -1;
    std::map<std::string, uint32_t> columnIndex;
    uint32_t stride;
};

} // namespace megamol::datatools

#endif /* MEGAMOL_DATATOOLS_FLOATTABLETOPARTICLES_H_INCLUDED */
