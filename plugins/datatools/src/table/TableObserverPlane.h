#ifndef MEGAMOL_DATATOOLS_FLOATTABLEOBSERVERPLANE_H_INCLUDED
#define MEGAMOL_DATATOOLS_FLOATTABLEOBSERVERPLANE_H_INCLUDED
#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "datatools/table/TableDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/renderer/CallClipPlane.h"
#include <map>

namespace megamol::datatools::table {

/**
 * This module converts from a generic table to the MultiParticleDataCall.
 */
class TableObserverPlane : public megamol::core::Module {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() {
        return "TableObserverPlane";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() {
        return "A plane that observes relevant items in local (xy) coordinates over dicrete time steps and stacks them "
               "along the z axis.";
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
    TableObserverPlane();

    /**
     * Finalises an instance.
     */
    ~TableObserverPlane() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    bool getObservedData(core::Call& call);

    bool getHash(core::Call& call);

    /**
     * Implementation of 'Release'.
     */
    void release() override;

private:
    bool assertData(table::TableDataCall* ft, megamol::core::view::CallClipPlane* cp, table::TableDataCall& out);

    bool anythingDirty();

    void resetAllDirty();

    std::string cleanUpColumnHeader(const std::string& header) const;
    std::string cleanUpColumnHeader(const vislib::TString& header) const;

    int getColumnIndex(const vislib::TString& colName);
    bool pushColumnIndex(std::vector<size_t>& cols, const vislib::TString& colName);

    /** Minimum coordinates of the bounding box. */
    float bboxMin[3];

    /** Maximum coordinates of the bounding box. */
    float bboxMax[3];

    float iMin, iMax;

    /** The slot for retrieving the stacked observed planes. */
    core::CalleeSlot slotCallObservedTable;

    /** The data callee slot. */
    core::CallerSlot slotCallInputTable;

    /** The clip plane slot defining the observation plane. */
    core::CallerSlot slotCallClipPlane;

    /** The name of the float column holding the x-coordinate. */
    core::param::ParamSlot slotColumnX;

    /** The name of the float column holding the y-coordinate. */
    core::param::ParamSlot slotColumnY;

    /** The name of the float column holding the z-coordinate. */
    core::param::ParamSlot slotColumnZ;

    /** The name of the float column holding the particle radius. */
    core::param::ParamSlot slotColumnRadius;

    /**
     * The constant radius of spheres if the data set does not provide
     * one.
     */
    core::param::ParamSlot slotGlobalRadius;

    core::param::ParamSlot slotStartTime;
    core::param::ParamSlot slotEndTime;

    // makes no sense without float time in FTC
    //core::param::ParamSlot slotTimeIncrement;
    core::param::ParamSlot slotSliceOffset;

    /** The color mode: explicit rgb, intensity or constant */
    core::param::ParamSlot slotRadiusMode;

    /** how particles are chosen for the result planes */
    core::param::ParamSlot slotObservationStrategy;

    std::vector<float> everything;
    SIZE_T inputHash;
    SIZE_T myHash;
    std::map<std::string, size_t> columnIndex;
    size_t stride;
    int frameID;
};

} // namespace megamol::datatools::table

#endif /* MEGAMOL_DATATOOLS_FLOATTABLEOBSERVERPLANE_H_INCLUDED */
