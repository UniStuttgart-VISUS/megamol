#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "datatools/table/TableDataCall.h"
#include "geometry_calls/VolumetricDataCall.h"

namespace megamol::datatools {

/**
 * This module converts from a VolumetricDataCall to a table
 */
class VolumeToTable : public megamol::core::Module {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() {
        return "VolumeToTable";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() {
        return "Converts a volume to generic tables.";
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
    VolumeToTable();

    /**
     * Finalises an instance.
     */
    ~VolumeToTable() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    bool getTableData(core::Call& call);

    bool getTableHash(core::Call& call);

    /**
     * Implementation of 'Release'.
     */
    void release() override;

private:
    bool assertVDC(geocalls::VolumetricDataCall* in, table::TableDataCall* tc);

    /** The slot for retrieving the data as multi particle data. */
    core::CalleeSlot slotTableOut;

    /** The data callee slot. */
    core::CallerSlot slotVolumeIn;

    std::vector<float> everything;

    SIZE_T inHash = SIZE_MAX;
    unsigned int inFrameID = std::numeric_limits<unsigned int>::max();
    std::vector<table::TableDataCall::ColumnInfo> column_infos;
    std::size_t num_voxels = 0;
};

} // namespace megamol::datatools
