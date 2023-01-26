#pragma once

#include "FrameStatistics.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "datatools/table/TableDataCall.h"

namespace megamol {
namespace datatools {

/**
 * This module renders a table via ImGui.
 */
class TableInspector : public megamol::core::Module {

public:
    static void requested_lifetime_resources(frontend_resources::ResourceRequest& req) {
        Module::requested_lifetime_resources(req);
        req.require<frontend_resources::FrameStatistics>();
    }

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName(void) {
        return "TableInspector";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description(void) {
        return "Direct inspection of table values, data is passed through.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable(void) {
        return true;
    }

    /**
     * Initialises a new instance.
     */
    TableInspector(void);

    /**
     * Finalises an instance.
     */
    ~TableInspector(void) override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create(void) override;

    bool getTableData(core::Call& call);

    bool getTableHash(core::Call& call);

    /**
     * Implementation of 'Release'.
     */
    void release(void) override;

private:
    void drawTable(table::TableDataCall* c);

    /** The slot for retrieving the data as multi particle data. */
    core::CalleeSlot slotTableOut;

    /** The data callee slot. */
    core::CallerSlot slotTableIn;

    uint32_t lastDrawnFrame = std::numeric_limits<uint32_t>::max();
};

} /* end namespace datatools */
} /* end namespace megamol */
