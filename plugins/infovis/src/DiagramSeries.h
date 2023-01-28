#pragma once

#include <array>
#include <vector>

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "datatools/table/TableDataCall.h"

#include "DiagramSeriesCall.h"

namespace megamol::infovis {

/**
 * Module to select a specific column from a table as diagram series.
 * Select multiple columns via cascading this module.
 * This module does not store a copy of the selected column.
 */
class DiagramSeries : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "DiagramSeries";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Module to select a specific column from a table as diagram series";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** ctor */
    DiagramSeries();

    /** dtor */
    ~DiagramSeries() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

private:
    /**
     * Callback that implements the cascade and eventually pushes info
     * about selected column to the caller.
     *
     * @param c DiagramSeriesCall
     */
    bool seriesSelectionCB(core::Call& c);

    /**
     * Updates selector parameter and creates info item about selected column.
     */
    bool assertData(const datatools::table::TableDataCall* const ft);

    /**
     * Checks if any parameter is dirty.
     */
    bool isAnythingDirty() const;

    /**
     * Resets dirty flags of parameters.
     */
    void resetDirtyFlags();

    /**
     * Finds column based on name.
     *
     * @param colIdx [out] Column idx of column 'columnName'.
     * @param columnName Column to find.
     * @param ft call containing the table column infos to search.
     *
     * @return True, if column exists.
     */
    bool getColumnIdx(
        uint32_t& colIdx, const vislib::TString& columnName, const datatools::table::TableDataCall* const ft) const;

    /** Selected columns output call */
    core::CalleeSlot seriesOutSlot;

    /** Selected columns input call */
    core::CallerSlot seriesInSlot;

    /** Table input */
    core::CallerSlot ftInSlot;

    /** Column selector enum */
    core::param::ParamSlot columnSelectorParam;

    /** Optional scaling parameter */
    core::param::ParamSlot scalingParam;

    /** Color parameter */
    core::param::ParamSlot colorParam;

    /** Series info storing idx of selected column, etc. */
    DiagramSeriesCall::DiagramSeriesTuple series;

    /** Color of series */
    std::array<float, 3> color;

    /** The hash of the Table */
    size_t inputHash;

    /** Internal hash incremented if selection, etc. changes */
    size_t myHash;
}; /* end class DiagramSeries */

} // namespace megamol::infovis
