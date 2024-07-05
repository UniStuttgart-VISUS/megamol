/*
 * TableColumnFilter.h
 *
 * Copyright (C) 2016-2016 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "FrameStatistics.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/param/ParamSlot.h"

#include "datatools/table/TableDataCall.h"

namespace megamol::datatools::table {

/*
 * Module to filter columns from a table.
 */
class TableColumnFilter : public core::Module {
public:
    static std::string ModuleName;

    /** Return module class name */
    static const char* ClassName() {
        return ModuleName.c_str();
    }

    /** Return module class description */
    static const char* Description() {
        return "Filters columns from a table";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }

    static void requested_lifetime_resources(frontend_resources::ResourceRequest& req) {
        Module::requested_lifetime_resources(req);
        req.require<frontend_resources::FrameStatistics>();
    }

    /** Ctor */
    TableColumnFilter();

    /** Dtor */
    ~TableColumnFilter() override;

protected:
    /** Lazy initialization of the module */
    bool create() override;

    /** Resource release */
    void release() override;

private:
    /** Data callback */
    bool processData(core::Call& c);

    bool getExtent(core::Call& c);

    /** Data output slot */
    core::CalleeSlot dataOutSlot;

    /** Data output slot */
    core::CallerSlot dataInSlot;

    /** Parameter slot for column selection */
    core::param::ParamSlot selectionStringSlot;

    /** show the checkbox-based GUI */
    core::param::ParamSlot showGUISlot;

    /**
     * write the selected columns from the selectionStringSlot to the selectedColumns vector
     * return whether this changes anything.
     */
    bool parseSelectionString(size_t column_count, const TableDataCall::ColumnInfo* column_info);

    /** serialize the selectedColumns into the selectionStringSlot */ 
    void writeSelectionString(size_t column_count, const TableDataCall::ColumnInfo* column_info);

    /** how many checkboxes per row in the GUI */
    int columnsPerRow = 5;

    /** we need to know the current frame to avoid generating checkboxes per call */
    const frontend_resources::FrameStatistics* frameStatistics = nullptr;

    /** ID of the current frame */
    int frameID;

    /** Hash of the incoming data */
    size_t inDatahash;

    /** Hash of the current data */
    size_t datahash;

    /** Vector storing information about columns */
    std::vector<TableDataCall::ColumnInfo> columnInfos;

    /** Vector storing the actual float data */
    std::vector<float> data;

    /** Vector for the checkboxes */
    std::vector<bool> selectedColumns;

    /** whether checkbox changes are applied immediately (for slow follow-up modules like dimensionality reduction) */
    bool autoApply = true;

}; /* end class TableColumnFilter */

} // namespace megamol::datatools::table
