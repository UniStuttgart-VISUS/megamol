#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "datatools/table/TableDataCall.h"

namespace megamol::datatools::table {

class TableSplit : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "TableSplit";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "split table in two parts based on column name selection string";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    TableSplit();

    ~TableSplit() override;

protected:
    bool create() override;

    void release() override;

private:
    bool getSelectedDataCB(core::Call& c);

    bool getRestDataCB(core::Call& c);

    bool getHashCB(core::Call& c);

    bool isDirty() {
        return _column_selector_slot.IsDirty();
    }

    void resetDirty() {
        _column_selector_slot.ResetDirty();
    }

    bool processData(TableDataCall const& inCall);

    core::CalleeSlot _selected_data_slot;

    core::CalleeSlot _rest_data_slot;

    core::CallerSlot _in_data_slot;

    core::param::ParamSlot _column_selector_slot;

    int _frame_id = -1;

    size_t _in_data_hash = std::numeric_limits<size_t>::max();

    size_t _out_data_hash = 0;

    std::vector<float> _selected_data;

    std::vector<float> _rest_data;

    std::vector<TableDataCall::ColumnInfo> _selected_info;

    std::vector<TableDataCall::ColumnInfo> _rest_info;

}; // end class TableSplit

} // end namespace megamol::datatools::table
