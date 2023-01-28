/*
 * TableManipulator.h
 *
 * Copyright (C) 2019 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/LuaInterpreter.h"
#include "mmcore/Module.h"

#include "mmcore/param/ParamSlot.h"

#include "datatools/table/TableDataCall.h"

namespace megamol::datatools::table {

/*
 * Module to manipulate table (copy) via a LUA script.
 */
class TableManipulator : public core::Module {
public:
    static std::string ModuleName;
    static std::string defaultScript;

    /** Return module class name */
    static const char* ClassName() {
        return ModuleName.c_str();
    }

    /** Return module class description */
    static const char* Description() {
        return "manipulate table (copy) via a LUA script";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    TableManipulator();

    /** Dtor */
    ~TableManipulator() override;

protected:
    /** Lazy initialization of the module */
    bool create() override;

    /** Resource release */
    void release() override;

protected:
    /** Lua Interface */

    /** returns incoming rows, columns */
    int getInputSize(lua_State* L);

    /** resizes @info */
    int setOutputColumns(lua_State* L);

    /** (idx) get name of incoming column idx */
    int getInputColumnName(lua_State* L);

    /** (idx, name) set name of column idx */
    int setOutputColumnName(lua_State* L);

    /** (idx) get min, max of column idx */
    int getInputColumnRange(lua_State* L);

    /** (idx, min, max) set min, max of column idx in output data */
    int setOutputColumnRange(lua_State* L);

    /** (num) adds and allocates num rows to output data */
    int addOutputRows(lua_State* L);

    /** (row, col) returns incoming value in that cell */
    int getCellValue(lua_State* L);

    /** (row, col, value) sets value in that cell */
    int setCellValue(lua_State* L);


private:
    /** Data callback */
    bool processData(core::Call& c);

    bool getExtent(core::Call& c);

    /** Data output slot */
    core::CalleeSlot dataOutSlot;

    /** Data output slot */
    core::CallerSlot dataInSlot;

    /** Parameter slot for column selection */
    core::param::ParamSlot scriptSlot;

    /** ID of the current frame */
    int frameID;

    /** Hash of the current data */
    size_t in_datahash, out_datahash;

    /** Vector storing the actual float data */
    std::vector<float> data;

    /** Vector storing information about columns */
    std::vector<TableDataCall::ColumnInfo> info;

    megamol::core::LuaInterpreter<TableManipulator> theLua;

    /** number of columns coming in */
    size_t column_count = 0;

    /** info of about columns coming in */
    const TableDataCall::ColumnInfo* column_infos = nullptr;

    /** number of rows coming in */
    size_t row_count = 0;

    /** the data coming in */
    const float* in_data = nullptr;

}; /* end class TableManipulator */

} // namespace megamol::datatools::table
