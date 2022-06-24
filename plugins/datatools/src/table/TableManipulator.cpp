/*
 * TableManipulator.cpp
 *
 * Copyright (C) 2019 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "TableManipulator.h"

#include "mmcore/param/StringParam.h"

#include "mmcore/utility/log/Log.h"
#include "vislib/StringTokeniser.h"
#include <limits>

using namespace megamol::datatools;
using namespace megamol::datatools::table;
using namespace megamol;

std::string TableManipulator::ModuleName = std::string("TableManipulator");

std::string TableManipulator::defaultScript = "";

TableManipulator::TableManipulator(void)
        : core::Module()
        , dataOutSlot("dataOut", "Output")
        , dataInSlot("dataIn", "Input")
        , scriptSlot("script", "script to execute on incoming table data")
        , frameID(-1)
        , in_datahash(std::numeric_limits<unsigned long>::max())
        , out_datahash(0)
        , theLua(this) {

    this->dataInSlot.SetCompatibleCall<TableDataCallDescription>();
    this->MakeSlotAvailable(&this->dataInSlot);

    this->dataOutSlot.SetCallback(
        TableDataCall::ClassName(), TableDataCall::FunctionName(0), &TableManipulator::processData);
    this->dataOutSlot.SetCallback(
        TableDataCall::ClassName(), TableDataCall::FunctionName(1), &TableManipulator::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);

    this->scriptSlot << new core::param::StringParam(
        "-- example script copying everything from right to left\n"
        "rows, cols = mmGetInputSize()\n"
        "print('got data of size ' .. tostring(rows) .. ' x ' .. tostring(cols))\n"
        "\n"
        "-- always set number of output columns first!\n"
        "mmSetOutputColumns(cols)\n"
        "\n"
        "mins = {}\n"
        "maxes = {}\n"
        "\n"
        "for c = 0, cols - 1 do\n"
        "    n = mmGetInputColumnName(c)\n"
        "    min, max = mmGetInputColumnRange(c)\n"
        "    print('col ' .. tostring(c) .. '(' .. n .. '): [' .. tostring(min) .. ';' .. tostring(max) .. ']')\n"
        "    -- copy original names\n"
        "    mmSetOutputColumnName(c, n)\n"
        "    -- copy original ranges\n"
        "    mmSetOutputColumnRange(c, min, max)\n"
        "    mins[c] = math.huge\n"
        "    maxes[c] = -math.huge\n"
        "end\n"
        "\n"
        "-- this allocates the complete table at once for best performance\n"
        "-- you need to do this row-wise if you want to filter out data \n"
        "mmAddOutputRows(rows)\n"
        "\n"
        "for r = 0, rows -1 do\n"
        "    -- alternative to above: mmAddOutputRows(1), is slower because of re-allocations\n"
        "    for c = 0, cols - 1 do\n"
        "        v = mmGetCellValue(r, c)\n"
        "        if v < mins[c] then\n"
        "            mins[c] = v\n"
        "        end\n"
        "        if v > maxes[c] then\n"
        "            maxes[c] = v\n"
        "        end\n"
        "        mmSetCellValue(r, c, v)\n"
        "    end\n"
        "end\n"
        "\n"
        "print('setting new ranges:')\n"
        "for c = 0, cols - 1 do\n"
        "    print('col ' .. tostring(c) .. ': [' .. tostring(mins[c]) .. ';' .. tostring(maxes[c]) .. ']')\n"
        "    mmSetOutputColumnRange(c, mins[c], maxes[c])\n"
        "end\n");
    this->MakeSlotAvailable(&this->scriptSlot);
}

TableManipulator::~TableManipulator(void) {
    this->Release();
}

bool TableManipulator::create(void) {
    theLua.RegisterCallback<TableManipulator, &TableManipulator::getInputSize>(
        "mmGetInputSize", "()\n\treturns the number of rows, columns in the input data");
    theLua.RegisterCallback<TableManipulator, &TableManipulator::setOutputColumns>(
        "mmSetOutputColumns", "(int number)\n\tsets the number of columns of the output data");
    theLua.RegisterCallback<TableManipulator, &TableManipulator::getInputColumnName>(
        "mmGetInputColumnName", "(int idx)\n\treturns the name of column idx in the input data");
    theLua.RegisterCallback<TableManipulator, &TableManipulator::setOutputColumnName>(
        "mmSetOutputColumnName", "(int idx, string name)\n\tsets the name of column idx in the output data");
    theLua.RegisterCallback<TableManipulator, &TableManipulator::getInputColumnRange>(
        "mmGetInputColumnRange", "(int idx)\n\treturns min, max of column idx in the input data");
    theLua.RegisterCallback<TableManipulator, &TableManipulator::setOutputColumnRange>(
        "mmSetOutputColumnRange", "(int idx, float min, float max)\n\tsets the range of column idx in the output data");
    theLua.RegisterCallback<TableManipulator, &TableManipulator::addOutputRows>(
        "mmAddOutputRows", "(int num)\n\tadds and allocates num rows to the output data");
    theLua.RegisterCallback<TableManipulator, &TableManipulator::getCellValue>(
        "mmGetCellValue", "(int row, int col)\n\treturns value in cell (row, col) in the input data");
    theLua.RegisterCallback<TableManipulator, &TableManipulator::setCellValue>(
        "mmSetCellValue", "(int row, int col, float val)\n\tset cell (row, col) in the output data to val");

    return true;
}

void TableManipulator::release(void) {}

bool TableManipulator::processData(core::Call& c) {
    try {
        TableDataCall* outCall = dynamic_cast<TableDataCall*>(&c);
        if (outCall == NULL)
            return false;

        TableDataCall* inCall = this->dataInSlot.CallAs<TableDataCall>();
        if (inCall == NULL)
            return false;

        inCall->SetFrameID(outCall->GetFrameID());
        if (!(*inCall)())
            return false;

        if (this->in_datahash != inCall->DataHash() || this->frameID != inCall->GetFrameID() ||
            this->scriptSlot.IsDirty()) {
            this->in_datahash = inCall->DataHash();
            this->frameID = inCall->GetFrameID();
            this->scriptSlot.ResetDirty();
            this->out_datahash++;

            column_count = inCall->GetColumnsCount();
            column_infos = inCall->GetColumnsInfos();
            row_count = inCall->GetRowsCount();
            in_data = inCall->GetData();

            const std::string scriptString = std::string(this->scriptSlot.Param<core::param::StringParam>()->Value());

            this->info.clear();
            this->info.reserve(column_count);
            this->data.clear();
            this->data.reserve(column_count * row_count);

            std::string res;
            const bool ok = theLua.RunString(scriptString, res);

            if (!ok) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "TableManipulator: Lua execution is NOT OK and returned '%s'", res.c_str());
            }
        }

        outCall->SetFrameCount(inCall->GetFrameCount());
        outCall->SetFrameID(this->frameID);
        outCall->SetDataHash(this->out_datahash);

        if (!this->info.empty()) {
            outCall->Set(
                this->info.size(), this->data.size() / this->info.size(), this->info.data(), this->data.data());
        } else {
            outCall->Set(0, 0, NULL, NULL);
        }
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            _T("%hs: Failed to execute processData\n"), ModuleName.c_str());
        return false;
    }

    return true;
}

bool TableManipulator::getExtent(core::Call& c) {
    try {
        TableDataCall* outCall = dynamic_cast<TableDataCall*>(&c);
        if (outCall == NULL)
            return false;

        TableDataCall* inCall = this->dataInSlot.CallAs<TableDataCall>();
        if (inCall == NULL)
            return false;

        inCall->SetFrameID(outCall->GetFrameID());
        if (!(*inCall)(1))
            return false;

        outCall->SetFrameCount(inCall->GetFrameCount());
        outCall->SetDataHash(this->out_datahash); // TODO: this is actually crap if somebody properly checks it
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            _T("Failed to execute %hs::getExtent\n"), ModuleName.c_str());
        return false;
    }

    return true;
}

int TableManipulator::getInputSize(lua_State* L) {
    lua_pushinteger(L, row_count);
    lua_pushinteger(L, column_count);
    return 2;
}

int TableManipulator::setOutputColumns(lua_State* L) {
    const auto cnt = luaL_checkinteger(L, 1);
    this->info.resize(cnt);
    return 0;
}

int TableManipulator::getInputColumnName(lua_State* L) {
    const auto idx = luaL_checkinteger(L, 1);
    if (idx > -1 && idx < column_count) {
        lua_pushstring(L, column_infos[idx].Name().c_str());
        return 1;
    } else {
        lua_pushstring(L, "column index out of range");
        lua_error(L);
        return 0;
    }
}

int TableManipulator::setOutputColumnName(lua_State* L) {
    const auto idx = luaL_checkinteger(L, 1);
    const auto name = luaL_checkstring(L, 2);
    if (idx > -1 && idx < this->info.size()) {
        this->info[idx].SetName(name);
        this->info[idx].SetType(TableDataCall::ColumnType::QUANTITATIVE);
        return 0;
    } else {
        lua_pushstring(L, "column index out of range");
        lua_error(L);
        return 0;
    }
}

int TableManipulator::getInputColumnRange(lua_State* L) {
    const auto idx = luaL_checkinteger(L, 1);
    if (idx > -1 && idx < column_count) {
        lua_pushnumber(L, column_infos[idx].MinimumValue());
        lua_pushnumber(L, column_infos[idx].MaximumValue());
        return 2;
    } else {
        lua_pushstring(L, "column index out of range");
        lua_error(L);
        return 0;
    }
}

int TableManipulator::setOutputColumnRange(lua_State* L) {
    const auto idx = luaL_checkinteger(L, 1);
    const auto min = luaL_checknumber(L, 2);
    const auto max = luaL_checknumber(L, 3);
    if (idx > -1 && idx < this->info.size()) {
        info[idx].SetMinimumValue(min);
        info[idx].SetMaximumValue(max);
        return 0;
    } else {
        lua_pushstring(L, "column index out of range");
        lua_error(L);
        return 0;
    }
}

int TableManipulator::addOutputRows(lua_State* L) {
    const auto n = luaL_checkinteger(L, 1);
    if (n > 0) {
        if (!info.empty()) {
            this->data.resize(this->data.size() + n * info.size());
            return 0;
        } else {
            lua_pushstring(L, "you need to set the number of output columns first");
            lua_error(L);
            return 0;
        }
    } else {
        lua_pushstring(L, "you can only add rows");
        lua_error(L);
        return 0;
    }
}

int TableManipulator::getCellValue(lua_State* L) {
    const auto row = luaL_checkinteger(L, 1);
    const auto col = luaL_checkinteger(L, 2);
    if (row > -1 && row < row_count && col > -1 && col < column_count) {
        const uint32_t idx = row * column_count + col;
        lua_pushnumber(L, in_data[idx]);
        return 1;
    } else {
        lua_pushstring(L, "illegal cell index");
        lua_error(L);
        return 0;
    }
}

int TableManipulator::setCellValue(lua_State* L) {
    const auto row = luaL_checkinteger(L, 1);
    const auto col = luaL_checkinteger(L, 2);
    const auto val = luaL_checknumber(L, 3);

    if (!info.empty()) {
        const uint32_t idx = row * info.size() + col;
        if (idx >= 0 && idx < data.size()) {
            data[idx] = val;
            return 0;
        } else {
            lua_pushstring(L, "illegal cell index");
            lua_error(L);
            return 0;
        }
    } else {
        lua_pushstring(L, "you need to set the number of output columns first");
        lua_error(L);
        return 0;
    }
}
