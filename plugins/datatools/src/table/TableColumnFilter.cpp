/*
 * TableColumnFilter.cpp
 *
 * Copyright (C) 2016-2017 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "TableColumnFilter.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/StringTokeniser.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <limits>

using namespace megamol::datatools;
using namespace megamol::datatools::table;
using namespace megamol;

std::string TableColumnFilter::ModuleName = std::string("TableColumnFilter");

TableColumnFilter::TableColumnFilter()
        : core::Module()
        , dataOutSlot("dataOut", "Ouput")
        , dataInSlot("dataIn", "Input")
        , selectionStringSlot("selection", "Select columns by name separated by \";\"")
        , showGUISlot("show GUI", "shows a GUI for toggling detected columns")
        , frameID(-1)
        , inDatahash(std::numeric_limits<unsigned long>::max())
        , datahash(std::numeric_limits<unsigned long>::max()) {

    this->dataInSlot.SetCompatibleCall<TableDataCallDescription>();
    this->MakeSlotAvailable(&this->dataInSlot);

    this->dataOutSlot.SetCallback(
        TableDataCall::ClassName(), TableDataCall::FunctionName(0), &TableColumnFilter::processData);
    this->dataOutSlot.SetCallback(
        TableDataCall::ClassName(), TableDataCall::FunctionName(1), &TableColumnFilter::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);

    this->selectionStringSlot << new core::param::StringParam("x;y;z");
    this->MakeSlotAvailable(&this->selectionStringSlot);
    this->showGUISlot << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->showGUISlot);
}

TableColumnFilter::~TableColumnFilter() {
    this->Release();
}

bool TableColumnFilter::create() {
    frameStatistics = &frontend_resources.get<frontend_resources::FrameStatistics>();
    return true;
}

void TableColumnFilter::release() {}

bool TableColumnFilter::processData(core::Call& c) {
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

        auto column_count = inCall->GetColumnsCount();
        auto column_infos = inCall->GetColumnsInfos();
        selectedColumns.resize(column_count, true);

        bool anything_checked = false;
        bool apply = false;
        static size_t last_drawn_frame;

        if (selectionStringSlot.IsDirty()) {
            apply = parseSelectionString(column_count, column_infos);
        }

        bool valid_imgui_scope =
            ((ImGui::GetCurrentContext() != nullptr) ? (ImGui::GetCurrentContext()->WithinFrameScope) : (false));
        if (valid_imgui_scope) {
            bool showGUI = this->showGUISlot.Param<core::param::BoolParam>()->Value();
            if (showGUI && last_drawn_frame != frameStatistics->rendered_frames_count) {
                ImGui::SetNextWindowCollapsed(false, ImGuiCond_Once);
                if (ImGui::Begin("TableColumnFilterWin", &showGUI)) {
                    ImGui::InputInt("columns per row", &columnsPerRow);
                    if (ImGui::Button("select all")) {
                        std::fill(selectedColumns.begin(), selectedColumns.end(), true);
                        anything_checked = true;
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("select none")) {
                        std::fill(selectedColumns.begin(), selectedColumns.end(), false);
                        anything_checked = true;
                    }
                    for (auto x = 0; x < column_count; ++x) {
                        bool val = selectedColumns[x];
                        if (ImGui::Checkbox(column_infos[x].Name().c_str(), &val)) {
                            selectedColumns[x] = val;
                            anything_checked = true;
                        }
                        if (x % columnsPerRow != columnsPerRow - 1 && x != static_cast<int>(column_count) - 1) {
                            ImGui::SameLine();
                        }
                    }
                    ImGui::Checkbox("Auto-apply", &autoApply);
                    ImGui::SameLine();
                    // did we already get a changed string param? then we do not check the GUI
                    if (!apply) {
                        apply = autoApply && anything_checked;
                        if (ImGui::Button("apply")) {
                            apply = true;
                        }
                    }
                }
                ImGui::End();
                last_drawn_frame = frameStatistics->rendered_frames_count;
                if (apply) {
                    writeSelectionString(column_count, column_infos);
                }
            }
            this->showGUISlot.Param<core::param::BoolParam>()->SetValue(showGUI);
        }
        selectionStringSlot.ResetDirty();

        if (this->inDatahash != inCall->DataHash() || this->frameID != inCall->GetFrameID() || apply) {
            this->inDatahash = inCall->DataHash();
            this->datahash++;
            this->selectionStringSlot.ResetDirty();
            this->frameID = inCall->GetFrameID();

            auto rows_count = inCall->GetRowsCount();
            auto in_data = inCall->GetData();

            auto num_selected_columns = std::count(selectedColumns.begin(), selectedColumns.end(), true);

            this->columnInfos.clear();
            this->columnInfos.reserve(num_selected_columns);
            for (size_t col = 0; col < column_count; ++col) {
                if (selectedColumns[col]) {
                    this->columnInfos.push_back(column_infos[col]);
                }
            }

            if (num_selected_columns == 0) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    _T("%hs: No matches for column names have been found\n"), ModuleName.c_str());
                this->data.clear();
                return false;
            }

            this->data.clear();
            this->data.reserve(rows_count * num_selected_columns);

            for (size_t row = 0; row < rows_count; row++) {
                for (size_t col = 0; col < column_count; ++col) {
                    if (selectedColumns[col]) {
                        this->data.push_back(in_data[col + row * column_count]);
                    }
                }
            }
        }

        outCall->SetFrameCount(inCall->GetFrameCount());
        outCall->SetFrameID(this->frameID);
        outCall->SetDataHash(this->datahash);

        if (this->columnInfos.size() != 0) {
            outCall->Set(this->columnInfos.size(), this->data.size() / this->columnInfos.size(),
                this->columnInfos.data(), this->data.data());
        } else {
            outCall->Set(0, 0, NULL, NULL);
        }
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            _T("Failed to execute %hs::processData\n"), ModuleName.c_str());
        return false;
    }

    return true;
}

bool TableColumnFilter::getExtent(core::Call& c) {
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
        outCall->SetDataHash(this->datahash);
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            _T("Failed to execute %hs::getExtent\n"), ModuleName.c_str());
        return false;
    }

    return true;
}
bool TableColumnFilter::parseSelectionString(size_t column_count, const TableDataCall::ColumnInfo* column_info) {
    auto selectionString = std::stringstream(this->selectionStringSlot.Param<core::param::StringParam>()->Value());

    auto ichar_equals = [](char a, char b) -> bool {
        return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
    };
    std::vector<bool> new_selection(column_count, false);
    for (std::string column; std::getline(selectionString, column, ';');) {
        for (size_t col = 0; col < column_count; col++) {
            if (std::equal(column.begin(), column.end(), column_info[col].Name().begin(), column_info[col].Name().end(),
                    ichar_equals)) {
                new_selection[col] = true;
            }
        }
    }
    if (std::equal(new_selection.begin(), new_selection.end(), selectedColumns.begin())) {
        return false;
    } else {
        selectedColumns = new_selection;
        return true;
    }
}
void TableColumnFilter::writeSelectionString(size_t column_count, const TableDataCall::ColumnInfo* column_info) {
    std::string sel;
    bool first = true;
    for (size_t col = 0; col < column_count; col++) {
        if (selectedColumns[col]) {
            if (!first) {
                sel += ";";
            } else {
                first = false;
            }
            sel += column_info[col].Name();
        }
    }
    this->selectionStringSlot.Param<core::param::StringParam>()->SetValue(sel);
}
