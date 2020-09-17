/*
 * GraphCollection.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "GraphCollectionPresentation.h"

#include "GraphCollection.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::GraphCollectionPresentation::GraphCollectionPresentation(void)
    : file_browser(), graph_delete_uid(GUI_INVALID_ID) {}


megamol::gui::GraphCollectionPresentation::~GraphCollectionPresentation(void) {}


void megamol::gui::GraphCollectionPresentation::Present(
    megamol::gui::GraphCollection& inout_graph_collection, GraphState_t& state) {

    try {
        if (ImGui::GetCurrentContext() == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return;
        }

        const auto child_flags = ImGuiWindowFlags_None;
        ImGui::BeginChild("graph_child_window", ImVec2(state.graph_width, 0.0f), true, child_flags);

        // Assuming only one closed tab/graph per frame.
        bool popup_close_unsaved = false;

        // Draw Graphs
        ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_Reorderable;
        ImGui::BeginTabBar("Graphs", tab_bar_flags);

        for (auto& graph : inout_graph_collection.GetGraphs()) {

            // Draw graph
            graph->PresentGUI(state);

            // Do not delete graph while looping through graphs list
            if (state.graph_delete) {
                this->graph_delete_uid = state.graph_selected_uid;
                if (graph->IsDirty()) {
                    popup_close_unsaved = true;
                }
                state.graph_delete = false;
            }

            // Catch call drop event and create new call(s) ...
            if (const ImGuiPayload* payload = ImGui::GetDragDropPayload()) {
                if (payload->IsDataType(GUI_DND_CALLSLOT_UID_TYPE) && payload->IsDelivery()) {
                    ImGuiID* dragged_slot_uid_ptr = (ImGuiID*)payload->Data;
                    auto drag_slot_uid = (*dragged_slot_uid_ptr);
                    auto drop_slot_uid = graph->present.GetDropSlot();
                    graph->AddCall(inout_graph_collection.GetCallsStock(), drag_slot_uid, drop_slot_uid);
                }
            }
        }
        ImGui::EndTabBar();

        // Save selected graph
        this->SaveProjectToFile(state.graph_save, inout_graph_collection, state);
        state.graph_save = false;

        // Delete selected graph when tab is closed and unsaved changes should be discarded.
        bool confirmed = false;
        bool aborted = false;
        bool popup_open = MinimalPopUp::PopUp(
            "Closing unsaved Project", popup_close_unsaved, "Discard changes?", "Yes", confirmed, "No", aborted);
        if (this->graph_delete_uid != GUI_INVALID_ID) {
            if (aborted) {
                this->graph_delete_uid = GUI_INVALID_ID;
            } else if (confirmed || !popup_open) {
                inout_graph_collection.DeleteGraph(graph_delete_uid);
                this->graph_delete_uid = GUI_INVALID_ID;
                state.graph_selected_uid = GUI_INVALID_ID;
            }
        }

        ImGui::EndChild();

    } catch (std::exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}


void megamol::gui::GraphCollectionPresentation::SaveProjectToFile(
    bool open_popup, GraphCollection& inout_graph_collection, GraphState_t& state) {

    bool confirmed, aborted;
    bool popup_failed = false;
    std::string project_filename;
    GraphPtr_t graph_ptr;
    if (inout_graph_collection.GetGraph(state.graph_selected_uid, graph_ptr)) {
        project_filename = graph_ptr->GetFilename();
    }
    if (this->file_browser.PopUp(
            FileBrowserWidget::FileBrowserFlag::SAVE, "Save Editor Project", open_popup, project_filename)) {
        popup_failed = !inout_graph_collection.SaveProjectToFile(state.graph_selected_uid, project_filename, false);
    }
    MinimalPopUp::PopUp("Failed to Save Project", popup_failed, "See console log output for more information.", "",
        confirmed, "Cancel", aborted);
}
