/*
 * Graph.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Graph.h"


using namespace megamol;
using namespace megamol::gui::configurator;


int megamol::gui::configurator::Graph::generated_uid = 0;


megamol::gui::configurator::Graph::Graph(const std::string& graph_name)
    : modules(), calls(), uid(this->generate_unique_id()), name(graph_name), dirty_flag(true) {

}


megamol::gui::configurator::Graph::~Graph(void) {}


bool megamol::gui::configurator::Graph::AddModule(const ModuleStockType& stock_modules, const std::string& module_class_name) {

    try {
        bool found = false;
        for (auto& mod : stock_modules) {
            if (module_class_name == mod.class_name) {
                auto mod_ptr = std::make_shared<Module>(this->generate_unique_id());
                mod_ptr->class_name = mod.class_name;
                mod_ptr->description = mod.description;
                mod_ptr->plugin_name = mod.plugin_name;
                mod_ptr->is_view = mod.is_view;
                // mod_ptr->name = "module_name";              /// get from core
                // mod_ptr->full_name = "full_name";           /// get from core
                // mod_ptr->is_view_instance = false;          /// get from core

                for (auto& p : mod.parameters) {
                    Parameter param_slot(this->generate_unique_id(), p.type);
                    param_slot.class_name = p.class_name;
                    param_slot.description = p.description;
                    // param_slot.full_name = "full_name"; /// get from core

                    mod_ptr->parameters.emplace_back(param_slot);
                }
                for (auto& call_slots_type : mod.call_slots) {
                    for (auto& c : call_slots_type.second) {
                        CallSlot call_slot(this->generate_unique_id());
                        call_slot.name = c.name;
                        call_slot.description = c.description;
                        call_slot.compatible_call_idxs = c.compatible_call_idxs;
                        call_slot.type = c.type;

                        mod_ptr->AddCallSlot(std::make_shared<CallSlot>(call_slot));
                    }
                }
                for (auto& call_slot_type_list : mod_ptr->GetCallSlots()) {
                    for (auto& call_slot : call_slot_type_list.second) {
                        call_slot->ConnectParentModule(mod_ptr);
                    }
                }
                this->modules.emplace_back(mod_ptr);

                vislib::sys::Log::DefaultLog.WriteWarn("CREATED MODULE: %s [%s, %s, line %d]\n",
                    mod_ptr->class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);

                this->dirty_flag = true;
                return true;
            }
        }

    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    vislib::sys::Log::DefaultLog.WriteError(
        "Unable to find module: %s [%s, %s, line %d]\n", module_class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
    return false;
}


bool megamol::gui::configurator::Graph::DeleteModule(int module_uid) {

    try {
        for (auto iter = this->modules.begin(); iter != this->modules.end(); iter++) {
            if ((*iter)->uid == module_uid) {
                (*iter)->RemoveAllCallSlots();

                vislib::sys::Log::DefaultLog.WriteWarn("Found %i references pointing to module. [%s, %s, line %d]\n",
                    (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
                assert((*iter).use_count() == 1);

                vislib::sys::Log::DefaultLog.WriteWarn("DELETED MODULE: %s [%s, %s, line %d]\n",
                    (*iter)->class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);

                (*iter).reset();
                this->modules.erase(iter);
                this->DeleteDisconnectedCalls();

                this->dirty_flag = true;
                return true;
            }
        }

    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    vislib::sys::Log::DefaultLog.WriteWarn("Invalid module uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return false;
}


bool megamol::gui::configurator::Graph::AddCall(
    const CallStockType& stock_calls, int call_idx, CallSlotPtrType call_slot_1, CallSlotPtrType call_slot_2) {

    try {
        if ((call_idx > stock_calls.size()) || (call_idx < 0)) {
            vislib::sys::Log::DefaultLog.WriteWarn(
                "Compatible call index out of range. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        auto call = stock_calls[call_idx];
        auto call_ptr = std::make_shared<Call>(this->generate_unique_id());
        call_ptr->class_name = call.class_name;
        call_ptr->description = call.description;
        call_ptr->plugin_name = call.plugin_name;
        call_ptr->functions = call.functions;

        if (call_ptr->ConnectCallSlots(call_slot_1, call_slot_2) && call_slot_1->ConnectCall(call_ptr) &&
            call_slot_2->ConnectCall(call_ptr)) {

            this->calls.emplace_back(call_ptr);

            vislib::sys::Log::DefaultLog.WriteWarn("CREATED and connected CALL: %s [%s, %s, line %d]\n",
                call_ptr->class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);

            this->dirty_flag = true;
        } else {
            // Clean up
            this->DeleteCall(call_ptr->uid);
            return false;
        }

    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}

bool megamol::gui::configurator::Graph::DeleteDisconnectedCalls(void) {

    try {
        // Create separate uid list to avoid iterator conflict when operating on calls list while deleting.
        std::vector<int> call_uids;
        for (auto& call : this->calls) {
            if (!call->IsConnected()) {
                call_uids.emplace_back(call->uid);
            }
        }
        for (auto& id : call_uids) {
            this->DeleteCall(id);
            this->dirty_flag = true;
        }
    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


bool megamol::gui::configurator::Graph::DeleteCall(int call_uid) {

    try {
        for (auto iter = this->calls.begin(); iter != this->calls.end(); iter++) {
            if ((*iter)->uid == call_uid) {
                (*iter)->DisConnectCallSlots();

                vislib::sys::Log::DefaultLog.WriteWarn("Found %i references pointing to call. [%s, %s, line %d]\n",
                    (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
                assert((*iter).use_count() == 1);

                vislib::sys::Log::DefaultLog.WriteWarn("DELETED CALL: %s [%s, %s, line %d]\n",
                    (*iter)->class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);

                (*iter).reset();
                this->calls.erase(iter);

                this->dirty_flag = true;
                return true;
            }
        }
    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    vislib::sys::Log::DefaultLog.WriteWarn("Invalid call uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return false;
}


// GRAPH PRESENTATION ####################################################

megamol::gui::configurator::Graph::Presentation::Presentation(void)
    : utils()
    , slot_radius(8.0f)
    , canvas_position(ImVec2(0.0f, 0.0f))
    , canvas_size(ImVec2(1.0f, 1.0f))
    , canvas_scrolling(ImVec2(0.0f, 0.0f))
    , canvas_zooming(1.0f)
    , canvas_offset(ImVec2(0.0f, 0.0f))
    , show_grid(false)
    , show_call_names(true)
    , show_slot_names(true)
    , selected_module_uid(-1)
    , selected_call_uid(-1)
    , hovered_slot_uid(-1)
    , selected_slot_ptr(nullptr)
    , process_selected_slot(0)
    , update_current_graph(false)
    , split_width(500.0f) {
}


megamol::gui::configurator::Graph::Presentation::~Presentation(void) {}


bool megamol::gui::configurator::Graph::Presentation::Present(megamol::gui::configurator::Graph& graph, float child_width) {

    if (ImGui::GetCurrentContext() == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    ImGui::PushID(graph.GetUID());

    // Tab showing one graph
    ImGuiTabItemFlags tab_flags = ImGuiTabItemFlags_None;
    if (graph.IsDirty()) {
        tab_flags |= ImGuiTabItemFlags_UnsavedDocument;
    }
    bool open = true;
    std::string graph_label = "    " + graph.GetName() + "  ###graph" + std::to_string(graph.GetUID());
    if (ImGui::BeginTabItem(graph_label.c_str(), &open, tab_flags)) {

        // Context menu
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::MenuItem("Rename")) {
                this->rename_popup_open = true;
                this->rename_popup_string = &graph.GetName();
            }
            ImGui::EndPopup();
        }
        // Set selected graph ptr
        if (ImGui::IsItemVisible()) {
            this->gui.graph_ptr = graph;
        }

        // Process module deletion
        if (std::get<1>(this->hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM])) {
            std::get<1>(this->hotkeys[HotkeyIndex::DELETE_GRAPH_ITEM]) = false;
            this->selected_slot_ptr = nullptr;
            if (this->selected_module_uid > 0) {
                graph.DeleteModule(this->selected_module_uid);
            }
            if (this->selected_call_uid > 0) {
                graph.DeleteCall(this->selected_call_uid);
            }
        }

        // Register trigger for connecting call
        if ((graph.gui.selected_slot_ptr != nullptr) && (io.MouseReleased[0])) {
            graph.gui.process_selected_slot = 2;
        }

        for (auto& mod : graph.GetGraphModules()) {
            this->update_module_size(graph, mod);
            for (auto& slot_pair : mod->GetCallSlots()) {
                for (auto& slot : slot_pair.second) {
                    this->update_slot_position(graph, slot);
                }
            }
        }

        // Update positions and sizes
        if (this->update_current_graph) {
            ///XXX this->update_graph_layout(graph);
            this->update_current_graph = false;
        }

        // Draw
        this->menu(graph);
        if ((this->selected_module_uid > 0)) {
            const float split_thickness = 10.0f;
            float child_width_auto = 0.0f;
            ImGui::BeginChild("splitter_subwindow", ImVec2(0.0f, 0.0f), false, ImGuiWindowFlags_None);
            this->utils.VerticalSplitter(split_thickness, &this->split_width, &child_width_auto);

            this->canvas(graph, this->split_width);
            ImGui::SameLine();
            this->parameters(graph, child_width_auto);

            ImGui::EndChild();
        }
        else {
            this->canvas(graph, child_width);
        }

        ImGui::EndTabItem();
    }

    ImGui::PopID();

    return true;
}




void megamol::gui::configurator::Graph::Presentation::menu(megamol::gui::configurator::Graph& graph) {

    const float child_height = ImGui::GetItemsLineHeightWithSpacing() * 1.0f;
    const auto child_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove;

    ImGui::BeginChild("graph_menu", ImVec2(0.0f, child_height), false, child_flags);

    if (ImGui::Button("Reset###reset_scrolling")) {
        this->canvas_scrolling = ImVec2(0.0f, 0.0f);
    }
    ImGui::SameLine();
    ImGui::Text("Scrolling: %.4f,%.4f", this->canvas_scrolling.x, this->canvas_scrolling.y);
    this->utils.HelpMarkerToolTip("Middle Mouse Button");

    ImGui::SameLine();
    if (ImGui::Button("Reset###reset_zooming")) {
        this->canvas_zooming = 1.0f;
    }
    ImGui::SameLine();
    ImGui::Text("Zooming: %.4f", this->canvas_zooming);
    this->utils.HelpMarkerToolTip("Mouse Wheel");

    ImGui::SameLine();
    ImGui::Checkbox("Show Grid", &this->show_grid);

    ImGui::SameLine();
    if (ImGui::Checkbox("Show Call Names", &this->show_call_names)) {
        this->update_current_graph = true;
    }

    ImGui::SameLine();
    if (ImGui::Checkbox("Show Slot Names", &this->show_slot_names)) {
        this->update_current_graph = true;
    }

    ImGui::SameLine();
    if (ImGui::Button("Layout Graph")) {
        this->update_current_graph = true;
    }

    ImGui::EndChild();
}


void megamol::gui::configurator::Graph::Presentation::canvas(megamol::gui::configurator::Graph& graph, float child_width) {



}


void megamol::gui::configurator::Graph::Presentation::parameters(megamol::gui::configurator::Graph& graph, float child_width) {

    ImGui::BeginGroup();

    const float param_child_height = ImGui::GetItemsLineHeightWithSpacing() * 2.25f;
    const auto child_flags = ImGuiWindowFlags_AlwaysUseWindowPadding | ImGuiWindowFlags_NoScrollbar;

    ImGui::BeginChild("parameter_search_child_window", ImVec2(child_width, param_child_height), false, child_flags);

    ImGui::Text("Parameters");
    ImGui::Separator();

    if (std::get<1>(this->hotkeys[HotkeyIndex::PARAMETER_SEARCH])) {
        std::get<1>(this->hotkeys[HotkeyIndex::PARAMETER_SEARCH]) = false;
        this->utils.SetSearchFocus(true);
    }
    std::string help_text = "[" + std::get<0>(this->hotkeys[HotkeyIndex::PARAMETER_SEARCH]).ToString() +
        "] Set keyboard focus to search input field.\n"
        "Case insensitive substring search in parameter names.";
    this->utils.StringSearch("Search", help_text);
    auto search_string = this->utils.GetSearchString();

    ImGui::EndChild();

    ImGui::BeginChild("parameter_list_child_window", ImVec2(child_width, 0.0f), true, ImGuiWindowFlags_HorizontalScrollbar);

    for (auto& mod : graph.GetGraphModules()) {
        if (mod->uid == this->selected_module_uid) {
            for (auto& param : mod->parameters) {
                param.Present();
            }
        }
    }
 
    ImGui::EndChild();

    ImGui::EndGroup();
}
