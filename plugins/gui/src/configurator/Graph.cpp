/*
 * Graph.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Graph.h"


using namespace megamol;
using namespace megamol::gui::graph;


int megamol::gui::graph::Graph::generated_uid = 0;


megamol::gui::graph::Graph::Graph(const std::string& graph_name)
    : modules(), calls(), uid(this->generate_unique_id()), name(graph_name), dirty_flag(true) {

}


megamol::gui::graph::Graph::~Graph(void) {}


bool megamol::gui::graph::Graph::AddModule(const ModuleStockType& stock_modules, const std::string& module_class_name) {

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


bool megamol::gui::graph::Graph::DeleteModule(int module_uid) {

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


bool megamol::gui::graph::Graph::AddCall(
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

bool megamol::gui::graph::Graph::DeleteDisconnectedCalls(void) {

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


bool megamol::gui::graph::Graph::DeleteCall(int call_uid) {

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


// GRAPH PRESENTATIONS ####################################################

megamol::gui::graph::Graph::Presentation::Presentation(void)
    : slot_radius(8.0f)
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
    , process_selected_slot(0) {
}


megamol::gui::graph::Graph::Presentation::~Presentation(void) {}


void megamol::gui::graph::Graph::Presentation::Present(Graph& graph) {

    /*
    // Tab showing one graph
    ImGuiTabItemFlags tab_flags = ImGuiTabItemFlags_None;
    if (graph->IsDirty()) {
        tab_flags |= ImGuiTabItemFlags_UnsavedDocument;
    }
    bool open = true;
    std::string graph_label = "    " + graph->GetName() + "  ###graph" + std::to_string(graph->GetUID());
    if (ImGui::BeginTabItem(graph_label.c_str(), &open, tab_flags)) {

        // Context menu
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::MenuItem("Rename")) {
                this->gui.rename_popup_open = true;
                /// XXX this->gui.rename_popup_string = &graph->GetName();
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
            graph->gui.selected_slot_ptr = nullptr;
            if (graph->gui.selected_module_uid > 0) {
                graph->DeleteModule(graph->gui.selected_module_uid);
            }
            if (graph->gui.selected_call_uid > 0) {
                graph->DeleteCall(graph->gui.selected_call_uid);
            }
        }

        // Register trigger for connecting call
        if ((graph->gui.selected_slot_ptr != nullptr) && (io.MouseReleased[0])) {
            graph->gui.process_selected_slot = 2;
        }

        for (auto& mod : graph->GetGraphModules()) {
            this->update_module_size(graph, mod);
            for (auto& slot_pair : mod->GetCallSlots()) {
                for (auto& slot : slot_pair.second) {
                    this->update_slot_position(graph, slot);
                }
            }
        }

        // Update positions and sizes
        if (this->gui.update_current_graph) {

            this->update_graph_layout(graph);
            this->gui.update_current_graph = false;
        }

        // Draw graph
        this->draw_graph_menu(graph);
        this->draw_graph_canvas(graph);

        ImGui::EndTabItem();
    }
    */
    /*
    if ((this->gui.graph_ptr != nullptr) && (this->gui.graph_ptr->gui.selected_module_uid > 0)) {

        ImGui::BeginChild("splitter_subwindow", ImVec2(0.0f, 0.0f), false, ImGuiWindowFlags_None);
        this->utils.VerticalSplitter(split_thickness, &this->gui.split_width_right, &auto_child);

        this->draw_window_graph(this->gui.split_width_right);
        ImGui::SameLine();
        this->draw_window_parameter_list(auto_child);

        ImGui::EndChild();
    }
    else {
        this->draw_window_graph(auto_child);
    }
    */

}
