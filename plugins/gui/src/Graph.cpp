/*
 * Graph.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Graph.h"


using namespace megamol;
using namespace megamol::gui;
using vislib::sys::Log;


// PARAM SLOT #################################################################


// CALL SLOT ##################################################################

megamol::gui::Graph::CallSlot::CallSlot(int uid) : uid(uid) {
    this->parent_module.reset();
    connected_calls.clear();
}


megamol::gui::Graph::CallSlot::~CallSlot() {

    // Call separately and check for reference count
    this->DisConnectCalls();
    this->DisConnectParentModule();
}


bool megamol::gui::Graph::CallSlot::UpdateGuiPos(void) {

    if (this->ParentModuleConnected()) {
        auto slot_count = this->GetParentModule()->GetCallSlots(this->type).size();
        size_t slot_idx = 0;
        for (size_t idx = 0; idx < slot_count; idx++) {
            if (this->name == this->GetParentModule()->GetCallSlots(this->type)[idx]->name) {
                slot_idx = idx;
            }
        }
        auto pos = this->GetParentModule()->gui.position;
        auto size = this->GetParentModule()->gui.size;
        this->gui.position = ImVec2(pos.x + ((this->type == Graph::CallSlotType::CALLER) ? (size.x) : (0.0f)),
            pos.y + size.y * ((float)slot_idx + 1) / ((float)slot_count + 1));
        return true;
    }
    return false;
}


bool megamol::gui::Graph::CallSlot::CallsConnected(void) const {

    /// TEMP Check for unclean references
    for (auto& call_ptr : this->connected_calls) {
        if (call_ptr == nullptr) {
            throw std::invalid_argument("Pointer to connected call is nullptr.");
        }
    }
    return (!this->connected_calls.empty());
}


bool megamol::gui::Graph::CallSlot::ConnectCall(Graph::CallPtrType call) {

    if (call == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to given call is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (this->type == Graph::CallSlotType::CALLER) {
        if (this->connected_calls.size() > 0) {
            vislib::sys::Log::DefaultLog.WriteWarn(
                "Caller slots can only be connected to one call. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
            return false;
        }
    }
    this->connected_calls.emplace_back(call);
    return true;
}


bool megamol::gui::Graph::CallSlot::DisConnectCall(int call_uid, bool called_by_call) {

    try {
        for (auto call_iter = this->connected_calls.begin(); call_iter != this->connected_calls.end(); call_iter++) {
            if ((*call_iter) == nullptr) {
                vislib::sys::Log::DefaultLog.WriteWarn(
                    "Call is already disconnected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            } else {
                if ((*call_iter)->uid == call_uid) {
                    if (!called_by_call) {
                        (*call_iter)->DisConnectCallSlots();
                    }
                    (*call_iter).reset();
                    this->connected_calls.erase(call_iter);
                    return true;
                }
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
    return false;
}


bool megamol::gui::Graph::CallSlot::DisConnectCalls(void) {

    try {
        // Since connected calls operate on this list for disconnecting slots
        // a local copy of the connected calls is required.
        auto connected_calls_copy = this->connected_calls;
        for (auto& call_ptr : connected_calls_copy) {
            if (call_ptr == nullptr) {
                vislib::sys::Log::DefaultLog.WriteWarn(
                    "Call is already disconnected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            } else {
                call_ptr->DisConnectCallSlots();
            }
        }
        this->connected_calls.clear();
        connected_calls_copy.clear();
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


const std::vector<Graph::CallPtrType> megamol::gui::Graph::CallSlot::GetConnectedCalls(void) {

    /// TEMP Check for unclean references
    for (auto& call_ptr : this->connected_calls) {
        if (call_ptr == nullptr) {
            throw std::invalid_argument("Pointer to connected call is nullptr.");
        }
    }

    return this->connected_calls;
}


bool megamol::gui::Graph::CallSlot::ParentModuleConnected(void) const { return (this->parent_module != nullptr); }


bool megamol::gui::Graph::CallSlot::ConnectParentModule(Graph::ModulePtrType parent_module) {

    if (parent_module == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to given parent module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    if (this->parent_module != nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to parent module is already set. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->parent_module = parent_module;
    return true;
}


bool megamol::gui::Graph::CallSlot::DisConnectParentModule(void) {

    if (parent_module == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to parent module is already nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->parent_module.reset();
    return true;
}


const Graph::ModulePtrType megamol::gui::Graph::CallSlot::GetParentModule(void) {

    if (this->parent_module == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Returned pointer to parent module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    return this->parent_module;
}


// CALL #######################################################################

megamol::gui::Graph::Call::Call(int uid) : uid(uid) {

    this->connected_call_slots.clear();
    this->connected_call_slots.emplace(Graph::CallSlotType::CALLER, nullptr);
    this->connected_call_slots.emplace(Graph::CallSlotType::CALLEE, nullptr);
}


megamol::gui::Graph::Call::~Call() { this->DisConnectCallSlots(); }


bool megamol::gui::Graph::Call::IsConnected(void) {

    unsigned int not_connected = 0;
    for (auto& call_slot_map : this->connected_call_slots) {
        if (call_slot_map.second != nullptr) {
            not_connected++;
        }
    }
    if (not_connected == 1) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Only one call slot is connected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return (not_connected == 2);
}


bool megamol::gui::Graph::Call::ConnectCallSlots(
    Graph::CallSlotPtrType call_slot_1, Graph::CallSlotPtrType call_slot_2) {

    if ((call_slot_1 == nullptr) || (call_slot_2 == nullptr)) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to given call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (call_slot_1->type == call_slot_2->type) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Call slots must have different type. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (call_slot_1->GetParentModule() == call_slot_2->GetParentModule()) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Call slots must have different parent module. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if ((this->connected_call_slots[call_slot_1->type] != nullptr) ||
        (this->connected_call_slots[call_slot_2->type] != nullptr)) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Call is already connected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    this->connected_call_slots[call_slot_1->type] = call_slot_1;
    this->connected_call_slots[call_slot_2->type] = call_slot_2;

    return true;
}


bool megamol::gui::Graph::Call::DisConnectCallSlots(void) {

    try {
        for (auto& call_slot_map : this->connected_call_slots) {
            if (call_slot_map.second == nullptr) {
                vislib::sys::Log::DefaultLog.WriteWarn(
                    "Call slot is already disconnected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            } else {
                call_slot_map.second->DisConnectCall(this->uid, true);
                call_slot_map.second.reset();
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
    return true;
}


const Graph::CallSlotPtrType megamol::gui::Graph::Call::GetCallSlot(Graph::CallSlotType type) {

    if (this->connected_call_slots[type] == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Returned pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    return this->connected_call_slots[type];
}


// MODULE #####################################################################

megamol::gui::Graph::Module::Module(int uid) : uid(uid) {

    this->call_slots.clear();
    this->call_slots.emplace(Graph::CallSlotType::CALLER, std::vector<Graph::CallSlotPtrType>());
    this->call_slots.emplace(Graph::CallSlotType::CALLEE, std::vector<Graph::CallSlotPtrType>());
}


megamol::gui::Graph::Module::~Module() { this->RemoveAllCallSlots(); }


bool megamol::gui::Graph::Module::AddCallSlot(Graph::CallSlotPtrType call_slot) {

    if (call_slot == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to given call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    auto type = call_slot->type;
    for (auto& call_slot_ptr : this->call_slots[type]) {
        if (call_slot_ptr == call_slot) {
            throw std::invalid_argument("Pointer to call slot already registered in modules call slot list.");
        }
    }
    this->call_slots[type].emplace_back(call_slot);
    return true;
}


bool megamol::gui::Graph::Module::RemoveAllCallSlots(void) {

    try {
        for (auto& call_slots_map : this->call_slots) {
            for (auto& call_slot_ptr : call_slots_map.second) {
                if (call_slot_ptr == nullptr) {
                    vislib::sys::Log::DefaultLog.WriteWarn(
                        "Call slot is already disconnected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                } else {
                    call_slot_ptr->DisConnectCalls();
                    call_slot_ptr->DisConnectParentModule();

                    vislib::sys::Log::DefaultLog.WriteWarn(
                        "Found %i references pointing to call slot. [%s, %s, line %d]\n", call_slot_ptr.use_count(),
                        __FILE__, __FUNCTION__, __LINE__);
                    assert(call_slot_ptr.use_count() == 1);

                    call_slot_ptr.reset();
                }
            }
            call_slots_map.second.clear();
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


const std::vector<Graph::CallSlotPtrType> megamol::gui::Graph::Module::GetCallSlots(Graph::CallSlotType type) {

    if (this->call_slots[type].empty()) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Returned call slot list is empty. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    return this->call_slots[type];
}


const std::map<Graph::CallSlotType, std::vector<Graph::CallSlotPtrType>> megamol::gui::Graph::Module::GetCallSlots(
    void) {

    return this->call_slots;
}


// GRAPH ######################################################################

megamol::gui::Graph::Graph(int graph_uid, const std::string& graph_name)
    : gui(), modules(), calls(), uid(graph_uid), name(graph_name), dirty_flag(true), generated_uid(0) {

    this->gui.slot_radius = 8.0f;
    this->gui.canvas_position = ImVec2(0.0f, 0.0f);
    this->gui.canvas_size = ImVec2(0.0f, 0.0f);
    this->gui.canvas_scrolling = ImVec2(0.0f, 0.0f);
    this->gui.canvas_zooming = 1.0f;
    this->gui.canvas_offset = ImVec2(0.0f, 0.0f);
    this->gui.show_grid = false;
    this->gui.show_call_names = true;
    this->gui.show_slot_names = true;
    this->gui.selected_module_uid = -1;
    this->gui.selected_call_uid = -1;
    this->gui.hovered_slot_uid = -1;
    this->gui.selected_slot_ptr = nullptr;
    this->gui.process_selected_slot = 0;
    // Layout not empty loaded graphs initially.
    this->gui.update_layout = true;
}


megamol::gui::Graph::~Graph(void) {}


bool megamol::gui::Graph::AddModule(Graph::ModuleStockType& stock_modules, const std::string& module_class_name) {

    try {
        bool found = false;
        for (auto& mod : stock_modules) {
            if (module_class_name == mod.class_name) {
                auto mod_ptr = std::make_shared<Graph::Module>(this->get_unique_id());
                mod_ptr->class_name = mod.class_name;
                mod_ptr->description = mod.description;
                mod_ptr->plugin_name = mod.plugin_name;
                mod_ptr->is_view = mod.is_view;
                mod_ptr->name = "module_name";     /// TODO get from core
                mod_ptr->full_name = "full_name";  /// TODO get from core
                mod_ptr->is_view_instance = false; /// TODO get from core
                mod_ptr->gui.update_size = true;
                mod_ptr->gui.position = ImVec2(0.0f, 0.0f); // Initialized in configurator
                mod_ptr->gui.size = ImVec2(0.0f, 0.0f);     // Initialized in configurator
                mod_ptr->gui.class_label = "";              // Initialized in configurator
                mod_ptr->gui.name_label = "";               // Initialized in configurator
                for (auto& p : mod.param_slots) {
                    Graph::ParamSlot param_slot(this->get_unique_id());
                    param_slot.class_name = p.class_name;
                    param_slot.description = p.description;
                    param_slot.type = p.type;
                    param_slot.full_name = "full_name";      /// TODO get from core
                    param_slot.value_string = "param_value"; /// TODO get from core
                    mod_ptr->param_slots.emplace_back(param_slot);
                }
                for (auto& call_slots_type : mod.call_slots) {
                    for (auto& c : call_slots_type.second) {
                        Graph::CallSlot call_slot(this->get_unique_id());
                        call_slot.name = c.name;
                        call_slot.description = c.description;
                        call_slot.compatible_call_idxs = c.compatible_call_idxs;
                        call_slot.type = c.type;
                        call_slot.gui.position = ImVec2(0.0f, 0.0f); // Initialized in configurator
                        mod_ptr->AddCallSlot(std::make_shared<Graph::CallSlot>(call_slot));
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


bool megamol::gui::Graph::DeleteModule(int module_uid) {

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


bool megamol::gui::Graph::AddCall(Graph::CallStockType& stock_calls, int call_idx, Graph::CallSlotPtrType call_slot_1,
    Graph::CallSlotPtrType call_slot_2) {

    try {
        if ((call_idx > stock_calls.size()) || (call_idx < 0)) {
            vislib::sys::Log::DefaultLog.WriteWarn(
                "Compatible call index out of range. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        auto call = stock_calls[call_idx];
        auto call_ptr = std::make_shared<Graph::Call>(this->get_unique_id());
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

bool megamol::gui::Graph::DeleteDisconnectedCalls(void) {

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


bool megamol::gui::Graph::DeleteCall(int call_uid) {

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
