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

megamol::gui::Graph::ParamSlot::ParamSlot(int gui_id) : gui_uid(gui_id) {}


megamol::gui::Graph::ParamSlot::~ParamSlot() {}


// CALL SLOT ##################################################################

megamol::gui::Graph::CallSlot::CallSlot(int gui_id) : gui_uid(gui_id) {
    this->parent_module.reset();
    connected_calls.clear();
}


megamol::gui::Graph::CallSlot::~CallSlot() {

    this->DisConnectCalls();
    this->RemoveParentModule();
}


ImVec2 megamol::gui::Graph::CallSlot::GetGuiPos(void) {

    /// TODO Calculate once and store permanently?

    ImVec2 retpos = ImVec2(-1.0f, -1.0f);
    if (this->ParentModuleConnected()) {
        auto slot_count = this->GetParentModule()->GetCallSlots(this->type).size();
        size_t slot_idx = 0;
        for (size_t i = 0; i < slot_count; i++) {
            if (this->name == this->GetParentModule()->GetCallSlots(this->type)[i]->name) {
                slot_idx = i;
            }
        }
        auto pos = this->parent_module->gui.position;
        auto size = this->parent_module->gui.size;
        retpos = ImVec2(pos.x + ((this->type == Graph::CallSlotType::CALLER) ? (size.x) : (0.0f)),
            pos.y + size.y * ((float)slot_idx + 1) / ((float)slot_count + 1));
    }
    return retpos;
}


bool megamol::gui::Graph::CallSlot::CallsConnected(void) const {

    // TEMP Check for unclean references
    for (auto& cc : this->connected_calls) {
        if (cc == nullptr) {
            throw std::invalid_argument("Pointer to connected call is nullptr.");
        }
    }
    return (!this->connected_calls.empty());
}


bool megamol::gui::Graph::CallSlot::ConnectCall(Graph::CallPtr call) {

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
    // (Connecting same call to callee slot several times is allowed)
    this->connected_calls.emplace_back(call);
    return true;
}


bool megamol::gui::Graph::CallSlot::DisConnectCall(Graph::CallPtr call) {

    if (call == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to given call is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    for (auto i = this->connected_calls.begin(); i != this->connected_calls.end(); i++) {
        if ((*i) == call) {
            (*i).reset();
            this->connected_calls.erase(i);
            return true;
        }
    }
    return false;
}


bool megamol::gui::Graph::CallSlot::DisConnectCalls(void) {

    for (auto& c : this->connected_calls) {
        c.reset();
    }
    this->connected_calls.clear();
    return true;
}


const std::vector<Graph::CallPtr> megamol::gui::Graph::CallSlot::GetConnectedCalls(void) {

    // TEMP Check for unclean references
    for (auto& c : this->connected_calls) {
        if (c == nullptr) {
            throw std::invalid_argument("Pointer to connected call is nullptr.");
        }
    }

    return this->connected_calls;
}


bool megamol::gui::Graph::CallSlot::ParentModuleConnected(void) const { return (this->parent_module != nullptr); }


bool megamol::gui::Graph::CallSlot::AddParentModule(Graph::ModulePtr parent_module) {

    if (parent_module == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to given parent module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (this->parent_module != nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to parent module is already set. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->parent_module = parent_module;
    return true;
}


bool megamol::gui::Graph::CallSlot::RemoveParentModule(void) {

    if (parent_module == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to arent module is already nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->parent_module.reset();
    return true;
}


const Graph::ModulePtr megamol::gui::Graph::CallSlot::GetParentModule(void) {

    if (this->parent_module == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Returned pointer to parent module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    return this->parent_module;
}


// CALL #######################################################################

megamol::gui::Graph::Call::Call(int gui_id) : gui_uid(gui_id) {

    this->connected_call_slots.clear();
    this->connected_call_slots.emplace(Graph::CallSlotType::CALLER, nullptr);
    this->connected_call_slots.emplace(Graph::CallSlotType::CALLEE, nullptr);
}


megamol::gui::Graph::Call::~Call() { this->DisConnectCallSlots(); }


bool megamol::gui::Graph::Call::IsConnected(void) {

    unsigned int not_connected = 0;
    for (auto& c : this->connected_call_slots) {
        if (c.second != nullptr) {
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


bool megamol::gui::Graph::Call::ConnectCallSlot(Graph::CallSlotPtr call_slot) {

    if (call_slot == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to given call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (this->connected_call_slots[call_slot->type] != nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Call slot is already connected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->connected_call_slots[call_slot->type] = call_slot;
    return true;
}


bool megamol::gui::Graph::Call::DisConnectCallSlot(Graph::CallSlotType type) {

    if (this->connected_call_slots[type] == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Call slot is already disconnected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->connected_call_slots[type]->GetParentModule()->RemoveCallSlot(this->connected_call_slots[type]);
    this->connected_call_slots[type].reset();
    return true;
}


bool megamol::gui::Graph::Call::DisConnectCallSlots(void) {

    bool retval = true;
    for (auto& cs : this->connected_call_slots) {
        retval = retval && this->DisConnectCallSlot(cs.first);
    }
    return retval;
}


const Graph::CallSlotPtr megamol::gui::Graph::Call::GetCallSlot(Graph::CallSlotType type) {

    if (this->connected_call_slots[type] == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Returned pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    return this->connected_call_slots[type];
}


// MODULE #####################################################################


megamol::gui::Graph::Module::Module(int gui_id) : gui_uid(gui_id) {

    this->call_slots.clear();
    this->call_slots.emplace(Graph::CallSlotType::CALLER, std::vector<Graph::CallSlotPtr>());
    this->call_slots.emplace(Graph::CallSlotType::CALLEE, std::vector<Graph::CallSlotPtr>());
}


megamol::gui::Graph::Module::~Module() { this->RemoveAllCallSlots(); }


bool megamol::gui::Graph::Module::AddCallSlot(Graph::CallSlotPtr call_slot) {

    if (call_slot == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Poniter to given call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    auto type = call_slot->type;
    for (auto& cs : this->call_slots[type]) {
        if (cs == call_slot) {
            throw std::invalid_argument("Pointer to call slot already registered in modules call slot list.");
        }
    }
    this->call_slots[type].emplace_back(call_slot);
    return true;
}


bool megamol::gui::Graph::Module::RemoveCallSlot(Graph::CallSlotPtr call_slot) {

    if (call_slot == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Poniter to given call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    auto type = call_slot->type;
    for (auto iter = this->call_slots[type].begin(); iter != this->call_slots[type].end(); iter++) {
        if ((*iter) == call_slot) {
            (*iter)->DisConnectCalls();
            (*iter)->RemoveParentModule();

            vislib::sys::Log::DefaultLog.WriteWarn("Found %i references pointing to call slot. [%s, %s, line %d]\n",
                (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
            assert((*iter).use_count() == 1);
            (*iter).reset();
            this->call_slots[type].erase(iter);
            return true;
        }
    }
    vislib::sys::Log::DefaultLog.WriteWarn(
        "Call slot not found for removal. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return false;
}


bool megamol::gui::Graph::Module::RemoveAllCallSlots(Graph::CallSlotType type) {

    for (auto& cs : this->call_slots[type]) {
        cs->DisConnectCalls();
        cs->RemoveParentModule();
        vislib::sys::Log::DefaultLog.WriteWarn("Found %i references pointing to call slot. [%s, %s, line %d]\n",
            cs.use_count(), __FILE__, __FUNCTION__, __LINE__);
        assert(cs.use_count() == 1);
        cs.reset();
    }
    this->call_slots[type].clear();
    return true;
}


bool megamol::gui::Graph::Module::RemoveAllCallSlots(void) {

    return (this->RemoveAllCallSlots(Graph::CallSlotType::CALLEE) && RemoveAllCallSlots(Graph::CallSlotType::CALLER));
}


const std::vector<Graph::CallSlotPtr> megamol::gui::Graph::Module::GetCallSlots(Graph::CallSlotType type) {

    if (this->call_slots[type].empty()) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Returned call slot list is empty. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    return this->call_slots[type];
}


const std::map<Graph::CallSlotType, std::vector<Graph::CallSlotPtr>> megamol::gui::Graph::Module::GetCallSlots(void) {

    return this->call_slots;
}


// GRAPH ######################################################################

megamol::gui::Graph::Graph(void) :  modules(), calls() {}


megamol::gui::Graph::~Graph(void) {}



bool megamol::gui::Graph::AddModule(const std::string& module_class_name) {

    try {
        bool found = false;
        for (auto& mod : this->modules_stock) {
            if (module_class_name == mod.class_name) {
                auto mod_ptr = std::make_shared<Graph::Module>(this->get_unique_id());
                mod_ptr->class_name = mod.class_name;
                mod_ptr->description = mod.description;
                mod_ptr->plugin_name = mod.plugin_name;
                mod_ptr->is_view = mod.is_view;
                /// TODO Set from core:
                mod_ptr->name = "module_name";
                /// TODO Set from core:
                mod_ptr->full_name = "full_name";
                /// TODO Set from core:
                mod_ptr->instance = "instance";
                mod_ptr->gui.initialized = false;
                // mod_ptr->gui.position = ImVec2(-1.0f, -1.0f);
                // mod_ptr->gui.size = ImVec2(-1.0f, -1.0f);
                for (auto& p : mod.param_slots) {
                    Graph::ParamSlot param_slot(this->get_unique_id());
                    param_slot.class_name = p.class_name;
                    param_slot.description = p.description;
                    param_slot.type = p.type;
                    /// TODO Set from core:
                    param_slot.full_name = "full_name";
                    mod_ptr->param_slots.emplace_back(param_slot);
                }
                for (auto& call_slots_type : mod.call_slots) {
                    for (auto& c : call_slots_type.second) {
                        Graph::CallSlot call_slot(this->get_unique_id());
                        call_slot.name = c.name;
                        call_slot.description = c.description;
                        call_slot.compatible_call_idxs = c.compatible_call_idxs;
                        call_slot.type = c.type;
                        mod_ptr->AddCallSlot(std::make_shared<Graph::CallSlot>(call_slot));
                    }
                }
                for (auto& call_slot_type_list : mod_ptr->GetCallSlots()) {
                    for (auto& call_slot : call_slot_type_list.second) {
                        call_slot->AddParentModule(mod_ptr);
                    }
                }
                this->modules.emplace_back(mod_ptr);
                found = true;
                break;
            }
        }
        if (!found) {
            vislib::sys::Log::DefaultLog.WriteError("Unable to find module: %s [%s, %s, line %d]\n",
                module_class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
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


bool megamol::gui::Graph::DeleteModule(int module_uid) {

    try {
        for (auto iter = this->modules.begin(); iter != this->modules.end();  iter++) {
            if ((*iter)->gui_uid == module_uid) {
                (*iter)->RemoveAllCallSlots();
                vislib::sys::Log::DefaultLog.WriteWarn("Found %i references pointing to module. [%s, %s, line %d]\n",
                    (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
                assert((*iter).use_count() == 1);
                this->modules.erase(iter);
                return true;
            }
        }
        // Search for calls with lost connection
        /// Create separate uid list to avoid iterator conflict when operating on calls lsit while deleting.
        std::vector<int> call_uids;
        for (auto& c : this->calls) {
            if (!c->IsConnected()) {
                call_uids.emplace_back(c->gui_uid);
            }
        }
        for (auto& id : call_uids) {
            this->DeleteCall(id);
        }

    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    vislib::sys::Log::DefaultLog.WriteWarn(
        "Module gui index not found. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return false;
}


bool megamol::gui::Graph::AddCall(int call_idx, CallSlotPtr call_slot_1, CallSlotPtr call_slot_2) {

    try {
        if ((call_idx > this->calls_stock.size()) || (call_idx < 0)) {
            vislib::sys::Log::DefaultLog.WriteWarn(
                "Compatible call index out of range. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        auto call = this->calls_stock[call_idx];
        auto call_ptr = std::make_shared<Graph::Call>(this->get_unique_id());
        call_ptr->class_name = call.class_name;
        call_ptr->description = call.description;
        call_ptr->plugin_name = call.plugin_name;
        call_ptr->functions = call.functions;

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

        /// TODO Check if call is assigned second time ...

        if (call_ptr->ConnectCallSlot(call_slot_1)) {
            if (call_ptr->ConnectCallSlot(call_slot_2)) {
                this->calls.emplace_back(call_ptr);
            } else {
                // Clean up if connection to call slots fails
                call_ptr->DisConnectCallSlot(call_slot_2->type);
            }
        }

        vislib::sys::Log::DefaultLog.WriteWarn(
            "CONNECTED: %s [%s, %s, line %d]\n", call_ptr->class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
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
        for (auto iter = this->calls.begin(); iter != this->calls.end();
             iter++) {
            if ((*iter)->gui_uid == call_uid) {
                (*iter)->DisConnectCallSlots();
                vislib::sys::Log::DefaultLog.WriteWarn("Found %i references pointing to call. [%s, %s, line %d]\n",
                    (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
                assert((*iter).use_count() == 1);
                this->calls.erase(iter);
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

    vislib::sys::Log::DefaultLog.WriteWarn(
        "Call gui index not found. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return false;
}


const Graph::ModuleGraphType& megamol::gui::Graph::GetGraphModules(void) {

    return this->modules;
}


const Graph::CallGraphType& megamol::gui::Graph::GetGraphCalls(void) {

    return this->calls;
}
