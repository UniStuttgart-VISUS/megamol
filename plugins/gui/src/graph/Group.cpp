/*
 * Group.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Group.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::Group::Group(ImGuiID uid) : uid(uid), present(), name(), modules(), interfaceslots() {}


megamol::gui::Group::~Group() {

    // Remove all modules from group
    std::vector<ImGuiID> module_uids;
    for (auto& module_ptr : this->modules) {
        module_uids.emplace_back(module_ptr->uid);
    }
    for (auto& module_uid : module_uids) {
        this->RemoveModule(module_uid);
    }
    this->modules.clear();

    // Remove all interface slots from group (should already be empty)
    this->interfaceslots[CallSlotType::CALLER].clear();
    this->interfaceslots[CallSlotType::CALLEE].clear();
}


bool megamol::gui::Group::AddModule(const ModulePtr_t& module_ptr) {

    if (module_ptr == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Pointer to module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Check if module is already part of the group
    for (auto& mod : this->modules) {
        if (mod->uid == module_ptr->uid) {
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[GUI] Module '%s' is already part of group '%s'.\n", mod->name.c_str(), this->name.c_str());
#endif // GUI_VERBOSE
            return false;
        }
    }

    this->modules.emplace_back(module_ptr);

    module_ptr->present.group.uid = this->uid;
    module_ptr->present.group.visible = this->present.ModulesVisible();
    module_ptr->present.group.name = this->name;

    this->present.ForceUpdate();
    this->RestoreInterfaceslots();

#ifdef GUI_VERBOSE
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
        "[GUI] Added module '%s' to group '%s'.\n", module_ptr->name.c_str(), this->name.c_str());
#endif // GUI_VERBOSE
    return true;
}


bool megamol::gui::Group::RemoveModule(ImGuiID module_uid) {

    try {
        for (auto mod_iter = this->modules.begin(); mod_iter != this->modules.end(); mod_iter++) {
            if ((*mod_iter)->uid == module_uid) {

                // Remove call slots from group interface
                for (auto& callslot_map : (*mod_iter)->GetCallSlots()) {
                    for (auto& callslot_ptr : callslot_map.second) {
                        this->InterfaceSlot_RemoveCallSlot(callslot_ptr->uid, true);
                    }
                }

                (*mod_iter)->present.group.uid = GUI_INVALID_ID;
                (*mod_iter)->present.group.visible = false;
                (*mod_iter)->present.group.name = "";

#ifdef GUI_VERBOSE
                megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                    "[GUI] Removed module '%s' from group '%s'.\n", (*mod_iter)->name.c_str(), this->name.c_str());
#endif // GUI_VERBOSE
                (*mod_iter).reset();
                this->modules.erase(mod_iter);

                this->present.ForceUpdate();

                return true;
            }
        }
    } catch (std::exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
        "[GUI] Invalid module uid. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return false;
}


bool megamol::gui::Group::ContainsModule(ImGuiID module_uid) {

    for (auto& mod : this->modules) {
        if (mod->uid == module_uid) {
            return true;
        }
    }
    return false;
}


ImGuiID megamol::gui::Group::AddInterfaceSlot(const CallSlotPtr_t& callslot_ptr, bool auto_add) {

    if (callslot_ptr == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    }

    // Check if call slot is already part of the group
    for (auto& interfaceslot_ptr : this->interfaceslots[callslot_ptr->type]) {
        if (interfaceslot_ptr->ContainsCallSlot(callslot_ptr->uid)) {
            return interfaceslot_ptr->uid;
        }
    }

    // Only add if parent module is already part of the group.
    bool parent_module_group_uid = false;
    if (callslot_ptr->IsParentModuleConnected()) {
        ImGuiID parent_module_uid = callslot_ptr->GetParentModule()->uid;
        for (auto& module_ptr : this->modules) {
            if (parent_module_uid == module_ptr->uid) {
                parent_module_group_uid = true;
            }
        }
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Call slot has no parent module connected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    }

    if (parent_module_group_uid) {
        InterfaceSlotPtr_t interfaceslot_ptr =
            std::make_shared<InterfaceSlot>(megamol::gui::GenerateUniqueID(), auto_add);
        if (interfaceslot_ptr != nullptr) {
            interfaceslot_ptr->present.group.uid = this->uid;
            this->interfaceslots[callslot_ptr->type].emplace_back(interfaceslot_ptr);
#ifdef GUI_VERBOSE
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[GUI] Added interface slot (uid %i) to group '%s'.\n", interfaceslot_ptr->uid, this->name.c_str());
#endif // GUI_VERBOSE

            if (interfaceslot_ptr->AddCallSlot(callslot_ptr, interfaceslot_ptr)) {
                interfaceslot_ptr->present.group.collapsed_view = this->present.IsViewCollapsed();
                return interfaceslot_ptr->uid;
            }
        }
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Parent module of call slot which should be added to group interface "
            "is not part of any group. [%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        return GUI_INVALID_ID;
    }
    return GUI_INVALID_ID;
}


bool megamol::gui::Group::InterfaceSlot_RemoveCallSlot(ImGuiID callslots_uid, bool force) {

    bool retval = false;
    try {
        std::vector<ImGuiID> empty_interfaceslots_uids;
        for (auto& interfaceslot_map : this->interfaceslots) {
            for (auto& interfaceslot_ptr : interfaceslot_map.second) {
                if ((interfaceslot_ptr->IsAutoCreated() || force) &&
                    interfaceslot_ptr->ContainsCallSlot(callslots_uid)) {
                    interfaceslot_ptr->RemoveCallSlot(callslots_uid);
                    retval = true;
                    if (interfaceslot_ptr->IsEmpty()) {
                        empty_interfaceslots_uids.emplace_back(interfaceslot_ptr->uid);
                    }
                }
            }
        }
        // Delete empty interface slots
        for (auto& interfaceslot_uid : empty_interfaceslots_uids) {
            this->DeleteInterfaceSlot(interfaceslot_uid);
        }
    } catch (std::exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return retval;
}


bool megamol::gui::Group::InterfaceSlot_ContainsCallSlot(ImGuiID callslot_uid) {

    for (auto& interfaceslots_map : this->interfaceslots) {
        for (auto& interfaceslot_ptr : interfaceslots_map.second) {
            if (interfaceslot_ptr->ContainsCallSlot(callslot_uid)) {
                return true;
            }
        }
    }
    return false;
}


bool megamol::gui::Group::GetInterfaceSlot(ImGuiID interfaceslot_uid, InterfaceSlotPtr_t& interfaceslot_ptr) {

    if (interfaceslot_uid != GUI_INVALID_ID) {
        for (auto& interfaceslots_map : this->interfaceslots) {
            for (auto& interfaceslot : interfaceslots_map.second) {
                if (interfaceslot->uid == interfaceslot_uid) {
                    interfaceslot_ptr = interfaceslot;
                    return true;
                }
            }
        }
    }
    return false;
}


bool megamol::gui::Group::DeleteInterfaceSlot(ImGuiID interfaceslot_uid) {

    if (interfaceslot_uid != GUI_INVALID_ID) {
        for (auto& interfaceslot_map : this->interfaceslots) {
            for (auto iter = interfaceslot_map.second.begin(); iter != interfaceslot_map.second.end(); iter++) {
                if ((*iter)->uid == interfaceslot_uid) {

                    // Remove all call slots from interface slot
                    std::vector<ImGuiID> callslots_uids;
                    for (auto& callslot_ptr : (*iter)->GetCallSlots()) {
                        callslots_uids.emplace_back(callslot_ptr->uid);
                    }
                    for (auto& callslot_uid : callslots_uids) {
                        (*iter)->RemoveCallSlot(callslot_uid);
                    }

                    if ((*iter).use_count() > 1) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] Unclean deletion. Found %i references pointing to interface slot. [%s, %s, line "
                            "%d]\n",
                            (*iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
                    }

#ifdef GUI_VERBOSE
                    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                        "[GUI] Deleted interface slot (uid %i) from group '%s'.\n", (*iter)->uid, this->name.c_str());
#endif // GUI_VERBOSE

                    (*iter).reset();
                    interfaceslot_map.second.erase(iter);

                    return true;
                }
            }
        }
    }
    return false;
}


bool megamol::gui::Group::ContainsInterfaceSlot(ImGuiID interfaceslot_uid) {

    if (interfaceslot_uid != GUI_INVALID_ID) {
        for (auto& interfaceslots_map : this->interfaceslots) {
            for (auto& interfaceslot : interfaceslots_map.second) {
                if (interfaceslot->uid == interfaceslot_uid) {
                    return true;
                }
            }
        }
    }
    return false;
}


void megamol::gui::Group::RestoreInterfaceslots(void) {

    /// 1] REMOVE connected call slots of group interface if connected module is part of same group
    for (auto& module_ptr : this->modules) {
        // CALLER
        for (auto& callerslot_ptr : module_ptr->GetCallSlots(CallSlotType::CALLER)) {
            for (auto& call_ptr : callerslot_ptr->GetConnectedCalls()) {
                auto calleeslot_ptr = call_ptr->GetCallSlot(CallSlotType::CALLEE);
                if (calleeslot_ptr->IsParentModuleConnected()) {
                    ImGuiID parent_module_group_uid = calleeslot_ptr->GetParentModule()->present.group.uid;
                    if (parent_module_group_uid == this->uid) {
                        this->InterfaceSlot_RemoveCallSlot(calleeslot_ptr->uid);
                    }
                }
            }
        }
        // CALLEE
        for (auto& calleeslot_ptr : module_ptr->GetCallSlots(CallSlotType::CALLEE)) {
            for (auto& call_ptr : calleeslot_ptr->GetConnectedCalls()) {
                auto callerslot_ptr = call_ptr->GetCallSlot(CallSlotType::CALLER);
                if (callerslot_ptr->IsParentModuleConnected()) {
                    ImGuiID parent_module_group_uid = callerslot_ptr->GetParentModule()->present.group.uid;
                    if (parent_module_group_uid == this->uid) {
                        this->InterfaceSlot_RemoveCallSlot(callerslot_ptr->uid);
                    }
                }
            }
        }
    }

    /// 2] ADD connected call slots to group interface if connected module is not part of same group
    for (auto& module_ptr : this->modules) {
        // CALLER
        for (auto& callerslot_ptr : module_ptr->GetCallSlots(CallSlotType::CALLER)) {
            for (auto& call_ptr : callerslot_ptr->GetConnectedCalls()) {
                auto calleeslot_ptr = call_ptr->GetCallSlot(CallSlotType::CALLEE);
                if (calleeslot_ptr->IsParentModuleConnected()) {
                    ImGuiID parent_module_group_uid = calleeslot_ptr->GetParentModule()->present.group.uid;
                    if (parent_module_group_uid != this->uid) {
                        this->AddInterfaceSlot(callerslot_ptr);
                    }
                }
            }
        }
        // CALLEE
        for (auto& calleeslot_ptr : module_ptr->GetCallSlots(CallSlotType::CALLEE)) {
            for (auto& call_ptr : calleeslot_ptr->GetConnectedCalls()) {
                auto callerslot_ptr = call_ptr->GetCallSlot(CallSlotType::CALLER);
                if (callerslot_ptr->IsParentModuleConnected()) {
                    ImGuiID parent_module_group_uid = callerslot_ptr->GetParentModule()->present.group.uid;
                    if (parent_module_group_uid != this->uid) {
                        this->AddInterfaceSlot(calleeslot_ptr);
                    }
                }
            }
        }
    }
}
