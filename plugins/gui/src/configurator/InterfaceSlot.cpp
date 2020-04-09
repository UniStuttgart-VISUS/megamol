/*
 * InterfaceSlot.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "InterfaceSlot.h"
#include "CallSlot.h"


using namespace megamol;
using namespace megamol::gui;
using namespace megamol::gui::configurator;


megamol::gui::configurator::InterfaceSlot::InterfaceSlot(void) {}


megamol::gui::configurator::InterfaceSlot::~InterfaceSlot(void) {}


bool megamol::gui::configurator::InterfaceSlot::AddCallSlot(CallSlotPtrType callslot_ptr) {
    
    try {
        if (callslot_ptr == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        
        if (!(this->ContainsCallSlot(callslot_ptr->uid)) && (this->IsCallSlotCompatible(callslot_ptr))) {
            this->callslots.emplace_back(callslot_ptr);
            
            callslot_ptr->GUI_SetGroupInterface(std::make_shared<InterfaceSlot>(*this));

            vislib::sys::Log::DefaultLog.WriteInfo(
                "Added call slot '%s' to interface slot of group.\n", callslot_ptr->name.c_str());
            return true;   
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
        "Unable to add call slot '%s' to interface slot of group.[%s, %s, line %d]\n", callslot_ptr->name.c_str(), __FILE__, __FUNCTION__, __LINE__);
    return false;
}


bool megamol::gui::configurator::InterfaceSlot::RemoveCallSlot(ImGuiID callslot_uid) {
    
    try {
        for (auto iter = this->callslots.begin(); iter != this->callslots.end(); iter++) {
            if ((*iter)->uid == callslot_uid) {
                
                (*iter)->GUI_SetGroupInterface(nullptr);

                vislib::sys::Log::DefaultLog.WriteInfo(
                    "Removed call slot '%s' from interface slot of group.\n", (*iter)->name.c_str());
                (*iter).reset();
                this->callslots.erase(iter);

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
    return false;
}


bool megamol::gui::configurator::InterfaceSlot::ContainsCallSlot(ImGuiID callslot_uid) {
    
    for (auto& callslot : this->callslots) {
        if (callslot_uid == callslot->uid) {
            return true;
        }
    }
    return false;        
}


bool megamol::gui::configurator::InterfaceSlot::IsCallSlotCompatible(CallSlotPtrType callslot_ptr) {
    
    if (callslot_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    
    // Check for compatibility (with all available call slots...)
    size_t compatible = 0;
    for (auto& callslot : this->callslots) {
        if ((callslot_ptr->type == callslot->type) && (callslot_ptr->compatible_call_idxs == callslot->compatible_call_idxs)) {
            compatible++;
        }
    }
    
    bool retval = (compatible == this->callslots.size());

    if ((compatible > 0) && (compatible != this->callslots.size())) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Interface slot contains incompatible call slots. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    return retval;
}


bool megamol::gui::configurator::InterfaceSlot::IsEmpty(void) { 
    
    return (callslots.size() == 0); 
}
    


// GROUP INTERFACE SLOT PRESENTATION ###########################################

megamol::gui::configurator::InterfaceSlot::Presentation::Presentation(void)
    : position()
    , utils()
    , selected(false){

}


megamol::gui::configurator::InterfaceSlot::Presentation::~Presentation(void) {}


void megamol::gui::configurator::InterfaceSlot::Presentation::Present(
    megamol::gui::configurator::InterfaceSlot& inout_interfaceslot, megamol::gui::GraphItemsStateType& state, bool collapsed_view) {

    if (ImGui::GetCurrentContext() == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }
    ImGuiStyle& style = ImGui::GetStyle();
    
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    try {

        // Compatible Call Highlight
        //if (CallSlot::CheckCompatibleAvailableCallIndex(state.interact.callslot_compat_ptr, inout_callslot) != GUI_INVALID_ID) {
        //    slot_color = COLOR_SLOT_COMPATIBLE;
        //}


    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}


void megamol::gui::configurator::InterfaceSlot::Presentation::SetPosition(InterfaceSlot& inout_interfaceslot, ImVec2 pos) {
    
    this->position = pos;
}
