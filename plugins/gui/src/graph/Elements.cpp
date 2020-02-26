/*
 * Graph.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Elements.h"


using namespace megamol;
using namespace megamol::gui::graph;


// PARAM SLOT #################################################################

megamol::gui::graph::Parameter::Parameter(int uid, megamol::gui::graph::Parameter::ParamType type)
    : uid(uid), type(type), minval(), maxval(), storage(), value() {

    // Initialize variant types which should/can not be changed afterwards.
    // Default ctor of variants initializes std::monostate.
    switch (this->type) {
    case (Parameter::ParamType::BOOL): {
        this->value = bool(false);
    } break;
    case (Parameter::ParamType::BUTTON): {
        this->storage = megamol::core::view::KeyCode();
    } break;
    case (Parameter::ParamType::COLOR): {
        this->value = megamol::core::param::ColorParam::ColorType();
    } break;
    case (Parameter::ParamType::ENUM): {
        this->value = int(0);
        this->storage = vislib::Map<int, vislib::TString>();
    } break;
    case (Parameter::ParamType::FILEPATH): {
        this->value = std::string();
    } break;
    case (Parameter::ParamType::FLEXENUM): {
        this->value = std::string();
        this->storage = megamol::core::param::FlexEnumParam::Storage_t();
    } break;
    case (Parameter::ParamType::FLOAT): {
        this->value = float(0.0f);
        this->minval = -FLT_MAX;
        this->maxval = FLT_MAX;
    } break;
    case (Parameter::ParamType::INT): {
        this->value = int();
        this->minval = INT_MIN;
        this->maxval = INT_MAX;
    } break;
    case (Parameter::ParamType::STRING): {
        this->value = std::string();
    } break;
    case (Parameter::ParamType::TERNARY): {
        this->value = vislib::math::Ternary();
    } break;
    case (Parameter::ParamType::TRANSFERFUNCTION): {
        this->value = std::string();
    } break;
    case (Parameter::ParamType::VECTOR2F): {
        this->value = glm::vec2();
        this->minval = glm::vec2(-FLT_MAX, -FLT_MAX);
        this->maxval = glm::vec2(FLT_MAX, FLT_MAX);
    } break;
    case (Parameter::ParamType::VECTOR3F): {
        this->value = glm::vec3();
        this->minval = glm::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        this->maxval = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    } break;
    case (Parameter::ParamType::VECTOR4F): {
        this->value = glm::vec4();
        this->minval = glm::vec4(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
        this->maxval = glm::vec4(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
    } break;
    default:
        break;
    }
}


std::string megamol::gui::graph::Parameter::GetValueString(void) {
    std::string value_string = "UNKNOWN PARAMETER TYPE";
    auto visitor = [this, &value_string](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, bool>) {
            auto param = megamol::core::param::BoolParam(arg);
            value_string = std::string(param.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, megamol::core::param::ColorParam::ColorType>) {
            auto param = megamol::core::param::ColorParam(arg);
            value_string = std::string(param.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, float>) {
            auto param = megamol::core::param::FloatParam(arg);
            value_string = std::string(param.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, int>) {
            switch (this->type) {
            case (Parameter::ParamType::INT): {
                auto param = megamol::core::param::IntParam(arg);
                value_string = std::string(param.ValueString().PeekBuffer());
            }
            case (Parameter::ParamType::ENUM): {
                auto param = megamol::core::param::EnumParam(arg);
                value_string = std::string(param.ValueString().PeekBuffer());
            }
            default:
                break;
            }
        } else if constexpr (std::is_same_v<T, std::string>) {
            switch (this->type) {
            case (Parameter::ParamType::STRING): {
                auto param = megamol::core::param::StringParam(arg.c_str());
                value_string = std::string(param.ValueString().PeekBuffer());
                break;
            }
            case (Parameter::ParamType::TRANSFERFUNCTION): {
                auto param = megamol::core::param::TransferFunctionParam(arg);
                value_string = std::string(param.ValueString().PeekBuffer());
                break;
            }
            case (Parameter::ParamType::FILEPATH): {
                auto param = megamol::core::param::FilePathParam(arg.c_str());
                value_string = std::string(param.ValueString().PeekBuffer());
                break;
            }
            case (Parameter::ParamType::FLEXENUM): {
                auto param = megamol::core::param::FlexEnumParam(arg.c_str());
                value_string = std::string(param.ValueString().PeekBuffer());
                break;
            }
            default:
                break;
            }
        } else if constexpr (std::is_same_v<T, vislib::math::Ternary>) {
            auto param = megamol::core::param::TernaryParam(arg);
            value_string = std::string(param.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, glm::vec2>) {
            auto param = megamol::core::param::Vector2fParam(vislib::math::Vector<float, 2>(arg.x, arg.y));
            value_string = std::string(param.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, glm::vec3>) {
            auto param = megamol::core::param::Vector3fParam(vislib::math::Vector<float, 3>(arg.x, arg.y, arg.z));
            value_string = std::string(param.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, glm::vec4>) {
            auto param =
                megamol::core::param::Vector4fParam(vislib::math::Vector<float, 4>(arg.x, arg.y, arg.z, arg.w));
            value_string = std::string(param.ValueString().PeekBuffer());
        } else if constexpr (std::is_same_v<T, std::monostate>) {
            switch (this->type) {
            case (Parameter::ParamType::BUTTON): {
                auto param = megamol::core::param::ButtonParam();
                value_string = std::string(param.ValueString().PeekBuffer());
                break;
            }
            default:
                break;
            }
        }
    };
    std::visit(visitor, this->value);
    return value_string;
}


// CALL SLOT ##################################################################

megamol::gui::graph::CallSlot::CallSlot(int uid) : uid(uid) {
    this->parent_module.reset();
    connected_calls.clear();
}


megamol::gui::graph::CallSlot::~CallSlot() {

    // Call separately and check for reference count
    this->DisConnectCalls();
    this->DisConnectParentModule();
}


bool megamol::gui::graph::CallSlot::CallsConnected(void) const {

    /// TEMP Check for unclean references
    for (auto& call_ptr : this->connected_calls) {
        if (call_ptr == nullptr) {
            throw std::invalid_argument("Pointer to connected call is nullptr.");
        }
    }
    return (!this->connected_calls.empty());
}


bool megamol::gui::graph::CallSlot::ConnectCall(megamol::gui::graph::CallPtrType call) {

    if (call == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to given call is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (this->type == CallSlot::CallSlotType::CALLER) {
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


bool megamol::gui::graph::CallSlot::DisConnectCall(int call_uid, bool called_by_call) {

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


bool megamol::gui::graph::CallSlot::DisConnectCalls(void) {

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


const std::vector<megamol::gui::graph::CallPtrType> megamol::gui::graph::CallSlot::GetConnectedCalls(void) {

    /// TEMP Check for unclean references
    for (auto& call_ptr : this->connected_calls) {
        if (call_ptr == nullptr) {
            throw std::invalid_argument("Pointer to connected call is nullptr.");
        }
    }

    return this->connected_calls;
}


bool megamol::gui::graph::CallSlot::ParentModuleConnected(void) const { return (this->parent_module != nullptr); }


bool megamol::gui::graph::CallSlot::ConnectParentModule(megamol::gui::graph::ModulePtrType parent_module) {

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


bool megamol::gui::graph::CallSlot::DisConnectParentModule(void) {

    if (parent_module == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Pointer to parent module is already nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->parent_module.reset();
    return true;
}


const megamol::gui::graph::ModulePtrType megamol::gui::graph::CallSlot::GetParentModule(void) {

    if (this->parent_module == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Returned pointer to parent module is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    return this->parent_module;
}


// CALL #######################################################################

megamol::gui::graph::Call::Call(int uid) : uid(uid) {

    this->connected_call_slots.clear();
    this->connected_call_slots.emplace(CallSlot::CallSlotType::CALLER, nullptr);
    this->connected_call_slots.emplace(CallSlot::CallSlotType::CALLEE, nullptr);
}


megamol::gui::graph::Call::~Call() { this->DisConnectCallSlots(); }


bool megamol::gui::graph::Call::IsConnected(void) {

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


bool megamol::gui::graph::Call::ConnectCallSlots(
    megamol::gui::graph::CallSlotPtrType call_slot_1, megamol::gui::graph::CallSlotPtrType call_slot_2) {

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


bool megamol::gui::graph::Call::DisConnectCallSlots(void) {

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


const megamol::gui::graph::CallSlotPtrType megamol::gui::graph::Call::GetCallSlot(megamol::gui::graph::CallSlot::CallSlotType type) {

    if (this->connected_call_slots[type] == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Returned pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    return this->connected_call_slots[type];
}


// MODULE #####################################################################

megamol::gui::graph::Module::Module(int uid) : uid(uid) {

    this->call_slots.clear();
    this->call_slots.emplace(megamol::gui::graph::CallSlot::CallSlotType::CALLER, std::vector<CallSlotPtrType>());
    this->call_slots.emplace(megamol::gui::graph::CallSlot::CallSlotType::CALLEE, std::vector<CallSlotPtrType>());
}


megamol::gui::graph::Module::~Module() { this->RemoveAllCallSlots(); }


bool megamol::gui::graph::Module::AddCallSlot(megamol::gui::graph::CallSlotPtrType call_slot) {

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


bool megamol::gui::graph::Module::RemoveAllCallSlots(void) {

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


const std::vector<megamol::gui::graph::CallSlotPtrType> megamol::gui::graph::Module::GetCallSlots(megamol::gui::graph::CallSlot::CallSlotType type) {

    // if (this->call_slots[type].empty()) {
    //    vislib::sys::Log::DefaultLog.WriteWarn(
    //        "Returned call slot list is empty. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    //}
    return this->call_slots[type];
}


const std::map<megamol::gui::graph::CallSlot::CallSlotType, std::vector<megamol::gui::graph::CallSlotPtrType>> megamol::gui::graph::Module::GetCallSlots(
    void) {

    return this->call_slots;
}
