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

/// nothing so far ...


// CALL SLOT ##################################################################

ImVec2 megamol::gui::Graph::CallSlot::GetGuiPos(void) {
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
        retpos = ImVec2(pos.x + ((this->type == Graph::CallSlotType::CALLER) ? (size.x) : (0.0f)), pos.y + size.y * ((float)slot_idx + 1) / ((float)slot_count + 1));
    }
    return retpos;
}


bool megamol::gui::Graph::CallSlot::CallsConnected(void) const {
    return (!this->connected_calls.empty());
}


bool megamol::gui::Graph::CallSlot::ConnectCall(Graph::CallPtr call) {
    if (call == nullptr) return false;
    this->connected_calls.emplace_back(call);
    return true;
}


bool megamol::gui::Graph::CallSlot::DisConnectCall(Graph::CallPtr call) {
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
    for (auto& c : this->connected_calls) {
        if (c == nullptr) {
            throw std::invalid_argument("Pointer to one of the connected calls is nullptr.");
        }
    }
    return this->connected_calls;
}


bool megamol::gui::Graph::CallSlot::ParentModuleConnected(void) const {
    return (this->parent_module != nullptr);
}


bool megamol::gui::Graph::CallSlot::AddParentModule(Graph::ModulePtr parent_module) {
    if (parent_module == nullptr) return false;
    this->parent_module = parent_module;
    return true;
}


bool megamol::gui::Graph::CallSlot::RemoveParentModule(void) {
    if (parent_module == nullptr) return false;
    this->parent_module.reset();
    return true;

}


const Graph::ModulePtr megamol::gui::Graph::CallSlot::GetParentModule(void) {
    if (this->parent_module == nullptr) {
        throw std::invalid_argument("Pointer to parent module is nullptr.");
        return nullptr;
    }
    return this->parent_module;
}


// CALL #######################################################################

bool megamol::gui::Graph::Call::IsConnected(void) {
    unsigned int connected = 0;
    for (auto& c : this->connected_call_slots) {
        if (c.second == nullptr) {
            connected++;
        }
    }
    if (connected == 1) {
        vislib::sys::Log::DefaultLog.WriteWarn("One call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return (connected == 0);
}

bool megamol::gui::Graph::Call::ConnectCallSlot(Graph::CallSlotPtr call_slot) {
    if (call_slot == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn("Pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (this->connected_call_slots[call_slot->type] != nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn("Call slot is already connected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->connected_call_slots[call_slot->type] = call_slot;
    return true;
}


bool megamol::gui::Graph::Call::DisConnectCallSlot(Graph::CallSlotType type) {
    if (this->connected_call_slots[type] == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn("Call slot is already disconnected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    this->connected_call_slots[type]->GetParentModule()->RemoveCallSlot(type, this->connected_call_slots[type]);
    this->connected_call_slots[type].reset();
    return true;
}


bool megamol::gui::Graph::Call::DisConnectCallSlots(void) {
    return (this->DisConnectCallSlot(Graph::CallSlotType::CALLER) && this->DisConnectCallSlot(Graph::CallSlotType::CALLEE));
}


const Graph::CallSlotPtr megamol::gui::Graph::Call::GetCallSlot(Graph::CallSlotType type) {
    if (this->connected_call_slots[type] == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn("Returned pointer to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    return this->connected_call_slots[type];
}


// MODULE #####################################################################

bool megamol::gui::Graph::Module::AddCallSlot(Graph::CallSlotType type, Graph::CallSlotPtr call_slot) {
    if (call_slot == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn("Poniter to call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    for (auto& cs : this->call_slots[type]) {
        if (cs == call_slot) {
            vislib::sys::Log::DefaultLog.WriteWarn("Pointer to call slot already registered in module list. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        }
    }
    this->call_slots[type].emplace_back(call_slot);
    return true;
}


bool megamol::gui::Graph::Module::RemoveCallSlot(Graph::CallSlotType type, Graph::CallSlotPtr call_slot) {
    for (auto iter = this->call_slots[type].begin(); iter != this->call_slots[type].end(); iter++) {
        if ((*iter) == call_slot) {
            (*iter)->DisConnectCalls();
            (*iter)->RemoveParentModule();
            (*iter).reset();
            this->call_slots[type].erase(iter);
            return true;
        }
    }
    vislib::sys::Log::DefaultLog.WriteWarn("Call slot not found for removal. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    return false;
}


bool megamol::gui::Graph::Module::RemoveAllCallSlot(Graph::CallSlotType type) {
    for (auto& cs : this->call_slots[type]) {
        cs->DisConnectCalls();
        cs->RemoveParentModule();
        cs.reset();
    }
    this->call_slots[type].clear();
    return true;
}


bool megamol::gui::Graph::Module::RemoveAllCallSlot(void) {
    return (this->RemoveAllCallSlot(Graph::CallSlotType::CALLEE) && RemoveAllCallSlot(Graph::CallSlotType::CALLER));
}


const std::vector<Graph::CallSlotPtr> megamol::gui::Graph::Module::GetCallSlots(Graph::CallSlotType type) {
    if (this->call_slots[type].empty()) {
        vislib::sys::Log::DefaultLog.WriteWarn("Returned call slot list is empty. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
    return this->call_slots[type];
}


const std::map<Graph::CallSlotType, std::vector<Graph::CallSlotPtr>> megamol::gui::Graph::Module::GetCallSlots(void) {
    return this->call_slots;
}


// GRAPH ######################################################################

megamol::gui::Graph::Graph(void)
    : modules_graph()
    , calls_graph()
    , modules_stock()
    , calls_stock() {

}


megamol::gui::Graph::~Graph(void) {

}


bool megamol::gui::Graph::AddModule(const std::string& module_class_name) {

    try {
        bool found = false;
        for (auto& mod : this->modules_stock) {
            if (module_class_name == mod.class_name) {
                auto mod_ptr = std::make_shared<Graph::Module>();
                mod_ptr->class_name = mod.class_name;
                mod_ptr->description = mod.description;
                mod_ptr->plugin_name = mod.plugin_name;
                mod_ptr->is_view = mod.is_view;
                mod_ptr->name = "module_name"; /// TODO Set ...
                mod_ptr->full_name = "full_name"; /// TODO Set ...
                mod_ptr->instance = "instance"; /// TODO Set ...
                mod_ptr->gui.initialized = false;
                //mod_ptr->gui.id = -1;
                //mod_ptr->gui.position = ImVec2(-1.0f, -1.0f);
                //mod_ptr->gui.size = ImVec2(-1.0f, -1.0f);
                for (auto& p : mod.param_slots) {
                    Graph::ParamSlot param_slot;
                    param_slot.class_name = p.class_name;
                    param_slot.description = p.description;
                    param_slot.type = p.type;
                    param_slot.full_name = "full_name"; /// TODO Set ...
                    mod_ptr->param_slots.emplace_back(param_slot);
                }
                for (auto& call_slots_type : mod.call_slots) {
                    for (auto& c : call_slots_type.second) {
                        Graph::CallSlot call_slot;
                        call_slot.name = c.name;
                        call_slot.description = c.description;
                        call_slot.compatible_call_idxs = c.compatible_call_idxs;
                        call_slot.type = c.type;
                        mod_ptr->AddCallSlot(call_slot.type, std::make_shared<Graph::CallSlot>(call_slot));
                    }
                }
                for (auto& call_slot_type_list : mod_ptr->GetCallSlots()) {
                    for (auto& call_slot : call_slot_type_list.second) {
                        call_slot->AddParentModule(mod_ptr);
                    }
                }
                this->modules_graph.emplace_back(mod_ptr);
                found = true;
                break;
            }
        }
        if (!found) {
            vislib::sys::Log::DefaultLog.WriteError("Unable to find module: %s [%s, %s, line %d]\n", module_class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
    }
    catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


bool megamol::gui::Graph::DeleteModule(int gui_id) {

    try {
        for (auto i = this->modules_graph.begin(); i != this->modules_graph.end(); i++) {
            if ((*i)->gui.id == gui_id) {
                (*i)->RemoveAllCallSlot();
                //vislib::sys::Log::DefaultLog.WriteWarn("Found %i references pointing to module. [%s, %s, line %d]\n", i->use_count(), __FILE__, __FUNCTION__, __LINE__);
                assert(i->use_count() == 1);
                this->modules_graph.erase(i);
                return true;
            }
        }
    }
    catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return false;
}


bool megamol::gui::Graph::AddCall(size_t call_idx, CallSlotPtr call_slot_1, CallSlotPtr call_slot_2) {

    try {
        if (call_idx > this->calls_stock.size()) {
            vislib::sys::Log::DefaultLog.WriteError("Invalid call index. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        auto call = this->calls_stock[call_idx];
        auto call_ptr = std::make_shared<Graph::Call>();
        call_ptr->class_name = call.class_name;
        call_ptr->description = call.description;
        call_ptr->plugin_name = call.plugin_name;
        call_ptr->functions = call.functions;
        call_ptr->gui.id = -1;

        if (call_slot_1->type == call_slot_2->type) {
            vislib::sys::Log::DefaultLog.WriteError("Call slots must have different type. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        if (call_slot_1->GetParentModule() == call_slot_2->GetParentModule()) {
            vislib::sys::Log::DefaultLog.WriteError("Call slots must have different parent module. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        /// TODO Check is call is assigned second time

        call_ptr->ConnectCallSlot(call_slot_1);
        call_ptr->ConnectCallSlot(call_slot_2);

        this->calls_graph.emplace_back(call_ptr);
        vislib::sys::Log::DefaultLog.WriteWarn("CONNECTED: %s [%s, %s, line %d]\n",
            call_ptr->class_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
    }
    catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


bool megamol::gui::Graph::DeleteCall(int gui_id) {

    try {
        for (auto i = this->calls_graph.begin(); i != this->calls_graph.end(); i++) {
            if ((*i)->gui.id == gui_id) {
                (*i)->DisConnectCallSlots();
                //vislib::sys::Log::DefaultLog.WriteWarn("Found %i references pointing to module. [%s, %s, line %d]\n", i->use_count(), __FILE__, __FUNCTION__, __LINE__);
                assert(i->use_count() == 1);
                this->calls_graph.erase(i);
                return true;
            }
        }
    }
    catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


bool megamol::gui::Graph::UpdateAvailableModulesCallsOnce(const megamol::core::CoreInstance* core_instance) {

    bool retval = true;
    if (core_instance == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("Pointer to Core Instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    try {
        std::string plugin_name;
        auto start_time = std::chrono::system_clock::now();

        // CALLS ------------------------------------------------------------------
        /// ! Get calls before getting modules for having calls in place for setting compatible call indices of slots!
        if (this->calls_stock.empty()) {

            //Get plugin calls (get prior to core calls for being  able to find duplicates in core instance call desc. manager)
            const std::vector<core::utility::plugins::AbstractPluginInstance::ptr_type>& plugins = core_instance->Plugins().GetPlugins();
            for (core::utility::plugins::AbstractPluginInstance::ptr_type plugin : plugins) {
                plugin_name = plugin->GetAssemblyName();
                for (auto& c_desc : plugin->GetCallDescriptionManager()) {
                    Graph::StockCall call;
                    call.plugin_name = plugin_name;
                    this->read_call_data(call, c_desc);
                    this->calls_stock.emplace_back(call);
                }
            }
            
            // Get core calls
            plugin_name = "Core"; // (core_instance->GetAssemblyName() = "")
            for (auto& c_desc : core_instance->GetCallDescriptionManager()) {
                std::string class_name = std::string(c_desc->ClassName());
                if (std::find_if(this->calls_stock.begin(), this->calls_stock.end(), [class_name](const Graph::StockCall & call) { return (call.class_name == class_name); }) == this->calls_stock.end()) {
                    Graph::StockCall call;
                    call.plugin_name = plugin_name;
                    this->read_call_data(call, c_desc);
                    this->calls_stock.emplace_back(call);
                }

            }
        }

        // MODULES ----------------------------------------------------------------
        if (this->modules_stock.empty()) {

            // Get plugin modules (get prior to core modules for being  able to find duplicates in core instance module desc. manager)
            const std::vector<core::utility::plugins::AbstractPluginInstance::ptr_type>& plugins = core_instance->Plugins().GetPlugins();
            for (core::utility::plugins::AbstractPluginInstance::ptr_type plugin : plugins) {
                plugin_name = plugin->GetAssemblyName();
                for (auto& m_desc : plugin->GetModuleDescriptionManager()) {
                    Graph::StockModule mod;
                    mod.plugin_name = plugin_name;
                    this->read_module_data(mod, m_desc);
                    this->modules_stock.emplace_back(mod);
                }
            }

            // Get core modules
            plugin_name = "Core";  // (core_instance->GetAssemblyName() = "")
            for (auto& m_desc : core_instance->GetModuleDescriptionManager()) {
                std::string class_name = std::string(m_desc->ClassName());
                if (std::find_if(this->modules_stock.begin(), this->modules_stock.end(), [class_name](const Graph::StockModule & mod) { return (mod.class_name == class_name); }) == this->modules_stock.end()){
                    Graph::StockModule mod;
                    mod.plugin_name = plugin_name;
                    this->read_module_data(mod, m_desc);
                    this->modules_stock.emplace_back(mod);
                }
            }

            // Sorting module by alphabetically ascending class names.
            std::sort(this->modules_stock.begin(), this->modules_stock.end(), [](Graph::StockModule mod1, Graph::StockModule mod2) {
                std::vector<std::string> v;
                v.clear();
                v.emplace_back(mod1.class_name);
                v.emplace_back(mod2.class_name);
                std::sort(v.begin(), v.end());
                return (v.front() != mod2.class_name);
            });
        }

        auto delta_time = static_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - start_time).count();
        vislib::sys::Log::DefaultLog.WriteInfo("Reading available modules and calls ... DONE (duration: %.3f seconds)\n", delta_time); // [%s, %s, line %d]\n", delta_time, __FILE__, __FUNCTION__, __LINE__);
    }
    catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return retval;
}


bool megamol::gui::Graph::read_module_data(Graph::StockModule& mod, const std::shared_ptr<const megamol::core::factories::ModuleDescription> mod_desc) {

    /// mod.plugin_name is not available in mod_desc (set from AbstractAssemblyInstance or AbstractPluginInstance). 
    mod.class_name = std::string(mod_desc->ClassName());
    mod.description = std::string(mod_desc->Description());
    mod.is_view = false;
    mod.param_slots.clear();
    mod.call_slots.clear();
    mod.call_slots.emplace(Graph::CallSlotType::CALLER, std::vector<Graph::StockCallSlot>());
    mod.call_slots.emplace(Graph::CallSlotType::CALLEE, std::vector<Graph::StockCallSlot>());

    if (this->calls_stock.empty()) {
        vislib::sys::Log::DefaultLog.WriteError("Call list is empty. Call read_call_data() prior to that. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    try {
        // SLOTS ------------------------------------------------------------------
        /// (Following code is adapted from megamol::core::job::job::PluginsStateFileGeneratorJob.cpp)

        megamol::core::Module::ptr_type new_mod(mod_desc->CreateModule(nullptr));
        if (new_mod == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError("Unable to create module: %s. [%s, %s, line %d]\n", mod_desc->ClassName(), __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        megamol::core::RootModuleNamespace::ptr_type root_mod_ns = std::make_shared<megamol::core::RootModuleNamespace>();
        root_mod_ns->AddChild(new_mod);

        std::shared_ptr<const core::view::AbstractView> viewptr = std::dynamic_pointer_cast<const core::view::AbstractView>(new_mod);
        mod.is_view = (viewptr != nullptr);

        std::vector<std::shared_ptr<core::param::ParamSlot> > paramSlots;
        std::vector<std::shared_ptr<core::CallerSlot> > callerSlots;
        std::vector<std::shared_ptr<core::CalleeSlot> > calleeSlots;

        core::Module::child_list_type::iterator ano_end = new_mod->ChildList_End();
        for (core::Module::child_list_type::iterator ano_i = new_mod->ChildList_Begin(); ano_i != ano_end; ++ano_i) {
            std::shared_ptr<core::param::ParamSlot> p_ptr = std::dynamic_pointer_cast<core::param::ParamSlot>(*ano_i);
            if (p_ptr != nullptr) paramSlots.push_back(p_ptr);
            std::shared_ptr<core::CallerSlot> cr_ptr = std::dynamic_pointer_cast<core::CallerSlot>(*ano_i);
            if (cr_ptr != nullptr) callerSlots.push_back(cr_ptr);
            std::shared_ptr<core::CalleeSlot> ce_ptr = std::dynamic_pointer_cast<core::CalleeSlot>(*ano_i);
            if (ce_ptr != nullptr) calleeSlots.push_back(ce_ptr);
        }

        // Param Slots
        for (std::shared_ptr<core::param::ParamSlot> param_slot : paramSlots) {
            Graph::StockParamSlot psd;
            psd.class_name = std::string(param_slot->Name().PeekBuffer());
            psd.description = std::string(param_slot->Description().PeekBuffer());

            if (auto* p_ptr = param_slot->Param<core::param::ButtonParam>()) { psd.type = Graph::ParamType::BUTTON; }
            else if (auto* p_ptr = param_slot->Param<core::param::BoolParam>()) { psd.type = Graph::ParamType::BOOL; }
            else if (auto* p_ptr = param_slot->Param<core::param::ColorParam>()) { psd.type = Graph::ParamType::COLOR; }
            else if (auto* p_ptr = param_slot->Param<core::param::EnumParam>()) { psd.type = Graph::ParamType::ENUM; }
            else if (auto* p_ptr = param_slot->Param<core::param::FilePathParam>()) { psd.type = Graph::ParamType::FILEPATH; }
            else if (auto* p_ptr = param_slot->Param<core::param::FlexEnumParam>()) { psd.type = Graph::ParamType::FLEXENUM; }
            else if (auto* p_ptr = param_slot->Param<core::param::FloatParam>()) { psd.type = Graph::ParamType::FLOAT; }
            else if (auto* p_ptr = param_slot->Param<core::param::IntParam>()) { psd.type = Graph::ParamType::INT; }
            else if (auto* p_ptr = param_slot->Param<core::param::StringParam>()) { psd.type = Graph::ParamType::STRING; }
            else if (auto* p_ptr = param_slot->Param<core::param::TernaryParam>()) { psd.type = Graph::ParamType::TERNARY; }
            else if (auto* p_ptr = param_slot->Param<core::param::TransferFunctionParam>()) { psd.type = Graph::ParamType::TRANSFERFUNCTION; }
            else if (auto* p_ptr = param_slot->Param<core::param::Vector2fParam>()) { psd.type = Graph::ParamType::VECTOR2F; }
            else if (auto* p_ptr = param_slot->Param<core::param::Vector3fParam>()) { psd.type = Graph::ParamType::VECTOR3F; }
            else if (auto* p_ptr = param_slot->Param<core::param::Vector4fParam>()) { psd.type = Graph::ParamType::VECTOR4F; }
            else { psd.type = Graph::ParamType::UNKNOWN; }

            mod.param_slots.emplace_back(psd);
        }

        // CallerSlots
        for (std::shared_ptr<core::CallerSlot> caller_slot : callerSlots) {
            Graph::StockCallSlot csd;
            csd.name = std::string(caller_slot->Name().PeekBuffer());
            csd.description = std::string(caller_slot->Description().PeekBuffer());
            csd.compatible_call_idxs.clear();
            csd.type = Graph::CallSlotType::CALLER;

            SIZE_T callCount = caller_slot->GetCompCallCount();
            for (SIZE_T i = 0; i < callCount; ++i) {
                std::string comp_call_class_name = std::string(caller_slot->GetCompCallClassName(i));
                size_t calls_cnt = this->calls_stock.size();
                for (size_t idx = 0; idx < calls_cnt; ++idx) {
                    if (this->calls_stock[idx].class_name == comp_call_class_name) {
                        csd.compatible_call_idxs.emplace_back(idx);
                    }
                }
            }

            mod.call_slots[csd.type].emplace_back(csd);
        }

        // CalleeSlots
        for (std::shared_ptr<core::CalleeSlot> callee_slot : calleeSlots) {
            Graph::StockCallSlot csd;
            csd.name = std::string(callee_slot->Name().PeekBuffer());
            csd.description = std::string(callee_slot->Description().PeekBuffer());
            csd.compatible_call_idxs.clear();
            csd.type = Graph::CallSlotType::CALLEE;

            SIZE_T callbackCount = callee_slot->GetCallbackCount();
            std::vector<std::string> callNames, funcNames;
            std::set<std::string> uniqueCallNames, completeCallNames;
            for (SIZE_T i = 0; i < callbackCount; ++i) {
                uniqueCallNames.insert(callee_slot->GetCallbackCallName(i));
                callNames.push_back(callee_slot->GetCallbackCallName(i));
                funcNames.push_back(callee_slot->GetCallbackFuncName(i));
            }
            size_t ll = callNames.size();
            assert(ll == funcNames.size());
            for (std::string callName : uniqueCallNames) {
                bool found_call = false;
                Graph::StockCall call;
                for (auto& c : this->calls_stock) {
                    if (callName == c.class_name) {
                        call = c;
                        found_call = true;
                        break;
                    }
                }
                bool allFound = true;
                if (found_call) {
                    for (auto& func_name : call.functions) {
                        bool found = false;
                        for (size_t j = 0; j < ll; ++j) {
                            if ((callNames[j] == callName) && (funcNames[j] == func_name)) {
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            allFound = false;
                            break;
                        }
                    }
                }
                else {
                    allFound = false;
                }
                if (allFound) {
                    completeCallNames.insert(callName);
                }
            }
            for (std::string callName : completeCallNames) {
                size_t calls_cnt = this->calls_stock.size();
                for (size_t idx = 0; idx < calls_cnt; ++idx) {
                    if (this->calls_stock[idx].class_name == callName) {
                        csd.compatible_call_idxs.emplace_back(idx);
                    }
                }
            }

            mod.call_slots[csd.type].emplace_back(csd);
        }

        paramSlots.clear();
        callerSlots.clear();
        calleeSlots.clear();
        root_mod_ns->RemoveChild(new_mod);
        new_mod->SetAllCleanupMarks();
        new_mod->PerformCleanup();
        new_mod.reset();

    }
    catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError("Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


bool megamol::gui::Graph::read_call_data(Graph::StockCall& call, const std::shared_ptr<const megamol::core::factories::CallDescription> call_desc) {

    try {
        /// call.plugin_name is not available in call_desc (set from AbstractAssemblyInstance or AbstractPluginInstance). 
        call.class_name = std::string(call_desc->ClassName());
        call.description = std::string(call_desc->Description());
        call.functions.clear();
        for (unsigned int i = 0; i < call_desc->FunctionCount(); ++i) {
            call.functions.emplace_back(call_desc->FunctionName(i));
        }
    }
    catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError("Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


bool megamol::gui::Graph::PROTOTYPE_SaveGraph(std::string project_filename, megamol::core::CoreInstance* core_instance) {

    if (core_instance == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("Pointer to CoreInstance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    std::string confstr;
    std::stringstream confInstances, confModules, confCalls, confParams;
    bool already_found_main_view = false;
    // ------------------------------------------------------------------------
    try {
        // Search for top most view
        for (auto& mod : this->modules_graph) {
            // Check for not connected calle_slots
            bool is_main_view = false;
            if (mod->is_view) {
                bool callee_not_connected = false;
                for (auto& call_slots : mod->GetCallSlots(Graph::CallSlotType::CALLEE)) {
                    if (!call_slots->CallsConnected()) {
                        callee_not_connected = true;
                    }
                }
                if (!callee_not_connected) {
                    if (!already_found_main_view) {
                        confInstances << "mmCreateView(\"" << mod->instance << "\",\"" << mod->class_name << "\",\"" << mod->full_name << "\")\n";
                        already_found_main_view = true;
                        is_main_view = true;
                    }
                    else {
                        vislib::sys::Log::DefaultLog.WriteError("Only one main view is allowed. Found another one. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                    }
                }
            }
            if (!is_main_view) {
                confModules << "mmCreateModule(\"" << mod->class_name << "\",\"" << mod->full_name << "\")\n";
            }

            /*
            for (auto& p : m.param_slots) {
                if (p.type != Graph::ParamType::BUTTON) {
                    std::string val = slot->Parameter()->ValueString().PeekBuffer();
                    // Encode to UTF-8 string
                    vislib::StringA valueString;
                    vislib::UTF8Encoder::Encode(valueString, vislib::StringA(val.c_str()));
                    val = valueString.PeekBuffer();
                    confParams << "mmSetParamValue(\"" << p.full_name << "\",[=[" << val << "]=])\n";
                }
            }
            */

            for (auto& cr : mod->GetCallSlots(Graph::CallSlotType::CALLER)) {
                for (auto& call : cr->GetConnectedCalls()) {
                    if (call->IsConnected()) {
                        confCalls << "mmCreateCall(\"" << call->class_name << "\",\""
                            << call->GetCallSlot(Graph::CallSlotType::CALLER)->GetParentModule()->full_name << "::" << call->GetCallSlot(Graph::CallSlotType::CALLER)->name << "\",\""
                            << call->GetCallSlot(Graph::CallSlotType::CALLEE)->GetParentModule()->full_name << "::" << call->GetCallSlot(Graph::CallSlotType::CALLEE)->name << "\")\n";
                    }
                }
            }
        }
       
        confstr = confInstances.str() + "\n" + confModules.str() + "\n" + confCalls.str() + "\n" + confParams.str() + "\n";
    }
    catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError("Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // ------------------------------------------------------------------------

    try {
        std::ofstream file;
        file.open(project_filename);
        if (file.good()) {
            file << confstr.c_str();
            file.close();
        }
        else {
            vislib::sys::Log::DefaultLog.WriteError("Unable to create project file. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            file.close();
            return false;
        }
    }
    catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError("Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}
