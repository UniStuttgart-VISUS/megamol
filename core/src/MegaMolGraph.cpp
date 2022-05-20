#include "mmcore/MegaMolGraph.h"
#include "mmcore/AbstractSlot.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/view/AbstractView_EventConsumption.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <numeric> // std::accumulate
#include <string>


// splits a string of the form "::one::two::three::" into an array of strings {"one", "two", "three"}
static std::vector<std::string> splitPathName(std::string const& path) {
    std::vector<std::string> result;

    size_t start = 0;
    while ((start = path.find_first_not_of(':', start)) != std::string::npos) {
        auto end = path.find_first_of(':', start);
        if (start < end)
            result.push_back(path.substr(start, end - start));
        start = end;
    }

    return result;
}

// modules search and compare slot names case insensitive (legacy behaviour)
// std::string operator== is case sensitive. so when looking for slots, we lower them first.
static std::string tolower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

// AbstractNamedObject::FullName() prepends extra :: to module names which leads to
// lookup problems when module names already start with ::
// so we normalize the amount of :: on module names coming into the public graph api functions
static std::string clean(std::string const& path) {

    auto begin = path.find_first_not_of(':');
    return "::" + path.substr(begin);
}

static std::string cut_off_prefix(std::string const& name, std::string const& prefix) {
    return name.substr(prefix.size());
}

static void log(std::string text) {
    const std::string msg = "MegaMolGraph: " + text;
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(msg.c_str());
}

static void log_error(std::string text) {
    const std::string msg = "MegaMolGraph: " + text;
    megamol::core::utility::log::Log::DefaultLog.WriteError(msg.c_str());
}

static megamol::core::param::AbstractParam* getParameterFromParamSlot(megamol::core::param::ParamSlot* param_slot) {
    if (!param_slot)
        return nullptr;

    if (param_slot->GetStatus() == megamol::core::AbstractSlot::STATUS_UNAVAILABLE) {
        log_error("error. cannot find parameter: " + std::string(param_slot->Name().PeekBuffer()) +
                  ", slot is not available");
        return nullptr;
    }
    if (param_slot->Parameter().IsNull()) {
        log_error("error. cannot find parameter: " + std::string(param_slot->Name().PeekBuffer()) +
                  ", slot has no parameter");
        return nullptr;
    }

    return param_slot->Parameter().DynamicCast<megamol::core::param::AbstractParam>();
}

megamol::core::MegaMolGraph::MegaMolGraph(megamol::core::CoreInstance& core,
    factories::ModuleDescriptionManager const& moduleProvider, factories::CallDescriptionManager const& callProvider)
        : moduleProvider_ptr{&moduleProvider}
        , callProvider_ptr{&callProvider}
        , dummy_namespace{std::make_shared<RootModuleNamespace>()}
        , convenience_functions{const_cast<MegaMolGraph*>(this)} {
    // the Core Instance is a parasite that needs to be passed to all modules
    // TODO: make it so there is no more core instance
    dummy_namespace->SetCoreInstance(core);
}

megamol::core::MegaMolGraph::~MegaMolGraph() {
    moduleProvider_ptr = nullptr;
    callProvider_ptr = nullptr;
}

const megamol::core::factories::ModuleDescriptionManager& megamol::core::MegaMolGraph::ModuleProvider() {
    return *moduleProvider_ptr; // not null because we took it by reference in constructor
}

const megamol::core::factories::CallDescriptionManager& megamol::core::MegaMolGraph::CallProvider() {
    return *callProvider_ptr;
}

/*
 * ------------- public Graph API begin -------------
 *
 * normalize :: of incoming module/call/param names
 */

bool megamol::core::MegaMolGraph::DeleteModule(std::string const& id) {
    return delete_module(clean(id));
}

bool megamol::core::MegaMolGraph::CreateModule(std::string const& className, std::string const& id) {
    return add_module(ModuleInstantiationRequest{className, clean(id)});
}

bool megamol::core::MegaMolGraph::RenameModule(std::string const& old, std::string const& neww) {
    auto oldId = clean(old);
    auto newId = clean(neww);

    auto module_it = find_module(oldId);
    if (module_it == module_list_.end()) {
        log_error("error. could not rename module. unable to find module named: " + oldId);
        return false;
    }
    if (!module_it->modulePtr) {
        log_error("error. could not rename module. module is nullptr: " + oldId);
        return false;
    }

    log("rename module " + module_it->request.id + " to " + newId);
    module_it->request.id = newId;
    module_it->modulePtr->setName(newId.c_str());

    for (auto child = module_it->modulePtr->ChildList_Begin(); child != module_it->modulePtr->ChildList_End();
         ++child) {
        auto ps = dynamic_cast<param::ParamSlot*>((*child).get());
        if (ps != nullptr) {
            auto p = ps->Param<param::ButtonParam>();
            if (p != nullptr) {
                auto command_name = oldId + std::string("_") + ps->Name().PeekBuffer();
                auto updated_command_name = newId + std::string("_") + ps->Name().PeekBuffer();
                auto c = m_command_registry->get_command(command_name);
                m_command_registry->remove_command_by_name(command_name);
                c.name = updated_command_name;
                c.parent = ps->FullName();
                m_command_registry->add_command(c);
            }
        }
    }

    const auto matches_old_prefix = [&](std::string const& call_slot) {
        auto res = call_slot.find(oldId);
        return (res != std::string::npos) && res == 0;
    };

    const auto put_new_prefix = [&](auto& name) {
        auto old = name;
        name = newId + cut_off_prefix(name, oldId);
        log("rename call at slot " + old + " to " + name);
    };

    for (auto& call : call_list_) {
        if (matches_old_prefix(call.request.from)) {
            put_new_prefix(call.request.from);
        }
        if (matches_old_prefix(call.request.to)) {
            put_new_prefix(call.request.to);
        }
    }

    // dont know what we are supposed to do when entry point renaming fails... how can it fail?
    if (module_it->isGraphEntryPoint) {
        bool view_rename_ok = m_image_presentation->rename_entry_point(oldId, newId);
        if (!view_rename_ok) {
            log_error("error renaming graph entry point. image presentation service could not rename module: " + oldId +
                      " -> " + newId);
            return false;
        }
    }

    return true;
}

bool megamol::core::MegaMolGraph::DeleteCall(std::string const& from, std::string const& to) {
    return delete_call(CallDeletionRequest{clean(from), clean(to)});
}

bool megamol::core::MegaMolGraph::CreateCall(
    std::string const& className, std::string const& from, std::string const& to) {
    return add_call(CallInstantiationRequest{className, clean(from), clean(to)});
}

megamol::core::Module::ptr_type megamol::core::MegaMolGraph::FindModule(std::string const& module) const {
    auto moduleName = clean(module);
    auto module_it = find_module(moduleName);

    if (module_it == module_list_.end()) {
        log_error("error. could not find module: " + moduleName);
        return nullptr;
    }

    return module_it->modulePtr;
}

megamol::core::Call::ptr_type megamol::core::MegaMolGraph::FindCall(
    std::string const& from_, std::string const& to_) const {
    auto from = clean(from_);
    auto to = clean(to_);
    auto call_it = find_call(from, to);

    if (call_it == call_list_.end()) {
        log_error("error. could not find call: " + from + " -> " + to);
        return nullptr;
    }

    return call_it->callPtr;
}

megamol::core::param::AbstractParam* megamol::core::MegaMolGraph::FindParameter(std::string const& paramName) const {
    return getParameterFromParamSlot(this->FindParameterSlot(clean(paramName)));
}

megamol::core::param::ParamSlot* megamol::core::MegaMolGraph::FindParameterSlot(std::string const& param) const {
    auto paramName = clean(param);
    // match module where module name is prefix of parameter slot name
    auto module_it = find_module_by_prefix(paramName);

    if (module_it == module_list_.end()) {
        log_error("error. could not find parameter, module name not found, parameter name: " + paramName + ")");
        return nullptr;
    }

    std::string module_name = module_it->request.id;
    std::string slot_name = cut_off_prefix(paramName, module_name + "::");

    AbstractSlot* slot_ptr = module_it->modulePtr->FindSlot(slot_name.c_str());
    param::ParamSlot* param_slot_ptr = dynamic_cast<param::ParamSlot*>(slot_ptr);

    if (slot_ptr == nullptr || param_slot_ptr == nullptr) {
        log_error("error. could not find parameter, slot not found or of wrong type. parameter name: " + paramName +
                  ", slot name: " + slot_name);
        return nullptr;
    }

    return param_slot_ptr;
}

std::vector<megamol::core::param::AbstractParam*> megamol::core::MegaMolGraph::EnumerateModuleParameters(
    std::string const& module) const {
    auto moduleName = clean(module);
    auto param_slots = EnumerateModuleParameterSlots(moduleName);

    std::vector<megamol::core::param::AbstractParam*> params;
    params.reserve(param_slots.size());

    for (auto& slot : param_slots)
        params.push_back(getParameterFromParamSlot(slot));

    return params;
}

std::vector<megamol::core::param::ParamSlot*> megamol::core::MegaMolGraph::EnumerateModuleParameterSlots(
    std::string const& module) const {
    auto moduleName = clean(module);
    std::vector<megamol::core::param::ParamSlot*> parameters;

    auto module_it = find_module(moduleName);

    if (module_it == module_list_.end()) {
        log_error("error. could not find module: " + moduleName);
        return parameters;
    }

    auto children_begin = module_it->modulePtr->ChildList_Begin();
    auto children_end = module_it->modulePtr->ChildList_End();

    while (children_begin != children_end) {
        AbstractNamedObject::ptr_type named_object = *children_begin;
        if (named_object != nullptr) {
            AbstractSlot* slot_ptr = dynamic_cast<AbstractSlot*>(named_object.get());
            param::ParamSlot* param_slot_ptr = dynamic_cast<param::ParamSlot*>(slot_ptr);

            if (slot_ptr && param_slot_ptr)
                parameters.push_back(param_slot_ptr);
        }

        children_begin++;
    }

    return parameters;
}

megamol::core::CallList_t const& megamol::core::MegaMolGraph::ListCalls() const {
    return call_list_;
}

megamol::core::ModuleList_t const& megamol::core::MegaMolGraph::ListModules() const {
    return module_list_;
}

std::vector<megamol::core::param::AbstractParam*> megamol::core::MegaMolGraph::ListParameters() const {
    auto param_slots = this->ListParameterSlots();

    std::vector<megamol::core::param::AbstractParam*> parameters;
    parameters.reserve(param_slots.size());

    for (auto& slot : param_slots)
        parameters.push_back(getParameterFromParamSlot(slot));

    return parameters;
}

std::vector<megamol::core::param::ParamSlot*> megamol::core::MegaMolGraph::ListParameterSlots() const {
    std::vector<megamol::core::param::ParamSlot*> param_slots;

    for (auto& mod : module_list_) {
        auto module_params = this->EnumerateModuleParameterSlots(mod.modulePtr->Name().PeekBuffer());

        param_slots.insert(param_slots.end(), module_params.begin(), module_params.end());
    }

    return param_slots;
}

bool megamol::core::MegaMolGraph::SetGraphEntryPoint(std::string module) {
    auto moduleName = clean(module);
    // currently, we expect the entry point to be derived from AbstractView
    auto module_it = find_module(moduleName);

    if (module_it == module_list_.end()) {
        log_error("error adding graph entry point. could not find module: " + moduleName);
        return false;
    }

    auto module_shared_ptr = module_it->modulePtr; // we cant cast shared_ptr to void* for image presentation rendering
    auto& module_ref = *module_shared_ptr;
    auto* module_raw_ptr = &module_ref;

    if (auto view_ptr = dynamic_cast<megamol::core::view::AbstractView*>(module_raw_ptr); view_ptr == nullptr) {
        log_error("error adding graph entry point. module is not an entry point type (AbstractView): " + moduleName);
        return false;
    }

    // the image presentation will issue the rendering and provide the view with resources for rendering
    // probably we dont care or dont check wheter the same view is added as entry point multiple times
    bool view_presentation_ok = m_image_presentation->add_entry_point(
        moduleName, {static_cast<void*>(module_raw_ptr), std::function{megamol::core::view::view_rendering_execution},
                        std::function{megamol::core::view::get_view_runtime_resources_requests}});

    if (!view_presentation_ok) {
        log_error("error adding graph entry point. image presentation service rejected module: " + moduleName);
        return false;
    }

    this->graph_entry_points.push_back(module_shared_ptr);

    module_it->isGraphEntryPoint = true;
    log("set graph entry point: " + moduleName);

    return true;
}

bool megamol::core::MegaMolGraph::RemoveGraphEntryPoint(std::string module) {
    auto moduleName = clean(module);
    auto module_it = find_module(moduleName);

    if (module_it == module_list_.end()) {
        log_error("error removing graph entry point. could not find module: " + moduleName);
        return false;
    }

    bool view_removal_ok = m_image_presentation->remove_entry_point(moduleName);

    // note that for us it is not an error to try to remove a module as graph entry point
    // if that module is not registered as an entry point
    // but maybe image presentation may tell us that for some other reason removal failed?
    if (!view_removal_ok) {
        log_error("error adding graph entry point. image presentation service could not remove module: " + moduleName);
        return false;
    }

    this->graph_entry_points.remove_if(
        [&](Module::ptr_type& module) { return std::string{module->Name().PeekBuffer()} == moduleName; });

    module_it->isGraphEntryPoint = false;
    log("remove graph entry point: " + moduleName);

    return true;
}

bool megamol::core::MegaMolGraph::AddFrontendResources(
    std::vector<megamol::frontend::FrontendResource> const& resources) {
    this->provided_resources_lookup = {resources};

    auto [success, graph_resources] = provided_resources_lookup.get_requested_resources(
        {"ImagePresentationEntryPoints", megamol::frontend_resources::CommandRegistry_Req_Name,
#ifdef PROFILING
            megamol::frontend_resources::PerformanceManager_Req_Name
#endif
        });

    if (!success)
        return false;

    m_image_presentation = &const_cast<megamol::frontend_resources::ImagePresentationEntryPoints&>(
        graph_resources[0].getResource<megamol::frontend_resources::ImagePresentationEntryPoints>());

    m_command_registry = &const_cast<megamol::frontend_resources::CommandRegistry&>(
        graph_resources[1].getResource<megamol::frontend_resources::CommandRegistry>());

#ifdef PROFILING
    m_perf_manager = &const_cast<frontend_resources::PerformanceManager&>(
        graph_resources[2].getResource<megamol::frontend_resources::PerformanceManager>());
#endif

    return true;
}

megamol::core::MegaMolGraph_Convenience& megamol::core::MegaMolGraph::Convenience() {
    return this->convenience_functions;
}

void megamol::core::MegaMolGraph::Clear() {
    // currently entry points are expected to be graph modules, i.e. views
    // therefore it is ok for us to clear all entry points if the graph shuts down
    call_list_.clear();
    m_image_presentation->clear_entry_points();
    graph_entry_points.clear();
    module_list_.clear();
}

/*
 * ------------- end public Graph API begin -------------
 */


megamol::core::ModuleList_t::iterator megamol::core::MegaMolGraph::find_module(std::string const& name) {
    auto it = std::find_if(this->module_list_.begin(), this->module_list_.end(),
        [&name](megamol::core::ModuleInstance_t const& el) { return el.request.id == name; });

    return it;
}

megamol::core::ModuleList_t::const_iterator megamol::core::MegaMolGraph::find_module(std::string const& name) const {

    auto it = std::find_if(this->module_list_.cbegin(), this->module_list_.cend(),
        [&name](megamol::core::ModuleInstance_t const& el) { return el.request.id == name; });

    return it;
}

megamol::core::CallList_t::iterator megamol::core::MegaMolGraph::find_call(
    std::string const& from, std::string const& to) {
    auto it =
        std::find_if(this->call_list_.begin(), this->call_list_.end(), [&](megamol::core::CallInstance_t const& el) {
            // tolower emulates case insensitive comparison in Module::FindSlot() during add_call
            return tolower(el.request.from) == tolower(from) && tolower(el.request.to) == tolower(to);
        });

    return it;
}

megamol::core::CallList_t::const_iterator megamol::core::MegaMolGraph::find_call(
    std::string const& from, std::string const& to) const {

    auto it =
        std::find_if(this->call_list_.cbegin(), this->call_list_.cend(), [&](megamol::core::CallInstance_t const& el) {
            // tolower emulates case insensitive comparison in Module::FindSlot() during add_call
            return tolower(el.request.from) == tolower(from) && tolower(el.request.to) == tolower(to);
        });

    return it;
}


bool megamol::core::MegaMolGraph::add_module(ModuleInstantiationRequest_t const& request) {
    factories::ModuleDescription::ptr module_description = this->ModuleProvider().Find(request.className.c_str());
    if (!module_description) {
        log_error("error. module factory could not find module class name: " + request.className);
        return false;
    }

    Module::ptr_type module_ptr = module_description->CreateModule(request.id);
    if (!module_ptr) {
        log_error("error. could not instantiate module from module description: " + request.className);
        return false;
    }

    auto module_lifetime_resource_request = module_ptr->requested_lifetime_resources();

    auto [success, module_lifetime_dependencies] =
        provided_resources_lookup.get_requested_resources(module_lifetime_resource_request);

    if (!success) {
        std::string requested_deps = "";
        std::string found_deps = "";
        for (auto& req : module_lifetime_resource_request)
            requested_deps += " " + req;
        for (auto& dep : module_lifetime_dependencies)
            found_deps += " " + dep.getIdentifier();
        log_error("error. could not create module " + request.className + "(" + request.id +
                  "), not all requested resources available: ");
        log_error("requested: " + requested_deps);
        log_error("found: " + found_deps);

        return false;
    }

    this->module_list_.push_front(
        {module_ptr, request, false, module_lifetime_resource_request, module_lifetime_dependencies});

    module_ptr->setParent(this->dummy_namespace);

    const auto create_module = [module_description, module_ptr](auto& module_lifetime_dependencies) {
        const bool init_ok =
            module_ptr->Create(module_lifetime_dependencies); // seems like Create() internally checks IsAvailable()

        if (!init_ok)
            log_error("error. could not create module, IsAvailable() or Create() failed: " +
                      std::string((module_ptr->Name()).PeekBuffer()));
        else
            log("create module: " + std::string((module_ptr->Name()).PeekBuffer()));

        return init_ok;
    };

    bool isCreateOk = create_module(this->module_list_.front().lifetime_resources);

    if (!isCreateOk) {
        this->module_list_.pop_front();
    } else {
        // iterate parameters, add hotkeys to CommandRegistry
        for (auto child = module_ptr->ChildList_Begin(); child != module_ptr->ChildList_End(); ++child) {
            auto ps = dynamic_cast<param::ParamSlot*>((*child).get());
            if (ps != nullptr) {
                auto p = ps->Param<param::ButtonParam>();
                if (p != nullptr) {
                    frontend_resources::Command c;
                    c.key = p->GetKeyCode();
                    c.parent = ps->FullName();
                    c.name = module_ptr->Name().PeekBuffer() + std::string("_") + ps->Name().PeekBuffer();
                    c.effect = this->Parameter_Lambda;
                    m_command_registry->add_command(c);
                }
            }
        }
    }
    return isCreateOk;
}

bool megamol::core::MegaMolGraph::add_call(CallInstantiationRequest_t const& request) {

    factories::CallDescription::ptr call_description = this->CallProvider().Find(request.className.c_str());

    const auto getCallSlotOfModule = [this, &call_description](
                                         std::string const& name) -> std::pair<AbstractSlot*, Module::ptr_type> {
        auto module_it = find_module_by_prefix(name);
        if (module_it == this->module_list_.end()) {
            log_error(
                "error. could not find module for requested call: " + name + "(" + call_description->ClassName() + ")");
            return {nullptr, nullptr};
        }
        const auto module_name = module_it->request.id;
        const auto slot_name = cut_off_prefix(name, module_name + "::");

        Module::ptr_type module_ptr = module_it->modulePtr;
        AbstractSlot* slot_ptr = module_ptr->FindSlot(slot_name.c_str());
        if (!slot_ptr) {
            log_error("error. could not find slot named: " + slot_name +
                      " to connect requested call: " + std::string(call_description->ClassName()));
            return {nullptr, nullptr};
        }

        if (!slot_ptr->IsCallCompatible(call_description)) {
            log_error("error. call: " + std::string(call_description->ClassName()) +
                      " is not compatible with slot: " + slot_name);
            return {nullptr, nullptr};
        }

        if (!slot_ptr->GetStatus() == AbstractSlot::STATUS_ENABLED) {
            log_error("error. slot: " + slot_name +
                      " is not enabled. can not connect call: " + std::string(call_description->ClassName()));
            return {nullptr, nullptr};
        }

        return {slot_ptr, module_ptr};
    };

    if (call_description == nullptr) {
        log_error("error. could not find call class name: " + request.className);
        return false;
    }

    auto from_slot = getCallSlotOfModule(request.from);
    if (!from_slot.first) {
        auto m = find_module_by_prefix(request.from);
        std::string slot_names = "none.";
        if (m != this->module_list_.end()) {
            const auto slots = m->modulePtr->GetSlots<CallerSlot>();
            if (!slots.empty()) {
                slot_names = "";
                for (auto x = 0; x < slots.size() - 1; ++x) {
                    slot_names += slots[x]->Name() + ", ";
                }
                slot_names += slots[slots.size() - 1]->Name();
            }
        }
        log_error("error. could not find from-slot: " + request.from +
                  " for call: " + std::string(call_description->ClassName()) + "; possible slots: " + slot_names);
        return false; // error when looking for from-slot
    }
    CallerSlot* caller = dynamic_cast<CallerSlot*>(from_slot.first);

    auto to_slot = getCallSlotOfModule(request.to);
    if (!to_slot.first) {
        auto m = find_module_by_prefix(request.to);
        std::string slot_names = "none.";
        if (m != this->module_list_.end()) {
            const auto slots = m->modulePtr->GetSlots<CalleeSlot>();
            if (!slots.empty()) {
                slot_names = "";
                for (auto x = 0; x < slots.size() - 1; ++x) {
                    slot_names += slots[x]->Name() + ", ";
                }
                slot_names += slots[slots.size() - 1]->Name();
            }
        }
        log_error("error. could not find to-slot: " + request.to +
                  " for call: " + std::string(call_description->ClassName()) + "; possible slots: " + slot_names);
        return false; // error when looking for to-slot
    }
    CalleeSlot* callee = dynamic_cast<CalleeSlot*>(to_slot.first);

    if ((caller->GetStatus() == AbstractSlot::STATUS_CONNECTED) ||
        (callee->GetStatus() == AbstractSlot::STATUS_CONNECTED)) {
        Call* tstCall = caller->IsConnectedTo(callee);
        if (tstCall && call_description->IsDescribing(tstCall)) {
            log_error("error. caller (" + request.from + ") and callee (" + request.to +
                      ") are already connected by call: " + std::string(call_description->ClassName()));
            return false; // call already exists
        }
    }

    // TODO: kill parents of modules/calls when new graph structure is in place
    caller->setParent(from_slot.second);
    callee->setParent(to_slot.second);

    Call::ptr_type call = Call::ptr_type(call_description->CreateCall());
    if (!callee->ConnectCall(call.get(), call_description)) {
        log_error("error. connecting call: " + std::string(call_description->ClassName()) + " failed at callee");
        return false;
    }
    if (!caller->ConnectCall(call.get())) {
        log_error("error. connecting call: " + std::string(call_description->ClassName()) + " failed at caller");
        // FIXME: if connecting the callER fails, how to disconnect call from callEE?
        // callee->DisconnectCalls();
        return false;
    }

    log("create call: " + request.from + " -> " + request.to + " (" + std::string(call_description->ClassName()) + ")");
    this->call_list_.emplace_front(CallInstance_t{call, request});
#ifdef PROFILING
    auto the_call = call.get();
    //printf("adding timers for @ %p = %s \n", reinterpret_cast<void*>(the_call), the_call->GetDescriptiveText().c_str());
    the_call->cpu_queries =
        m_perf_manager->add_timers(the_call, frontend_resources::PerformanceManager::query_api::CPU);
    if (the_call->GetCapabilities().OpenGLRequired()) {
        the_call->gl_queries =
            m_perf_manager->add_timers(the_call, frontend_resources::PerformanceManager::query_api::OPENGL);
    }
    the_call->perf_man = m_perf_manager;
#endif

    return true;
}

static std::list<megamol::core::CallList_t::iterator> find_all_of(
    megamol::core::CallList_t list, std::function<bool(megamol::core::CallInstance_t const&)> const& func) {

    std::list<megamol::core::CallList_t::iterator> result;

    for (auto begin = list.begin(); begin != list.end(); begin++)
        if (func(*begin))
            result.push_back(begin);

    return result;
}


bool megamol::core::MegaMolGraph::delete_module(ModuleDeletionRequest_t const& request) {

    auto module_it = find_module(request);
    if (module_it == this->module_list_.end()) {
        log_error("error. could not find module for deletion: " + request);
        return false;
    }

    auto module_ptr = module_it->modulePtr;

    if (!module_ptr) {
        log_error("error. no object behind pointer when deleting module: " + request);
        return false;
    }

    // iterate parameters, remove hotkeys from CommandRegistry
    for (auto child = module_ptr->ChildList_Begin(); child != module_ptr->ChildList_End(); ++child) {
        auto ps = dynamic_cast<param::ParamSlot*>((*child).get());
        if (ps != nullptr) {
            auto p = ps->Param<param::ButtonParam>();
            if (p != nullptr) {
                m_command_registry->remove_command_by_parent(ps->FullName().PeekBuffer());
            }
        }
    }

    // delete all outgoing/incoming calls
    auto discard_calls = find_all_of(call_list_, [&](CallInstance_t const& call_info) {
        return (call_info.request.from.find(request) != std::string::npos ||
                call_info.request.to.find(request) != std::string::npos);
    });

    std::for_each(discard_calls.begin(), discard_calls.end(), [&](auto const& call_it) {
        delete_call(CallDeletionRequest_t{call_it->request.from, call_it->request.to});
    });

    if (module_it->isGraphEntryPoint)
        this->RemoveGraphEntryPoint(request);

    const auto release_module = [module_ptr](auto& module_lifetime_dependencies) -> bool {
        module_ptr->Release(module_lifetime_dependencies);
        log("release module: " + std::string(module_ptr->Name().PeekBuffer()));
        return true;
        // end of lambda scope deletes last shared_ptr to module
        // thus the module gets deleted after execution and deletion of this command callback
    };

    release_module(module_it->lifetime_resources);

    this->module_list_.erase(module_it);

    return true;
}


bool megamol::core::MegaMolGraph::delete_call(CallDeletionRequest_t const& request) {

    auto call_it = find_call(request.from, request.to);

    if (call_it == this->call_list_.end()) {
        log_error("error. could not find call for deletion: " + request.from + " -> " + request.to);
        return false;
    }

    auto target = call_it->callPtr->PeekCalleeSlotNoConst();
    auto source = call_it->callPtr->PeekCallerSlotNoConst();

    if (!target || !source) {
        log_error("error. could not get callee or caller slot for call deletion of call: " +
                  std::string(call_it->callPtr->ClassName()) + "\n(" + request.from + " -> " + request.to + ")");
        return false;
    }

#ifdef PROFILING
    auto the_call = call_it->callPtr;
    m_perf_manager->remove_timers(the_call->cpu_queries);
    if (the_call->GetCapabilities().OpenGLRequired()) {
        m_perf_manager->remove_timers(the_call->gl_queries);
    }
#endif

    source->SetCleanupMark(true);
    source->DisconnectCalls();
    source->PerformCleanup();  // does nothing
    target->DisconnectCalls(); // does nothing

    this->call_list_.erase(call_it);

    return true;
}

static const auto check_module_is_prefix = [](std::string const& request, auto const& module) {
    const auto& module_name = module.request.id;
    const auto substring = request.substr(0, module_name.size());
    return (module_name == substring) &&         // module name is prefix of request
           (request.size() == module_name.size() // module name matches whole request
               || (request.size() >= module_name.size() + 2 &&
                      request.substr(module_name.size(), 2) == "::")); // OR request has :: after module name
};

// find module where module name is prefix of request
megamol::core::ModuleList_t::iterator megamol::core::MegaMolGraph::find_module_by_prefix(std::string const& request) {
    return std::find_if(module_list_.begin(), module_list_.end(),
        [&](auto const& module) { return check_module_is_prefix(request, module); });
}

megamol::core::ModuleList_t::const_iterator megamol::core::MegaMolGraph::find_module_by_prefix(
    std::string const& request) const {
    return std::find_if(module_list_.begin(), module_list_.end(),
        [&](auto const& module) { return check_module_is_prefix(request, module); });
}
