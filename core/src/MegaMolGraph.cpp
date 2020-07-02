#include <iostream>

#include "stdafx.h"
#include "mmcore/MegaMolGraph.h"

#include "mmcore/AbstractSlot.h"
#include "mmcore/view/AbstractView_EventConsumption.h"

// splits a string of the form "::one::two::three::" into an array of strings {"one", "two", "three"}
static std::vector<std::string> splitPathName(std::string const& path) {
    std::vector<std::string> result;

    size_t start = 0;
    while ((start = path.find_first_not_of(':', start)) != std::string::npos) {
        auto end = path.find_first_of(':', start);
        if (start < end) result.push_back(path.substr(start, end - start));
        start = end;
    }

    return result;
}

static void log(std::string text) { std::cout << "MegaMolGraph: " << text << std::endl; }

static megamol::core::param::AbstractParam* getParameterFromParamSlot(megamol::core::param::ParamSlot* param_slot) {

    if (param_slot->GetStatus() == megamol::core::AbstractSlot::STATUS_UNAVAILABLE) {
        log("error. cannot find parameter: " + std::string(param_slot->Name().PeekBuffer()) +
            ", slot is not available");
        return nullptr;
    }
    if (param_slot->Parameter().IsNull()) {
        log("error. cannot find parameter: " + std::string(param_slot->Name().PeekBuffer()) +
            ", slot has no parameter");
        return nullptr;
    }

    return param_slot->Parameter().DynamicCast<megamol::core::param::AbstractParam>();
}

megamol::core::MegaMolGraph::MegaMolGraph(megamol::core::CoreInstance& core,
    factories::ModuleDescriptionManager const& moduleProvider, factories::CallDescriptionManager const& callProvider)
    : moduleProvider_ptr{&moduleProvider}
    , callProvider_ptr{&callProvider}
    , dummy_namespace{std::make_shared<RootModuleNamespace>()} {
    // the Core Instance is a parasite that needs to be passed to all modules
    // TODO: make it so there is no more core instance
    dummy_namespace->SetCoreInstance(core);
}

/**
 * A move of the graph should be OK, even without changing state of Modules in graph.
 */
megamol::core::MegaMolGraph::MegaMolGraph(MegaMolGraph&& rhs) noexcept {}

/**
 * Same is true for move-assignment.
 */
megamol::core::MegaMolGraph& megamol::core::MegaMolGraph::operator=(MegaMolGraph&& rhs) noexcept { return *this; }

/**
 * Construction from serialized string.
 */
// megamol::core::MegaMolGraph::MegaMolGraph(std::string const& descr) {
//}

/** dtor */
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

bool megamol::core::MegaMolGraph::DeleteModule(std::string const& id) {
    return delete_module(id);
}


bool megamol::core::MegaMolGraph::CreateModule(std::string const& className, std::string const& id) {
    return add_module(ModuleInstantiationRequest{className, id});
}

bool megamol::core::MegaMolGraph::DeleteCall(std::string const& from, std::string const& to) {
    return delete_call(CallDeletionRequest{from, to});
}


bool megamol::core::MegaMolGraph::CreateCall(
    std::string const& className, std::string const& from, std::string const& to) {
    return add_call(CallInstantiationRequest{className, from, to});
}

megamol::core::MegaMolGraph::ModuleList_t::iterator megamol::core::MegaMolGraph::find_module(std::string const& name) {
    auto it = std::find_if(this->module_list_.begin(), this->module_list_.end(),
        [&name](megamol::core::MegaMolGraph::ModuleInstance_t const& el) { return el.request.id == name; });

    return it;
}

megamol::core::MegaMolGraph::ModuleList_t::const_iterator megamol::core::MegaMolGraph::find_module(
    std::string const& name) const {

    auto it = std::find_if(this->module_list_.cbegin(), this->module_list_.cend(),
        [&name](megamol::core::MegaMolGraph::ModuleInstance_t const& el) { return el.request.id == name; });

    return it;
}

megamol::core::MegaMolGraph::CallList_t::iterator megamol::core::MegaMolGraph::find_call(
    std::string const& from, std::string const& to) {
    auto it = std::find_if(
        this->call_list_.begin(), this->call_list_.end(), [&](megamol::core::MegaMolGraph::CallInstance_t const& el) {
            return el.second.from == from && el.second.to == to;
        });

    return it;
}

megamol::core::MegaMolGraph::CallList_t::const_iterator megamol::core::MegaMolGraph::find_call(
    std::string const& from, std::string const& to) const {

    auto it = std::find_if(
        this->call_list_.cbegin(), this->call_list_.cend(), [&](megamol::core::MegaMolGraph::CallInstance_t const& el) {
            return el.second.from == from && el.second.to == to;
        });

    return it;
}


bool megamol::core::MegaMolGraph::add_module(ModuleInstantiationRequest_t const& request) {
    factories::ModuleDescription::ptr module_description = this->ModuleProvider().Find(request.className.c_str());
    if (!module_description) {
        log("error. module factory could not find module class name: " + request.className);
        return false;
    }

    const auto path_parts = splitPathName(request.id);
    if (path_parts.empty()) {
        log("error. requested module name does not seem to have valid namespace format: " + request.id +
            "\n. valid format is: [::]aa::bb::cc[::]");
        return false;
    }

    const auto module_name = vislib::StringA(path_parts.back().c_str());

    Module::ptr_type module_ptr = Module::ptr_type(module_description->CreateModule(module_name));
    if (!module_ptr) {
        log("error. could not instantiate module from module description: " + request.className);
        return false;
    }

	auto module_lifetime_dependencies_request = module_ptr->requested_lifetime_dependencies();

	auto module_lifetime_dependencies = get_requested_dependencies(module_lifetime_dependencies_request);

	if (module_lifetime_dependencies.size() != module_lifetime_dependencies_request.size()) {
        std::string requested_deps = "";
        std::string found_deps = "";
        for (auto& req : module_lifetime_dependencies_request) requested_deps += " " + req;
        for (auto& dep : module_lifetime_dependencies) found_deps += " " + dep.getIdentifier();
        log("error. could not create module, not all requested dependencies available: ");
        log("requested: " + requested_deps);
        log("found: " + found_deps);

		return false;
    }

    this->module_list_.push_front({module_ptr, request, false, module_lifetime_dependencies_request, module_lifetime_dependencies});

    module_ptr->setParent(this->dummy_namespace);

    const auto create_module = [module_description, module_ptr](auto& module_lifetime_dependencies) {
        const bool init_ok = module_ptr->Create(module_lifetime_dependencies); // seems like Create() internally checks IsAvailable()

        if (!init_ok)
            log("error. could not create module, IsAvailable() or Create() failed: " +
                std::string((module_ptr->Name()).PeekBuffer()));
        else
            log("create module: " + std::string((module_ptr->Name()).PeekBuffer()));

        return init_ok;
    };

	bool isCreateOk = create_module(this->module_list_.front().lifetime_dependencies);

	if (!isCreateOk) {
        this->module_list_.pop_front();
	}

    return isCreateOk;
}

bool megamol::core::MegaMolGraph::add_call(CallInstantiationRequest_t const& request) {

    factories::CallDescription::ptr call_description = this->CallProvider().Find(request.className.c_str());

    const auto getCallSlotOfModule = [this, &call_description](
                                         std::string const& name) -> std::pair<bool, AbstractSlot*> {
        const auto path = splitPathName(name);
        if (path.empty()) {
            log("error. encountered invalid namespace format: " + name + "\n. valid format is: [::]aa::bb::cc[::]");
            return {false, nullptr};
        }

        auto module_name = name.substr(0, name.size() - (path.back().size() + 2));
        auto module_it = this->find_module(module_name);
        if (module_it == this->module_list_.end()) {
            log("error. could not find module named: " + module_name +
                " to connect requested call: " + call_description->ClassName());
            return {false, nullptr};
        }

        Module::ptr_type module_ptr = module_it->modulePtr;
        const auto slot_name = vislib::StringA(path.back().c_str());
        AbstractSlot* slot_ptr = module_ptr->FindSlot(slot_name);
        if (!slot_ptr) {
            log("error. could not find slot named: " + std::string(slot_name.PeekBuffer()) +
                " to connect requested call: " + std::string(call_description->ClassName()));
            return {false, nullptr};
        }

        if (!slot_ptr->IsCallCompatible(call_description)) {
            log("error. call: " + std::string(call_description->ClassName()) +
                " is not compatible with slot: " + std::string(slot_name.PeekBuffer()));
            return {false, nullptr};
        }

        if (!slot_ptr->GetStatus() == AbstractSlot::STATUS_ENABLED) {
            log("error. slot: " + std::string(slot_name.PeekBuffer()) +
                " is not enabled. can not connect call: " + std::string(call_description->ClassName()));
            return {false, nullptr};
        }

        return {true, slot_ptr};
    };

    auto from_slot = getCallSlotOfModule(request.from);
    if (from_slot.first == false) {
        log("error. could not find from-slot: " + request.from +
            " for call: " + std::string(call_description->ClassName()));
        return false; // error when looking for from-slot
    }
    CallerSlot* caller = dynamic_cast<CallerSlot*>(from_slot.second);

    auto to_slot = getCallSlotOfModule(request.to);
    if (to_slot.first == false) {
        log("error. could not find to-slot: " + request.to +
            " for call: " + std::string(call_description->ClassName()));
        return false; // error when looking for to-slot
    }
    CalleeSlot* callee = dynamic_cast<CalleeSlot*>(to_slot.second);

    if ((caller->GetStatus() == AbstractSlot::STATUS_CONNECTED) ||
        (callee->GetStatus() == AbstractSlot::STATUS_CONNECTED)) {
        Call* tstCall = caller->IsConnectedTo(callee);
        if (tstCall && call_description->IsDescribing(tstCall)) {
            log("error. caller (" + request.from + ") and callee (" + request.to +
                ") are already connected by call: " + std::string(call_description->ClassName()));
            return false; // call already exists
        }
    }

    // TODO: kill parents of modules/calls when new graph structure is in place
    callee->setParent(this->dummy_namespace);
    caller->setParent(this->dummy_namespace);

    Call::ptr_type call = Call::ptr_type(call_description->CreateCall());
    if (!callee->ConnectCall(call.get(), call_description)) {
        log("error. connecting call: " + std::string(call_description->ClassName()) + " failed at callee");
        return false;
    }
    if (!caller->ConnectCall(call.get())) {
        log("error. connecting call: " + std::string(call_description->ClassName()) + " failed at caller");
        // FIXME: if connecting the callER fails, how to disconnect call from callEE?
        // callee->DisconnectCalls();
        return false;
    }

    log("create call: " + request.from + " -> " + request.to + " (" + std::string(call_description->ClassName()) + ")");
    this->call_list_.emplace_front(call, request);

    return true;
}

static std::list<megamol::core::MegaMolGraph::CallList_t::iterator> find_all_of(
    megamol::core::MegaMolGraph::CallList_t list,
    std::function<bool(megamol::core::MegaMolGraph::CallInstance_t const&)> const& func) {

    std::list<megamol::core::MegaMolGraph::CallList_t::iterator> result;

    for (auto begin = list.begin(); begin != list.end(); begin++)
        if (func(*begin)) result.push_back(begin);

    return result;
}


bool megamol::core::MegaMolGraph::delete_module(ModuleDeletionRequest_t const& request) {

    auto module_it = find_module(request);
    if (module_it == this->module_list_.end()) {
        log("error. could not find module for deletion: " + request);
        return false;
    }

    auto module_ptr = module_it->modulePtr;

    if (!module_ptr) {
        log("error. no object behind pointer when deleting module: " + request);
        return false;
    }

    // delete all outgoing/incoming calls
    auto discard_calls = find_all_of(call_list_,
        [&](auto const& call_info) { return (call_info.second.from == request || call_info.second.to == request); });

    std::for_each(discard_calls.begin(), discard_calls.end(), [&](auto const& call_it) {
        delete_call(CallDeletionRequest_t{call_it->second.from, call_it->second.to});
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

    release_module(module_it->lifetime_dependencies);

    this->module_list_.erase(module_it);

    return true;
}


bool megamol::core::MegaMolGraph::delete_call(CallDeletionRequest_t const& request) {

    auto call_it = find_call(request.from, request.to);

    if (call_it == this->call_list_.end()) {
        log("error. could not find call for deletion: " + request.from + " -> " + request.to);
        return false;
    }

    auto target = call_it->first->PeekCalleeSlotNoConst();
    auto source = call_it->first->PeekCallerSlotNoConst();

    if (!target || !source) {
        log("error. could not get callee or caller slot for call deletion of call: " +
            std::string(call_it->first->ClassName()) + "\n(" + request.from + " -> " + request.to + ")");
        return false;
    }

    source->SetCleanupMark(true);
    source->DisconnectCalls();
    source->PerformCleanup();  // does nothing
    target->DisconnectCalls(); // does nothing

    this->call_list_.erase(call_it);

    return true;
}


void megamol::core::MegaMolGraph::RenderNextFrame() {
	for (auto& entry : graph_entry_points) {
        entry.execute(entry.modulePtr, entry.entry_point_dependencies);
    }
}

megamol::core::Module::ptr_type megamol::core::MegaMolGraph::FindModule(std::string const& moduleName) const {
    auto module_it = find_module(moduleName);

    if (module_it == module_list_.end()) {
        log("error. could not find module: " + moduleName);
        return nullptr;
    }

    return module_it->modulePtr;
}

megamol::core::Call::ptr_type megamol::core::MegaMolGraph::FindCall(
    std::string const& from, std::string const& to) const {
    auto call_it = find_call(from, to);

    if (call_it == call_list_.end()) {
        log("error. could not find call: " + from + " -> " + to);
        return nullptr;
    }

    auto call_ptr = call_it->first;
    return call_ptr;
}

megamol::core::param::ParamSlot* megamol::core::MegaMolGraph::FindParameterSlot(std::string const& paramName) const {
    auto names = splitPathName(paramName);
    if (names.size() < 2) {
        log("error. could not find parameter, parameter name has invalid format: " + paramName +
            "\n(expected format: [::]aa::bb::cc[::]");
        return nullptr;
    }

    auto module_name = paramName.substr(0, paramName.size() - (names.back().size() + 2));
    auto module_it = find_module(module_name);

    if (module_it == module_list_.end()) {
        log("error. could not find parameter, module name not found: " + module_name +
            " (parameter name: " + paramName + ")");
        return nullptr;
    }

    auto& module = *module_it->modulePtr;
    std::string slot_name = names.back();
    AbstractSlot* slot_ptr = module.FindSlot(slot_name.c_str());
    param::ParamSlot* param_slot_ptr = dynamic_cast<param::ParamSlot*>(slot_ptr);

    if (slot_ptr == nullptr || param_slot_ptr == nullptr) {
        log("error. could not find parameter, slot not found or of wrong type. parameter name: " + paramName +
            ", slot name: " + slot_name);
        return nullptr;
    }

    return param_slot_ptr;
}

megamol::core::param::AbstractParam* megamol::core::MegaMolGraph::FindParameter(std::string const& paramName) const {
    return getParameterFromParamSlot(this->FindParameterSlot(paramName));
}


std::vector<megamol::core::param::ParamSlot*> megamol::core::MegaMolGraph::EnumerateModuleParameterSlots(
    std::string const& moduleName) const {
    std::vector<megamol::core::param::ParamSlot*> parameters;

    auto module_it = find_module(moduleName);

    if (module_it == module_list_.end()) {
        log("error. could not find module: " + moduleName);
        return parameters;
    }

    auto children_begin = module_it->modulePtr->ChildList_Begin();
    auto children_end = module_it->modulePtr->ChildList_End();

    while (children_begin != children_end) {
        AbstractNamedObject::ptr_type named_object = *children_begin;
        if (named_object != nullptr) {
            AbstractSlot* slot_ptr = dynamic_cast<AbstractSlot*>(named_object.get());
            param::ParamSlot* param_slot_ptr = dynamic_cast<param::ParamSlot*>(slot_ptr);

            if (slot_ptr && param_slot_ptr) parameters.push_back(param_slot_ptr);
        }

        children_begin++;
    }

    return parameters;
}

std::vector<megamol::core::param::AbstractParam*> megamol::core::MegaMolGraph::EnumerateModuleParameters(
    std::string const& moduleName) const {
    auto param_slots = EnumerateModuleParameterSlots(moduleName);

    std::vector<megamol::core::param::AbstractParam*> params;
    params.reserve(param_slots.size());

    for (auto& slot : param_slots) params.push_back(getParameterFromParamSlot(slot));

    return params;
}

megamol::core::MegaMolGraph::CallList_t const& megamol::core::MegaMolGraph::ListCalls() const { return call_list_; }

megamol::core::MegaMolGraph::ModuleList_t const& megamol::core::MegaMolGraph::ListModules() const {
    return module_list_;
}

std::vector<megamol::core::param::ParamSlot*> megamol::core::MegaMolGraph::ListParameterSlots() const {
    std::vector<megamol::core::param::ParamSlot*> param_slots;

    for (auto& mod : module_list_) {
        auto module_params = this->EnumerateModuleParameterSlots(mod.modulePtr->Name().PeekBuffer());

        param_slots.insert(param_slots.end(), module_params.begin(), module_params.end());
    }

    return param_slots;
}

std::vector<megamol::core::param::AbstractParam*> megamol::core::MegaMolGraph::ListParameters() const {
    auto param_slots = this->ListParameterSlots();

    std::vector<megamol::core::param::AbstractParam*> parameters;
    parameters.reserve(param_slots.size());

    for (auto& slot : param_slots) parameters.push_back(getParameterFromParamSlot(slot));

    return parameters;
}

bool megamol::core::MegaMolGraph::SetGraphEntryPoint(std::string moduleName, std::vector<std::string> execution_dependencies, EntryPointExecutionCallback callback) {
    auto module_it = find_module(moduleName);

    if (module_it == module_list_.end()) {
        log("error. could not find module: " + moduleName);
        return false;
    }

    auto module_ptr = module_it->modulePtr;

	auto dependencies = get_requested_dependencies(execution_dependencies);

	this->graph_entry_points.push_back({moduleName, module_ptr, dependencies, callback});

    return true;
}

bool megamol::core::MegaMolGraph::RemoveGraphEntryPoint(std::string moduleName) {
    auto module_it = find_module(moduleName);

    if (module_it == module_list_.end()) {
        log("error. could not find module: " + moduleName);
        return false;
    }

    auto module_ptr = module_it->modulePtr;
    megamol::core::view::AbstractView* view_ptr = dynamic_cast<megamol::core::view::AbstractView*>(module_ptr.get());

    if (!view_ptr) {
        log("error. module: " + moduleName + " is not a view module. could not set as graph rendering entry point.");
        return false;
    }

	this->graph_entry_points.remove_if([&](GraphEntryPoint& entry) { return entry.moduleName == moduleName; });

    module_it->isGraphEntryPoint = false;

    return true;
}

void megamol::core::MegaMolGraph::AddModuleDependencies(std::vector<megamol::render_api::RenderResource> const& dependencies) {
    this->provided_dependencies.insert(provided_dependencies.end(), dependencies.begin(), dependencies.end());
}

std::vector<megamol::render_api::RenderResource> megamol::core::MegaMolGraph::get_requested_dependencies(std::vector<std::string> dependency_requests) {
    std::vector<megamol::render_api::RenderResource> result;
    result.reserve(dependency_requests.size());

    for (auto& request : dependency_requests) {
        auto dependency_it = std::find_if(this->provided_dependencies.begin(), this->provided_dependencies.end(), [&](megamol::render_api::RenderResource& dependency){
			return request == dependency.getIdentifier();
		});

		if (dependency_it != provided_dependencies.end())
			result.push_back(*dependency_it);
    }

	return result;
}

