#include <iostream>

#include "stdafx.h"
#include "mmcore/MegaMolGraph.h"

#include "mmcore/view/AbstractView_EventConsumption.h"
#include "mmcore/AbstractSlot.h"

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

static void log(std::string text) {
	std::cout << "MegaMolGraph: " << text << std::endl;
}

static megamol::core::param::AbstractParam* getParameterFromParamSlot(megamol::core::param::ParamSlot* param_slot) {

    if (param_slot->GetStatus() == megamol::core::AbstractSlot::STATUS_UNAVAILABLE) {
        log("error. cannot find parameter: " + std::string(param_slot->Name().PeekBuffer()) + ", slot is not available");
		return nullptr;
	}
    if (param_slot->Parameter().IsNull()) {
        log("error. cannot find parameter: " + std::string(param_slot->Name().PeekBuffer()) + ", slot has no parameter");
        return nullptr;
    }

    return param_slot->Parameter().DynamicCast<megamol::core::param::AbstractParam>();
}

megamol::core::MegaMolGraph::MegaMolGraph(megamol::core::CoreInstance& core,
    factories::ModuleDescriptionManager const& moduleProvider, factories::CallDescriptionManager const& callProvider,
    std::unique_ptr<render_api::AbstractRenderAPI> rapi, std::string rapi_name)
    : moduleProvider_ptr{&moduleProvider}
    , callProvider_ptr{&callProvider}
    , rapi_{std::move(rapi)}
    , rapi_root_name{rapi_name}
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
    if (this->rapi_) rapi_->closeAPI();

    moduleProvider_ptr = nullptr;
    callProvider_ptr = nullptr;
}

const megamol::core::factories::ModuleDescriptionManager& megamol::core::MegaMolGraph::ModuleProvider() {
    return *moduleProvider_ptr; // not null because we took it by reference in constructor
}

const megamol::core::factories::CallDescriptionManager& megamol::core::MegaMolGraph::CallProvider() {
    return *callProvider_ptr;
}

bool megamol::core::MegaMolGraph::DeleteModule(std::string const& id) { return delete_module(id); }


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


bool megamol::core::MegaMolGraph::HasPendingRequests() { return this->rapi_commands.size() > 0; }

megamol::core::MegaMolGraph::ModuleList_t::iterator megamol::core::MegaMolGraph::find_module(
    std::string const& name) {
    auto it = std::find_if(this->module_list_.begin(), this->module_list_.end(),
        [&name](megamol::core::MegaMolGraph::ModuleInstance_t const& el) { return el.second.id == name; });

    return it;
}

megamol::core::MegaMolGraph::ModuleList_t::const_iterator megamol::core::MegaMolGraph::find_module(
        std::string const& name) const {

    auto it = std::find_if(this->module_list_.cbegin(), this->module_list_.cend(),
        [&name](megamol::core::MegaMolGraph::ModuleInstance_t const& el) { return el.second.id == name; });

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
        log("error. requested module name does not seem to have valid namespace format: " + request.id + "\n. valid format is: [::]aa::bb::cc[::]");
		return false;
	}

    const auto module_name = vislib::StringA(path_parts.back().c_str());

    Module::ptr_type module_ptr = Module::ptr_type(module_description->CreateModule(module_name));
    if (!module_ptr) {
        log("error. could not instantiate module from module description: " + request.className);
		return false;
    }

    this->module_list_.emplace_front(module_ptr, request);

    module_ptr->setParent(this->dummy_namespace);

    // execute IsAvailable() and Create() in GL context
    this->rapi_commands.emplace_front([module_description, module_ptr]() {

        const bool init_ok = module_description->IsAvailable() && module_ptr->Create();

        if (!init_ok)
			log("error. could not create module, IsAvailable() or Create() failed: " + std::string((module_ptr->Name()).PeekBuffer()));
        else
			log("create module: " + std::string((module_ptr->Name()).PeekBuffer()));

        return init_ok;
    }); // returns false if something went wrong

    // if the new module is a view module register if with a View Resource Feeder and set it up to get the default
    // resources of the GLFW context plus an empty handler for rendering
    megamol::core::view::AbstractView* view_ptr = nullptr;
    if (view_ptr = dynamic_cast<megamol::core::view::AbstractView*>(module_ptr.get())) {
        this->view_feeders.push_back(ViewResourceFeeder{
            view_ptr, {
                          // rendering resource handlers are executed in the order defined here
                          std::make_pair("KeyboardEvents", megamol::core::view::view_consume_keyboard_events),
                          std::make_pair("MouseEvents", megamol::core::view::view_consume_mouse_events),
                          std::make_pair("WindowEvents", megamol::core::view::view_consume_window_events),
                          std::make_pair("FramebufferEvents", megamol::core::view::view_consume_framebuffer_events),
                          std::make_pair("", megamol::core::view::view_poke_rendering),
                      }});
    }

    // TODO: make sure that requested rendering resources or inputs for the view are provided by some RAPI

    return true;
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

        auto module_name = path[0] + "::" + path[1];
        auto module_it = this->find_module(module_name);
        if (module_it == this->module_list_.end()) {
			log("error. could not find module named: " + module_name + " to connect requested call: " + call_description->ClassName());
			return {false, nullptr};
        }

        Module::ptr_type module_ptr = module_it->first;
        const auto slot_name = vislib::StringA(path.back().c_str());
        AbstractSlot* slot_ptr = module_ptr->FindSlot(slot_name);
        if (!slot_ptr) {
			log("error. could not find slot named: " + std::string(slot_name.PeekBuffer()) + " to connect requested call: " + std::string(call_description->ClassName()));
			return {false, nullptr};
        }

        if (!slot_ptr->IsCallCompatible(call_description)) {
			log("error. call: " + std::string(call_description->ClassName()) + " is not compatible with slot: " + std::string(slot_name.PeekBuffer()));
			return {false, nullptr};
        }

        if (!slot_ptr->GetStatus() == AbstractSlot::STATUS_ENABLED) {
			log("error. slot: " + std::string(slot_name.PeekBuffer()) + " is not enabled. can not connect call: " + std::string(call_description->ClassName()));
			return {false, nullptr};
        }

        return {true, slot_ptr};
    };

    auto from_slot = getCallSlotOfModule(request.from);
    if (from_slot.first == false) {
		log("error. could not find from-slot: " + request.from + " for call: " + std::string(call_description->ClassName()));
		return false; // error when looking for from-slot
    }
    CallerSlot* caller = dynamic_cast<CallerSlot*>(from_slot.second);

    auto to_slot = getCallSlotOfModule(request.to);
    if (to_slot.first == false) {
		log("error. could not find to-slot: " + request.to + " for call: " + std::string(call_description->ClassName()));
		return false; // error when looking for to-slot
    }
    CalleeSlot* callee = dynamic_cast<CalleeSlot*>(to_slot.second);

    if ((caller->GetStatus() == AbstractSlot::STATUS_CONNECTED) ||
        (callee->GetStatus() == AbstractSlot::STATUS_CONNECTED)) {
        Call* tstCall = caller->IsConnectedTo(callee);
        if (tstCall && call_description->IsDescribing(tstCall)) {
			log("error. caller (" + request.from + ") and callee (" + request.to + ") are already connected by call: " + std::string(call_description->ClassName()));
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

    auto module_ptr = module_it->first; // is std::shared_ptr, a copy stays alive until rapi_commands got executed and
                                        // the vector gets cleared
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

    // call Release() in GL context
    this->rapi_commands.emplace_back([module_ptr]() -> bool {
        module_ptr->Release();
        log("release module: " + std::string(module_ptr->Name().PeekBuffer()));
        return true;
        // end of lambda scope deletes last shared_ptr to module
        // thus the module gets deleted after execution and deletion of this command callback
    });

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
        log("error. could not get callee or caller slot for call deletion of call: " + std::string(call_it->first->ClassName()) 
			+ "\n(" + request.from + " -> " + request.to + ")");
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

    if (this->rapi_) this->rapi_->preViewRender();

    // OpenGL context for module Create() provided here by preViewRender() of RAPI with GL context
    bool some_command_failed = false;
    for (auto& command : this->rapi_commands)
        some_command_failed |= !command(); // module Create() or Release() called here

    this->rapi_commands.clear();

    if (some_command_failed) {
		// TODO
        // fail and stop execution of MegaMol because without the requested graph modules further execution makes no
        // sense
        for (auto& m : module_list_) {
            if (!m.first->isCreated())
				log("error. module not created: " + m.second.id + ", " + m.second.className);
        }
    }

    // process ui events and other resources
    // this also contains a handler that tells the view to render itself
    auto& resources = this->rapi_->getRenderResources();
    for (auto& view_feeder : view_feeders) view_feeder.consume(resources);

    // TODO: handle 'stop rendering' requests

    if (this->rapi_) this->rapi_->postViewRender();
}

megamol::core::Module::ptr_type megamol::core::MegaMolGraph::FindModule(std::string const& moduleName) const {
    auto module_it = find_module(moduleName);
	
	if (module_it == module_list_.end()) {
		log("error. could not find module: " + moduleName);
		return nullptr;
	}

    auto module_ptr = module_it->first;
    return module_ptr;
}

megamol::core::Call::ptr_type megamol::core::MegaMolGraph::FindCall(std::string const& from, std::string const& to) const {
    auto call_it = find_call(from, to);
	
	if (call_it == call_list_.end()) {
		log("error. could not find call: " + from + " -> " + to);
		return nullptr;
    }

    auto call_ptr = call_it->first;
    return call_ptr;
}

megamol::core::param::AbstractParam* megamol::core::MegaMolGraph::FindParameter(std::string const& paramName) const {
    auto names = splitPathName(paramName);
    if (names.size() < 2) {
		log("error. could not find parameter, parameter name has invalid format: " + paramName + "\n(expected format: [::]aa::bb::cc[::]");
		return nullptr;
    }

	auto module_name = paramName.substr(0, paramName.size() - (names.back().size() + 2));
    auto module_it = find_module(module_name);

	if (module_it == module_list_.end()) {
		log("error. could not find parameter, module name not found: " + module_name + " (parameter name: " + paramName + ")");
		return nullptr;
	}

    auto& module = *(module_it->first);
    std::string slot_name = names.back();
    AbstractSlot* slot_ptr = module.FindSlot(slot_name.c_str());
    param::ParamSlot* param_slot_ptr = dynamic_cast<param::ParamSlot*>(slot_ptr);

	if (slot_ptr == nullptr || param_slot_ptr == nullptr) {
		log("error. could not find parameter, slot not found or of wrong type. parameter name: " + paramName + ", slot name: " + slot_name);
		return nullptr;
    }

	return getParameterFromParamSlot(param_slot_ptr);
}

std::vector<megamol::core::param::AbstractParam*> megamol::core::MegaMolGraph::FindModuleParameters(std::string const& moduleName) const {
    std::vector<megamol::core::param::AbstractParam*> parameters;

    auto module_it = find_module(moduleName);

	if (module_it == module_list_.end()) {
		log("error. could not find module: " + moduleName);
		return parameters;
    }

	auto children_begin = module_it->first->ChildList_Begin();
	auto children_end = module_it->first->ChildList_End();

	while (children_begin != children_end) {
        AbstractNamedObject::ptr_type named_object = *children_begin;
        if (named_object != nullptr) {
			AbstractSlot* slot_ptr = dynamic_cast<AbstractSlot*>(named_object.get());
			param::ParamSlot* param_slot_ptr = dynamic_cast<param::ParamSlot*>(slot_ptr);

			if (slot_ptr && param_slot_ptr)
				parameters.push_back(getParameterFromParamSlot(param_slot_ptr));
		}

		children_begin++;
	}

	return parameters;
}

megamol::core::MegaMolGraph::CallList_t const& megamol::core::MegaMolGraph::ListCalls() const {
	return call_list_;
}

megamol::core::MegaMolGraph::ModuleList_t const& megamol::core::MegaMolGraph::ListModules() const {
    return module_list_;
}

std::vector<megamol::core::param::AbstractParam*> megamol::core::MegaMolGraph::ListParameters() const {
    std::vector<megamol::core::param::AbstractParam*> parameters;

	for (auto& mod : module_list_) {
        auto module_params = this->FindModuleParameters(mod.first->Name().PeekBuffer());

		parameters.insert(parameters.end(), module_params.begin(), module_params.end());
    }

	return parameters;
}

