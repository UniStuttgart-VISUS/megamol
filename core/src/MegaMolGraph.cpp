#include "stdafx.h"
#include "mmcore/MegaMolGraph.h"


megamol::core::MegaMolGraph::MegaMolGraph() {
}

/**
 * A move of the graph should be OK, even without changing state of Modules in graph.
 */
megamol::core::MegaMolGraph::MegaMolGraph(MegaMolGraph&& rhs) noexcept {
}

/**
 * Same is true for move-assignment.
 */
megamol::core::MegaMolGraph& megamol::core::MegaMolGraph::operator=(MegaMolGraph&& rhs) noexcept {
	return *this;
}

/**
 * Construction from serialized string.
 */
megamol::core::MegaMolGraph::MegaMolGraph(std::string const& descr) {
}

/** dtor */
megamol::core::MegaMolGraph::~MegaMolGraph() {
}

bool megamol::core::MegaMolGraph::QueueModuleDeletion(std::string const& id) {
    auto lock = module_deletion_queue_.AcquireLock();
    return QueueModuleDeletion(id);
}


bool megamol::core::MegaMolGraph::QueueModuleDeletionNoLock(std::string const& id) {
    module_deletion_queue_.Push(id);
    return true;
}


bool megamol::core::MegaMolGraph::QueueModuleInstantiation(std::string const& className, std::string const& id) {
    auto lock = module_instantiation_queue_.AcquireLock();
    return QueueModuleInstantiationNoLock(className, id);
}


bool megamol::core::MegaMolGraph::QueueModuleInstantiationNoLock(std::string const& className, std::string const& id) {
    module_instantiation_queue_.Emplace(ModuleInstantiationRequest{className, id});
    return true;
}


bool megamol::core::MegaMolGraph::QueueCallDeletion(std::string const& from, std::string const& to) {
    auto lock = call_deletion_queue_.AcquireLock();
    return QueueCallDeletionNoLock(from, to);
}


bool megamol::core::MegaMolGraph::QueueCallDeletionNoLock(std::string const& from, std::string const& to) {
    call_deletion_queue_.Emplace(CallDeletionRequest{from, to});
    return true;
}


bool megamol::core::MegaMolGraph::QueueCallInstantiation(
    std::string const& className, std::string const& from, std::string const& to) {
    auto lock = call_instantiation_queue_.AcquireLock();
    return QueueCallInstantiationNoLock(className, from, to);
}


bool megamol::core::MegaMolGraph::QueueCallInstantiationNoLock(
    std::string const& className, std::string const& from, std::string const& to) {
    call_instantiation_queue_.Emplace(CallInstantiationRequest{className, from, to});
    return true;
}


bool megamol::core::MegaMolGraph::HasPendingRequests() {
    auto lock = AcquireQueueLocks();
    return !module_deletion_queue_.Empty() || !module_instantiation_queue_.Empty() || !call_deletion_queue_.Empty() ||
           !call_instantiation_queue_.Empty();
}

std::scoped_lock<std::unique_lock<std::mutex>, std::unique_lock<std::mutex>, std::unique_lock<std::mutex>,
    std::unique_lock<std::mutex>>
megamol::core::MegaMolGraph::AcquireQueueLocks() {
    auto lock1 = module_deletion_queue_.AcquireDeferredLock();
    auto lock2 = module_instantiation_queue_.AcquireDeferredLock();
    auto lock3 = call_deletion_queue_.AcquireDeferredLock();
    auto lock4 = call_instantiation_queue_.AcquireDeferredLock();
    return std::scoped_lock(lock1, lock2, lock3, lock4);
}

[[nodiscard]]
megamol::core::MegaMolGraph::ModuleList_t::iterator megamol::core::MegaMolGraph::find_module(std::string const& name) {
    std::shared_lock<std::shared_mutex> lock(graph_lock_);

    auto it = std::find_if(
        this->module_list_.begin(), this->module_list_.end(), [&name](megamol::core::MegaMolGraph::ModuleDescr_t const& el) { return el.second.id == name; });

    return it;
}

[[nodiscard]]
megamol::core::MegaMolGraph::ModuleList_t::const_iterator megamol::core::MegaMolGraph::find_module(std::string const& name) const {
    std::shared_lock<std::shared_mutex> lock(graph_lock_);

    auto it = std::find_if(
        this->module_list_.cbegin(), this->module_list_.cend(), [&name](megamol::core::MegaMolGraph::ModuleDescr_t const& el) { return el.second.id == name; });

    return it;
}

bool megamol::core::MegaMolGraph::delete_module(std::string const& name) {
    auto const it = find_module(name);

    std::unique_lock<std::shared_mutex> lock(graph_lock_);

    if (it == this->module_list_.end()) {
        return false;
    }

    // TODO remove connections and corresponding calls

    this->module_list_.erase(it);
	//alarm
    return true;
}

[[nodiscard]]
megamol::core::MegaMolGraph::CallList_t::iterator megamol::core::MegaMolGraph::find_call(std::string const& name) {
    std::shared_lock<std::shared_mutex> lock(graph_lock_);

    auto it = std::find_if(
        this->call_list_.begin(), this->call_list_.end(), [&name](megamol::core::MegaMolGraph::CallDescr_t const& el) { return el.second.className == name; }); // FIXME

    return it;
}

// FIXME: same as above
[[nodiscard]]
megamol::core::MegaMolGraph::CallList_t::const_iterator megamol::core::MegaMolGraph::find_call(std::string const& name) const {
    std::shared_lock<std::shared_mutex> lock(graph_lock_);

    auto it = std::find_if(
        this->call_list_.cbegin(), this->call_list_.cend(), [&name](megamol::core::MegaMolGraph::CallDescr_t const& el) { return el.second.className == name; }); // FIXME

    return it;
}

bool megamol::core::MegaMolGraph::delete_call(std::string const& name) {
    auto const it = find_call(name);

    std::unique_lock<std::shared_mutex> lock(graph_lock_);

    if (it == this->call_list_.end()) {
        return false;
    }

    // TODO remove connections
    auto source = it->first->PeekCallerSlotNoConst();
    source->SetCleanupMark(true);
    source->DisconnectCalls();

    this->call_list_.erase(it);
	//alarm
    return true;
}

void megamol::core::MegaMolGraph::ExecuteGraphUpdates() {

}

void megamol::core::MegaMolGraph::RenderNextFrame() {

}




// void megamol::core::MegaMolGraph::GraphRoot::executeGraphCommands() {
//	// commands may create/destruct modules and calls into the modules_ and calls_ lists of the GraphRoot
//	// construction/destruction of modules/calls, as well as creation/release need to happen inside the Render API
// context
//	// because for OpenGL the GL context needs to be 'active' when creating GL resources - this may happen during module
// creation or call constructors
//
//    for (auto& command: this->graphCommands_)
//		command(*this);
//
//    this->graphCommands_.clear();
//}
//
// void megamol::core::MegaMolGraph::GraphRoot::renderNextFrame() {
//    if (!this->rapi)
//		return;
//
//	this->rapi->preViewRender();
//
//	if (this->graphCommands_.size())
//		this->executeGraphCommands();
//
//	_mmcRenderViewContext dummyRenderViewContext; // doesn't do anything, really
//	//void* apiContextDataPtr = this->rapi->getContextDataPtr();
//    if (this->view_)
//		this->view_->Render(/* want: 'apiContextDataPtr' instead of: */ dummyRenderViewContext );
//
//	this->rapi->postViewRender();
//}

//[[nodiscard]] std::shared_ptr<megamol::core::param::AbstractParam> FindParameter(std::string const& name, bool quiet = false) const;

// static const std::tuple<const std::string, const std::string> splitParameterName(std::string const& name) {
//     const std::string delimiter = "::";
// 
//     const auto lastDeliPos = name.rfind(delimiter);
//     if (lastDeliPos == std::string::npos) return {"", ""};
// 
//     auto secondLastDeliPos = name.rfind(delimiter, lastDeliPos - 1);
//     if (secondLastDeliPos == std::string::npos)
//         secondLastDeliPos = 0;
//     else
//         secondLastDeliPos = secondLastDeliPos + 2;
// 
//     const std::string parameterName = name.substr(lastDeliPos + 2);
//     const auto moduleNameStartPos = secondLastDeliPos;
//     const std::string moduleName = name.substr(moduleNameStartPos, /*cont*/ lastDeliPos - moduleNameStartPos);
// 
//     return {moduleName, parameterName};
// };
// 
// std::shared_ptr<megamol::core::param::AbstractParam> megamol::core::MegaMolGraph::FindParameter(
//     std::string const& name, bool quiet) const {
//     using vislib::sys::Log;
// 
//     // what exactly is the format of name? assumption: [::]moduleName::parameterName
//     // split name into moduleName and parameterName:
//     auto [moduleName, parameterName] = splitParameterName(name);
// 
//     // parameter or module does not exist
//     if (moduleName.compare("") == 0 || parameterName.compare("") == 0) {
//         if (!quiet)
//             Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
//                 "Cannot find parameter \"%s\": could not extract module and parameter name", name.c_str());
//         return nullptr;
//     }
// 
//     param::ParamSlot* slot = nullptr;
//     Module::ptr_type modPtr = nullptr;
// 
//     for (auto& graphRoot : this->subgraphs_) {
//         for (auto& mod : graphRoot.modules_)
//             if (mod.first->Name().Compare(moduleName.c_str())) {
//                 modPtr = mod.first;
// 
//                 const auto result = mod.first->FindChild((moduleName + "::" + parameterName).c_str());
//                 slot = dynamic_cast<param::ParamSlot*>(result.get());
// 
//                 break;
//             }
//     }
// 
//     if (!modPtr) {
//         if (!quiet)
//             Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Cannot find parameter \"%s\": module not found", name.c_str());
//         return nullptr;
//     }
// 
//     if (slot == nullptr) {
//         if (!quiet)
//             Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Cannot find parameter \"%s\": slot not found", name.c_str());
//         return nullptr;
//     }
//     if (slot->GetStatus() == AbstractSlot::STATUS_UNAVAILABLE) {
//         if (!quiet)
//             Log::DefaultLog.WriteMsg(
//                 Log::LEVEL_ERROR, "Cannot find parameter \"%s\": slot is not available", name.c_str());
//         return nullptr;
//     }
//     if (slot->Parameter().IsNull()) {
//         if (!quiet)
//             Log::DefaultLog.WriteMsg(
//                 Log::LEVEL_ERROR, "Cannot find parameter \"%s\": slot has no parameter", name.c_str());
//         return nullptr;
//     }
// 
// 
//     return slot->Parameter();
// }
