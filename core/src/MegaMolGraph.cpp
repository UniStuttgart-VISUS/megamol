#include "stdafx.h"
#include "mmcore/MegaMolGraph.h"


bool megamol::core::MegaMolGraph::QueueModuleDeletion(std::string const& id) {
    auto lock = module_deletion_queue_.AcquireLock();
    return QueueModuleDeletion(id);
}


bool megamol::core::MegaMolGraph::QueueModuleDeletionNoLock(std::string const& id) {
    return push_queue_element(module_deletion_queue_, id);
}


bool megamol::core::MegaMolGraph::QueueModuleInstantiation(std::string const& className, std::string const& id) {
    auto lock = module_instantiation_queue_.AcquireLock();
    return QueueModuleInstantiationNoLock(className, id);
}


bool megamol::core::MegaMolGraph::QueueModuleInstantiationNoLock(std::string const& className, std::string const& id) {
    return emplace_queue_element(module_instantiation_queue_, className, id);
}


bool megamol::core::MegaMolGraph::QueueCallDeletion(std::string const& from, std::string const& to) {
    auto lock = call_deletion_queue_.AcquireLock();
    return QueueCallDeletionNoLock(from, to);
}


bool megamol::core::MegaMolGraph::QueueCallDeletionNoLock(std::string const& from, std::string const& to) {
    return emplace_queue_element(call_deletion_queue_, from, to);
}


bool megamol::core::MegaMolGraph::QueueCallInstantiation(
    std::string const& className, std::string const& from, std::string const& to) {
    auto lock = call_instantiation_queue_.AcquireLock();
    return QueueCallInstantiationNoLock(className, from, to);
}


bool megamol::core::MegaMolGraph::QueueCallInstantiationNoLock(
    std::string const& className, std::string const& from, std::string const& to) {
    return emplace_queue_element(call_instantiation_queue_, className, from, to);
}

static
const std::tuple<const std::string, const std::string> splitParameterName(std::string const& name){
    const std::string delimiter = "::";

    const auto lastDeliPos = name.rfind(delimiter);
    if (lastDeliPos == std::string::npos)
		return {"", ""};

    auto secondLastDeliPos = name.rfind(delimiter, lastDeliPos-1);
    if (secondLastDeliPos == std::string::npos)
        secondLastDeliPos = 0;
    else
        secondLastDeliPos = secondLastDeliPos + 2;

    const std::string parameterName = name.substr(lastDeliPos + 2);
    const auto moduleNameStartPos = secondLastDeliPos;
    const std::string moduleName = name.substr(moduleNameStartPos, /*cont*/ lastDeliPos - moduleNameStartPos);

	return {moduleName, parameterName};
};

std::shared_ptr<megamol::core::param::AbstractParam> megamol::core::MegaMolGraph::FindParameter(
    std::string const& name, bool quiet) const {
    using vislib::sys::Log;

	// what exactly is the format of name? assumption: [::]moduleName::parameterName
	// split name into moduleName and parameterName:
	auto [moduleName, parameterName] = splitParameterName(name);

	// parameter or module does not exist
    if (moduleName.compare("") == 0 || parameterName.compare("") == 0) {
		 if (!quiet)
			Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Cannot find parameter \"%s\": could not extract module and parameter name", name.c_str());
		return nullptr;
	}

    param::ParamSlot* slot = nullptr;
    Module::ptr_type modPtr = nullptr;

	for (auto& graphRoot : this->subgraphs_) {
        for (auto& mod : graphRoot.modules_)
            if (mod.first->Name().Compare(moduleName.c_str())) {
                modPtr = mod.first;

                const auto result = mod.first->FindChild((moduleName + "::" + parameterName).c_str());
				slot = dynamic_cast<param::ParamSlot*>(result.get());

				break;
            }
	}

    if (!modPtr) {
        if (!quiet)
            Log::DefaultLog.WriteMsg(
                Log::LEVEL_ERROR, "Cannot find parameter \"%s\": module not found", name.c_str());
        return nullptr;
    }

    if (slot == nullptr) {
        if (!quiet)
            Log::DefaultLog.WriteMsg(
                Log::LEVEL_ERROR, "Cannot find parameter \"%s\": slot not found", name.c_str());
        return nullptr;
    }
    if (slot->GetStatus() == AbstractSlot::STATUS_UNAVAILABLE) {
        if (!quiet)
            Log::DefaultLog.WriteMsg(
                Log::LEVEL_ERROR, "Cannot find parameter \"%s\": slot is not available", name.c_str());
        return nullptr;
    }
    if (slot->Parameter().IsNull()) {
        if (!quiet)
            Log::DefaultLog.WriteMsg(
                Log::LEVEL_ERROR, "Cannot find parameter \"%s\": slot has no parameter", name.c_str());
        return nullptr;
    }


    return slot->Parameter();
}


void megamol::core::MegaMolGraph::GraphRoot::executeGraphCommands() {
	// commands may create/destruct modules and calls into the modules_ and calls_ lists of the GraphRoot
	// construction/destruction of modules/calls, as well as creation/release need to happen inside the Render API context
	// because for OpenGL the GL context needs to be 'active' when creating GL resources - this may happen during module creation or call constructors

    for (auto& command: this->graphCommands_)
		command(*this);

    this->graphCommands_.clear();
}

void megamol::core::MegaMolGraph::GraphRoot::renderNextFrame() {
    if (!this->rapi)
		return;

	this->rapi->preViewRender();

	if (this->graphCommands_.size())
		this->executeGraphCommands();

	_mmcRenderViewContext dummyRenderViewContext; // doesn't do anything, really
	//void* apiContextDataPtr = this->rapi->getContextDataPtr();
    if (this->view_)
		this->view_->Render(/* want: 'apiContextDataPtr' instead of: */ dummyRenderViewContext );

	this->rapi->postViewRender();
}

