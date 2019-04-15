#include "stdafx.h"
#include "mmcore/MegaMolGraph.h"


bool megamol::core::MegaMolGraph::QueueModuleDeletion(std::string const& id) {
    auto lock = module_deletion_queue_.AcquireLock();
    module_deletion_queue_.Push(id);
    return true;
}


bool megamol::core::MegaMolGraph::QueueModuleDeletion(std::string&& id) {
    auto lock = module_deletion_queue_.AcquireLock();
    module_deletion_queue_.Push(std::move(id));
    return true;
}


bool megamol::core::MegaMolGraph::QueueModuleInstantiation(std::string const& className, std::string const& id) {
    auto lock = module_instantiation_queue_.AcquireLock();
    module_instantiation_queue_.Emplace(className, id);
    return true;
}


bool megamol::core::MegaMolGraph::QueueModuleInstantiation(std::string&& className, std::string&& id) {
    auto lock = module_instantiation_queue_.AcquireLock();
    module_instantiation_queue_.Emplace(std::move(className), std::move(id));
    return true;
}


bool megamol::core::MegaMolGraph::QueueCallDeletion(std::string const& from, std::string const& to) {
    auto lock = call_deletion_queue_.AcquireLock();
    call_deletion_queue_.Emplace(from, to);
    return true;
}


bool megamol::core::MegaMolGraph::QueueCallDeletion(std::string&& from, std::string&& to) {
    auto lock = call_deletion_queue_.AcquireLock();
    call_deletion_queue_.Emplace(from, to);
    return true;
}
