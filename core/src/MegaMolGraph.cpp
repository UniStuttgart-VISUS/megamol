#include "stdafx.h"
#include "mmcore/MegaMolGraph.h"


bool megamol::core::MegaMolGraph::QueueModuleDeletion(std::string const& id) {
    auto lock = module_deletion_queue_.AcquireLock();
    return QueueModuleDeletion(id);
}


bool megamol::core::MegaMolGraph::QueueModuleDeletion(std::string&& id) {
    auto lock = module_deletion_queue_.AcquireLock();
    return QueueModuleDeletionNoLock(std::move(id));
}


bool megamol::core::MegaMolGraph::QueueModuleDeletionNoLock(std::string const& id) {
    return push_queue_element(module_deletion_queue_, id);
}


bool megamol::core::MegaMolGraph::QueueModuleDeletionNoLock(std::string&& id) {
    return push_queue_element(module_deletion_queue_, id);
}


bool megamol::core::MegaMolGraph::QueueModuleInstantiation(std::string const& className, std::string const& id) {
    auto lock = module_instantiation_queue_.AcquireLock();
    return QueueModuleInstantiationNoLock(className, id);
}


bool megamol::core::MegaMolGraph::QueueModuleInstantiation(std::string&& className, std::string&& id) {
    auto lock = module_instantiation_queue_.AcquireLock();
    return QueueModuleInstantiationNoLock(std::move(className), std::move(id));
}


bool megamol::core::MegaMolGraph::QueueModuleInstantiationNoLock(std::string const& className, std::string const& id) {
    return emplace_queue_element(module_instantiation_queue_, className, id);
}


bool megamol::core::MegaMolGraph::QueueModuleInstantiationNoLock(std::string&& className, std::string&& id) {
    return emplace_queue_element(module_instantiation_queue_, std::move(className), std::move(id));
}


bool megamol::core::MegaMolGraph::QueueCallDeletion(std::string const& from, std::string const& to) {
    auto lock = call_deletion_queue_.AcquireLock();
    return QueueCallDeletionNoLock(from, to);
}


bool megamol::core::MegaMolGraph::QueueCallDeletion(std::string&& from, std::string&& to) {
    auto lock = call_deletion_queue_.AcquireLock();
    return QueueCallDeletionNoLock(std::move(from), std::move(to));
}


bool megamol::core::MegaMolGraph::QueueCallDeletionNoLock(std::string const& from, std::string const& to) {
    return emplace_queue_element(call_deletion_queue_, from, to);
}


bool megamol::core::MegaMolGraph::QueueCallDeletionNoLock(std::string&& from, std::string&& to) {
    return emplace_queue_element(call_deletion_queue_, std::move(from), std::move(to));
}


bool megamol::core::MegaMolGraph::QueueCallInstantiation(
    std::string const& className, std::string const& from, std::string const& to) {
    auto lock = call_instantiation_queue_.AcquireLock();
    return QueueCallInstantiationNoLock(className, from, to);
}


bool megamol::core::MegaMolGraph::QueueCallInstantiation(
    std::string&& className, std::string&& from, std::string&& to) {
    auto lock = call_instantiation_queue_.AcquireLock();
    return QueueCallInstantiationNoLock(std::move(className), std::move(from), std::move(to));
}


bool megamol::core::MegaMolGraph::QueueCallInstantiationNoLock(
    std::string const& className, std::string const& from, std::string const& to) {
    return emplace_queue_element(call_instantiation_queue_, className, from, to);
}


bool megamol::core::MegaMolGraph::QueueCallInstantiationNoLock(
    std::string&& className, std::string&& from, std::string&& to) {
    return emplace_queue_element(call_instantiation_queue_, std::move(className), std::move(from), std::move(to));
}
