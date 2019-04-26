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


std::shared_ptr<megamol::core::param::AbstractParam> megamol::core::MegaMolGraph::FindParameter(
    std::string const& name, bool quiet) const {
    using vislib::sys::Log;

    // TODO do we really need to lock the graph for this operation
    AbstractNamedObject::GraphLocker locker(this->root_module_namespace_, false);
    vislib::sys::AutoLock lock(locker);

    vislib::Array<vislib::StringA> path = vislib::StringTokeniserA::Split(name, "::", true);
    vislib::StringA slotName("");
    if (path.Count() > 0) {
        slotName = path.Last();
        path.RemoveLast();
    }
    vislib::StringA modName("");
    if (path.Count() > 0) {
        modName = path.Last();
        path.RemoveLast();
    }

    ModuleNamespace::ptr_type mn;
    // parameter slots may have namespace operators in their names!
    while (!mn) {
        mn = this->root_module_namespace_->FindNamespace(path, false, true);
        if (!mn) {
            if (path.Count() > 0) {
                slotName = modName + "::" + slotName;
                modName = path.Last();
                path.RemoveLast();
            } else {
                if (!quiet)
                    Log::DefaultLog.WriteMsg(
                        Log::LEVEL_ERROR, "Cannot find parameter \"%s\": namespace not found", name.c_str());
                return nullptr;
            }
        }
    }

    Module::ptr_type mod = Module::dynamic_pointer_cast(mn->FindChild(modName));
    if (!mod) {
        if (!quiet)
            Log::DefaultLog.WriteMsg(
                Log::LEVEL_ERROR, "Cannot find parameter \"%s\": module not found", name.c_str());
        return nullptr;
    }

    param::ParamSlot* slot = dynamic_cast<param::ParamSlot*>(mod->FindChild(slotName).get());
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
