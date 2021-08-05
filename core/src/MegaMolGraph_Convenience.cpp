#include <sstream>
#include "mmcore/MegaMolGraph_Convenience.h"
#include "mmcore/MegaMolGraph.h"

#include "mmcore/utility/log/Log.h"

using namespace megamol::core;

static MegaMolGraph& get(void* ptr) { return *reinterpret_cast<MegaMolGraph*>(ptr); }

static void log(std::string text) {
    const std::string msg = "MegaMolGraph_Convenience: " + text + "\n";
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(msg.c_str());
}

static void err(std::string text) {
    const std::string msg = "MegaMolGraph_Convenience: " + text + "\n";
    megamol::core::utility::log::Log::DefaultLog.WriteError(msg.c_str());
}

MegaMolGraph_Convenience::MegaMolGraph_Convenience(void* graph_ptr) : m_graph_ptr{graph_ptr} {}

std::string megamol::core::MegaMolGraph_Convenience::SerializeModules() const {
    std::string serViews;
    std::string serModules;

    for (auto& module : get(m_graph_ptr).ListModules()) {
        if (module.isGraphEntryPoint) {
            auto first = module.request.id.find_first_not_of(':');
            auto last = module.request.id.find_first_of(':', first);
            auto first_name = module.request.id.substr(first, last-first);
            auto improvised_instance_name = "::"+first_name;
            serViews.append("mmCreateView(\"" + improvised_instance_name + "\",\"" + module.request.className + "\",\"" + module.request.id + "\")\n");
        }
    }

    for (auto& module : get(m_graph_ptr).ListModules()) {
        if (!module.isGraphEntryPoint) {
            serModules.append("mmCreateModule(\"" + module.request.className + "\",\"" + module.request.id + "\")\n");
        }
    }

    return serViews + '\n' + serModules + '\n';
}

std::string megamol::core::MegaMolGraph_Convenience::SerializeCalls() const {
    std::string serCalls;

    for (auto& call : get(m_graph_ptr).ListCalls()) {
        serCalls.append("mmCreateCall("
           "\"" + call.request.className + "\","
            "\"" + call.request.from + "\","
            "\"" + call.request.to + "\""
            + ")\n");
    }

    return serCalls + '\n';
}

#include "mmcore/param/ButtonParam.h"
std::string megamol::core::MegaMolGraph_Convenience::SerializeModuleParameters(std::string const& module_name) const {
    std::string serParams;

    for (auto& paramSlot : get(m_graph_ptr).EnumerateModuleParameterSlots(module_name)) {
        // it seems serialiing button params is illegal
        if (auto* p_ptr = paramSlot->template Param<core::param::ButtonParam>()) {
            continue;
        }
        auto name = std::string{paramSlot->FullName()};
        // as FullName() prepends :: to module names, normalize multiple leading :: in parameter name path
        name = "::" + name.substr(name.find_first_not_of(':'));
        auto value = std::string{paramSlot->Parameter()->ValueString().PeekBuffer()};
        serParams.append("mmSetParamValue(\"" + name + "\",[=[" + value + "]=])\n");
    }

    return serParams + '\n';
}

std::string megamol::core::MegaMolGraph_Convenience::SerializeAllParameters() const {
    std::string serAllParams;

    for (auto& module : get(m_graph_ptr).ListModules()) {
        serAllParams += SerializeModuleParameters(module.request.id);
    }

    return serAllParams + '\n';
}

std::string megamol::core::MegaMolGraph_Convenience::SerializeGraph() const {
    std::string serialization;

    serialization += SerializeModules();
    serialization += SerializeCalls();
    serialization += SerializeAllParameters();

    return serialization;
}

MegaMolGraph_Convenience::ParameterGroup& MegaMolGraph_Convenience::CreateParameterGroup(
    const std::string& group_name) {
    m_parameter_groups[group_name] = {group_name, {}, this->m_graph_ptr};

    return m_parameter_groups.at(group_name);
}

MegaMolGraph_Convenience::ParameterGroup* MegaMolGraph_Convenience::FindParameterGroup(const std::string& group_name) {
    auto find_it = m_parameter_groups.find(group_name);

    if (find_it != m_parameter_groups.end()) {
        return &find_it->second;
    }

    return nullptr;
}

std::vector<std::reference_wrapper<MegaMolGraph_Convenience::ParameterGroup>>
MegaMolGraph_Convenience::ListParameterGroups() {
    std::vector<std::reference_wrapper<ParameterGroup>> v;
    v.reserve(m_parameter_groups.size());

    for (auto& pg : m_parameter_groups) {
        v.push_back({pg.second});
    }

    return v;
}

bool MegaMolGraph_Convenience::ParameterGroup::QueueParameterValue(const std::string& id, const std::string& value) {
    auto parameterPtr = get(graph).FindParameter(id);

    if (parameterPtr) {
        this->parameter_values[id] = value;

        return true;
    }

    return false;
}

bool MegaMolGraph_Convenience::ParameterGroup::ApplyQueuedParameterValues() {
    bool result = true;
    for (auto& pv : parameter_values) {
        result &= get(graph).FindParameter(pv.first)->ParseValue(pv.second.c_str());
    }

    parameter_values.clear();
    return result;
}

bool MegaMolGraph_Convenience::CreateChainCall(
    const std::string callName, const std::string from_slot_name, const std::string to_slot_name) {
    auto className = callName;
    std::string chainStart = from_slot_name;
    std::string to = to_slot_name;

    const auto split_name = [&](std::string name) -> std::optional<std::tuple<std::string, std::string>> {
        auto pos = name.find_last_of("::");
        if (pos < 4 || name.length() < pos + 2) {
            err("chainStart module/slot name weird");
            return std::nullopt;
        }
        auto moduleName = name.substr(0, pos - 1);
        auto slotName = name.substr(pos + 1, -1);

        return std::make_optional<>(std::make_tuple<>(moduleName, slotName));
    };

    const auto throwError = [&](auto detail) {
        std::ostringstream out;
        out << "could not create \"";
        out << className;
        out << "\" call (" << detail << ")";
        err(out.str());
    };

    const auto get_module_slot = [&](std::tuple<std::string, std::string>& names_tuple) -> AbstractSlot* {
        auto moduleName = std::get<0>(names_tuple);
        auto slotName = std::get<1>(names_tuple);

        auto modulePtr = get(m_graph_ptr).FindModule(moduleName);
        if (!modulePtr) {
            throwError("no module named " + moduleName);
            return nullptr;
        }

        auto slotPtr = modulePtr->FindSlot(slotName.c_str());
        if (!slotPtr) {
            throwError("module has no slot named " + slotName);
            return nullptr;
        }

        return slotPtr;
    };

    auto to_names = split_name(to);
    if (!to_names.has_value()) {
        return 0;
    }
    auto to_slot = get_module_slot(to_names.value());
    if (!to_slot) {
        throwError("to slot " + to + " not found");
        return 0;
    }
    if (to_slot->GetStatus() == AbstractSlot::SlotStatus::STATUS_UNAVAILABLE) {
        throwError("to slot " + to + " UNAVAILABLE");
        return 0;
    }

    CalleeSlot* toSlot = dynamic_cast<CalleeSlot*>(to_slot);
    if (!toSlot) {
        throwError("to slot " + to + " is not CalleeSlot");
        return 0;
    }

    auto from_names = split_name(chainStart);
    if (!from_names.has_value()) {
        return 0;
    }

    // implementation with the new traversers
    // TODO
    //const auto doStuff = [&](Module::ptr_type mod) {
    //    // now enumerate the slots
    //    // TODO
    //};
    //TraverseGraph(std::get<0>(from_names.value()), doStuff, callName);

    auto from_slot = get_module_slot(from_names.value());
    CallerSlot* from_caller_slot = dynamic_cast<CallerSlot*>(from_slot);

    // walk along chain calls until one is available and not connected
    while (from_caller_slot && from_caller_slot->GetStatus() != AbstractSlot::SlotStatus::STATUS_UNAVAILABLE &&
           from_caller_slot->GetStatus() == AbstractSlot::SlotStatus::STATUS_CONNECTED) {
        Call* call = from_caller_slot->CallAs<Call>();
        if (!call) {
            throwError("call at " + std::string(from_caller_slot->FullName().PeekBuffer()) + " returned nullptr");
            return 0;
        }

        CalleeSlot* calleeSlot = call->PeekCalleeSlotNoConst();
        if (!calleeSlot) {
            throwError("callee slot of call " + std::string(call->ClassName()) + " at " +
                       std::string(from_caller_slot->FullName().PeekBuffer()) + " returned nullptr");
            return 0;
        }

        // get callee module
        std::string calleeSlotName{calleeSlot->FullName().PeekBuffer()};
        auto pos_it = calleeSlotName.find_last_of("::");
        std::string calleeModuleName = calleeSlotName.substr(0, pos_it - 1);
        auto callee_module = get(m_graph_ptr).FindModule(calleeModuleName);

        if (!callee_module) {
            throwError("module " + calleeModuleName + " not found via callee slot " + calleeSlotName);
            return 0;
        }

        // find compatible outgoing chain-call slot among all outgoing slots
        auto children_begin_it = callee_module->ChildList_Begin();
        auto children_end_it = callee_module->ChildList_End();
        for (auto child_it = children_begin_it; child_it != children_end_it; child_it++) {
            AbstractNamedObject::ptr_type& named_object_ptr = *child_it;
            CallerSlot* caller_slot = dynamic_cast<CallerSlot*>(named_object_ptr.get());

            if (caller_slot && caller_slot->IsCallCompatible(className)) {
                from_caller_slot = caller_slot;
            }
        }
    }
    if (!from_caller_slot) {
        throwError("no starting slot found for chain call " + std::string(className) + ". search returned nullptr.");
        return 0;
    }
    if (from_caller_slot->GetStatus() == AbstractSlot::SlotStatus::STATUS_UNAVAILABLE) {
        throwError("chain call from slot " + std::string{from_caller_slot->FullName().PeekBuffer()} + " UNAVAILABLE");
        return 0;
    }
    if (from_caller_slot->GetStatus() == AbstractSlot::SlotStatus::STATUS_CONNECTED) {
        throwError("chain call from slot " + std::string{from_caller_slot->FullName().PeekBuffer()} + " CONNECTED");
        return 0;
    }
    // this should never trigger
    if (from_caller_slot->GetStatus() != AbstractSlot::SlotStatus::STATUS_ENABLED) {
        throwError("chain call from slot " + std::string{from_caller_slot->FullName().PeekBuffer()} + " not ENABLED");
        return 0;
    }

    std::string from = from_caller_slot->FullName().PeekBuffer();

    bool call_ok = get(m_graph_ptr).CreateCall(className, from, to);

    if (!call_ok) {
        throwError("from " + from + " to " + to);
    }

    return 0;
}

ModuleList_t MegaMolGraph_Convenience::ListModules(const std::string startModuleName) {
    ModuleList_t ret;
    auto modulePtr = get(m_graph_ptr).FindModule(startModuleName);
    if (!modulePtr) {
        err("Could not traverse graph from \"" + startModuleName + "\": module not found");
    } else {
        ret = ListModules(modulePtr);
    }
    return ret;
}

ModuleList_t MegaMolGraph_Convenience::ListModules(const Module::ptr_type startModule) {
    ModuleList_t ret;
    const auto fun = [&ret, this](Module::ptr_type mod) {
        // TODO everything using moduleinstance or the module ptr ;_;
        for (auto& mi : get(this->m_graph_ptr).ListModules()) {
            if (mi.modulePtr == mod) {
                ret.push_back(mi);
            }
        }
    };
    TraverseGraph(startModule, fun);
    return ret;
}

void MegaMolGraph_Convenience::TraverseGraph(const std::string startModuleName,
    std::function<void(Module::ptr_type)> cb, const std::string allowedCallType) {
    auto modulePtr = get(m_graph_ptr).FindModule(startModuleName);
    if (!modulePtr) {
        err("Could not traverse graph from \"" + startModuleName + "\": module not found");
    } else {
        TraverseGraph(modulePtr, cb, allowedCallType);
    }
}

void MegaMolGraph_Convenience::TraverseGraph(const Module::ptr_type startModule,
    std::function<void(Module::ptr_type)> cb, const std::string allowedCallType) {

    const auto throwError = [&](auto detail) {
        std::ostringstream out;
        out << "could not traverse graph from \"";
        out << startModule->Name();
        out << "\": " << detail << "";
        err(out.str());
    };

    std::vector<Module::ptr_type> moduleStack;
    std::map<Module::ptr_type, bool> visitedFlag;
    for (auto& m : get(m_graph_ptr).ListModules()) {
        visitedFlag[m.modulePtr] = false;
    }
    if (startModule != nullptr) {
        moduleStack.push_back(startModule);
    } else {
        // add all entrypoints instead, i.e. search everywhere
        for (auto& m : get(m_graph_ptr).ListModules()) {
            if (m.isGraphEntryPoint) {
                moduleStack.push_back(m.modulePtr);
            }
        }
    }

    while (!moduleStack.empty()) {
        auto& mod = moduleStack.back();
        moduleStack.pop_back();

        // a module might be queued multiple times, so re-checking is required all the time
        if (!visitedFlag[mod]) {
            visitedFlag[mod] = true;
            cb(mod);
            const auto it_end = mod->ChildList_End();
            for (auto it = mod->ChildList_Begin(); it != it_end; ++it) {
                auto* cs = dynamic_cast<CallerSlot*>((*it).get());
                if (cs) {
                    auto* c = cs->CallAs<Call>();
                    if (c) {
                        bool ok = allowedCallType.empty() || allowedCallType == c->ClassName();
                        if (ok) {
                            auto mod2 =
                                get(m_graph_ptr).FindModule(c->PeekCalleeSlot()->Parent()->FullName().PeekBuffer());
                            if (mod2 && !visitedFlag[mod2]) {
                                moduleStack.push_back(mod2);
                            }
                        }
                    }
                }
            }
        }
    }
}
