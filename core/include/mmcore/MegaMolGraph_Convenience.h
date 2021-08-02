#pragma once

#include <unordered_map>
#include <functional>

#include "MegaMolGraphTypes.h"

namespace megamol {
namespace core {

class MEGAMOLCORE_API MegaMolGraph_Convenience {
public:

    // parameter groups hold requests for new parameter values
    // values are applied to the graph upon request at once
    struct ParameterGroup {
        std::string name;
        std::unordered_map<std::string, std::string> parameter_values;
        void* graph; // graph to apply queued values to

        bool QueueParameterValue(const std::string& id, const std::string& value);
        bool ApplyQueuedParameterValues();
    };

    MegaMolGraph_Convenience(void* graph_ptr = nullptr);

    std::string SerializeModules() const;
    std::string SerializeCalls() const;
    std::string SerializeModuleParameters(std::string const& module_name) const;
    std::string SerializeAllParameters() const;
    std::string SerializeGraph() const;

    ParameterGroup& CreateParameterGroup(const std::string& group_name);
    ParameterGroup* FindParameterGroup(const std::string& group_name);
    std::vector<std::reference_wrapper<ParameterGroup>> ListParameterGroups();

    bool CreateChainCall(const std::string callName, const std::string from_slot_name, const std::string to_slot_name);
    ModuleList_t ListModules(const std::string startModuleName);
    ModuleList_t ListModules(const Module::ptr_type startModule);
    void TraverseGraph(const std::string startModuleName, std::function<void(Module::ptr_type)> cb, const std::string allowedCallType = "");
    void TraverseGraph(const Module::ptr_type startModule, std::function<void(Module::ptr_type)> cb, const std::string allowedCallType = "");
    // enumerate incoming/compatible calls on module
    // enumerate outgoing/compatible calls on module
    // enumerate empty/compatible callee slots on module
    // enumerate empty/compatible caller slots on module

private:
    void* m_graph_ptr = nullptr;

    std::unordered_map<std::string, ParameterGroup> m_parameter_groups;
};


} /* namespace core */
} // namespace megamol

