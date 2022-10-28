#pragma once

#include <functional>
#include <list>
#include <string>
#include <vector>

#include "FrontendResource.h"
#include "FrontendResourcesLookup.h"
#include "ImagePresentationEntryPoints.h"
#include "ModuleGraphSubscription.h"
#include "mmcore/MegaMolGraphTypes.h"
#include "mmcore/MegaMolGraph_Convenience.h"
#include "mmcore/RootModuleNamespace.h"
#include "mmcore/factories/CallDescriptionManager.h"
#include "mmcore/factories/ModuleDescription.h"
#include "mmcore/factories/ModuleDescriptionManager.h"
#include "mmcore/param/AbstractParam.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace core {

class MegaMolGraph {
public:
    MegaMolGraph(factories::ModuleDescriptionManager const& moduleProvider,
        factories::CallDescriptionManager const& callProvider);

    virtual ~MegaMolGraph();

private:
    const factories::ModuleDescriptionManager& ModuleProvider();
    const factories::CallDescriptionManager& CallProvider();
    const factories::ModuleDescriptionManager* moduleProvider_ptr;
    const factories::CallDescriptionManager* callProvider_ptr;

public:
    /*
     * Each module should be serializable, i.e. the modules capture their entire state.
     * As a result, an entire MegaMolGraph can basically be copied by reinitializing the serialized descriptor.
     * Therefore the MegaMolGraph creates its descriptor by iterating through all modules and calls in the graph.
     *
     * Maybe the ModuleGraph should even allow external objects to iterate through a linearized array of containing
     * modules and calls.
     */

    bool DeleteModule(std::string const& id);

    bool CreateModule(std::string const& className, std::string const& id);

    bool RenameModule(std::string const& oldId, std::string const& newId);

    bool DeleteCall(std::string const& from, std::string const& to);

    bool CreateCall(std::string const& className, std::string const& from, std::string const& to);

    megamol::core::Module::ptr_type FindModule(std::string const& moduleName) const;

    megamol::core::Call::ptr_type FindCall(std::string const& from, std::string const& to) const;

    megamol::core::param::AbstractParam* FindParameter(std::string const& paramName) const;

    bool SetParameter(std::string const& paramName, std::string const& value);

    megamol::core::param::ParamSlot* FindParameterSlot(std::string const& paramName) const;

    std::vector<megamol::core::param::AbstractParam*> EnumerateModuleParameters(std::string const& moduleName) const;

    std::vector<megamol::core::param::ParamSlot*> EnumerateModuleParameterSlots(std::string const& moduleName) const;

    CallList_t const& ListCalls() const;

    ModuleList_t const& ListModules() const;

    std::vector<megamol::core::param::AbstractParam*> ListParameters() const;

    std::vector<megamol::core::param::ParamSlot*> ListParameterSlots() const;

    bool SetGraphEntryPoint(std::string moduleName);

    bool RemoveGraphEntryPoint(std::string moduleName);

    bool AddFrontendResources(std::vector<megamol::frontend::FrontendResource> const& resources);

    // shut down all calls, modules, graph entry points
    void Clear();

    MegaMolGraph_Convenience& Convenience();

    frontend_resources::MegaMolGraph_SubscriptionRegistry& GraphSubscribers();

    bool Broadcast_graph_subscribers_parameter_changes();

private:
    [[nodiscard]] ModuleList_t::iterator find_module(std::string const& name);
    [[nodiscard]] ModuleList_t::iterator find_module_by_prefix(std::string const& name);

    [[nodiscard]] ModuleList_t::const_iterator find_module(std::string const& name) const;
    [[nodiscard]] ModuleList_t::const_iterator find_module_by_prefix(std::string const& name) const;

    [[nodiscard]] CallList_t::iterator find_call(std::string const& from, std::string const& to);

    [[nodiscard]] CallList_t::const_iterator find_call(std::string const& from, std::string const& to) const;

    // modules are named using the exact same string that gets requested,
    // i.e. we dont split namespaces like ::Project_1::Group_1::View3D_2_1 to extract the 'actual' modul name 'View3D_2_1'
    [[nodiscard]] bool add_module(ModuleInstantiationRequest_t const& request);

    [[nodiscard]] bool add_call(CallInstantiationRequest_t const& request);

    bool delete_module(ModuleDeletionRequest_t const& request);

    bool delete_call(CallDeletionRequest_t const& request);


    // the dummy_namespace must be above the call_list_ and module_list_ because it needs to be destroyed AFTER all
    // calls and modules during ~MegaMolGraph()
    std::shared_ptr<RootModuleNamespace> dummy_namespace; // serves as parent object for stupid fat modules

    /** List of modules that this graph owns */
    ModuleList_t module_list_;

    /** List of call that this graph owns */
    CallList_t call_list_;

    megamol::frontend_resources::FrontendResourcesLookup provided_resources_lookup;

    // for each View in the MegaMol graph we create a EntryPoint
    // that entry point is used by the Image Presentation Service to
    // poke the rendering, collect the resulting View renderings and present them to the user appropriately
    std::list<Module::ptr_type> graph_entry_points;
    megamol::frontend_resources::ImagePresentationEntryPoints* m_image_presentation = nullptr;

    MegaMolGraph_Convenience convenience_functions;

    frontend_resources::MegaMolGraph_SubscriptionRegistry graph_subscribers;

    // module params may change their internal value on their own
    // the graph uses the AbstractParam::indicateChange() mechanism to inject
    // a callback that notifies the graph of param changes.
    // these parameter changes get collected in the following queue and are issued to graph subscribers when possible.
    std::vector<core::param::AbstractParamSlot*> module_param_changes_queue;
    core::param::AbstractParam::ParamChangeCallback param_change_callback = [&](core::param::AbstractParamSlot* slot) {
        module_param_changes_queue.push_back(slot);
        return true;
    };
};


} /* namespace core */
} // namespace megamol
