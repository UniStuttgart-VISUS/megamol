#pragma once

#include <functional>
#include <list>
#include <string>
#include <vector>

#include "mmcore/factories/CallDescriptionManager.h"
#include "mmcore/factories/ModuleDescription.h"
#include "mmcore/factories/ModuleDescriptionManager.h"
#include "mmcore/param/AbstractParam.h"
#include "mmcore/param/ParamSlot.h"

#include "mmcore/deferrable_construction.h"
#include "mmcore/serializable.h"

#include "mmcore/MegaMolGraphTypes.h"
#include "mmcore/MegaMolGraph_Convenience.h"

#include "mmcore/RootModuleNamespace.h"

#include "FrontendResource.h"
#include "ImagePresentationEntryPoints.h"

namespace megamol {
namespace core {

class MEGAMOLCORE_API MegaMolGraph { //: public serializable, public deferrable_construction {

    // todo: where do the descriptionmanagers go?
    // todo: what about the view / job descriptions?
public:
    // todo: constructor(s)? see serialization
    // todo: probably get rid of RootModuleNamespace altogether

    ///////////////////////////// types ////////////////////////////////////////



    //////////////////////////// ctor / dtor ///////////////////////////////

    /**
     * Bare construction as stub for deserialization
     */
    MegaMolGraph(megamol::core::CoreInstance& core, factories::ModuleDescriptionManager const& moduleProvider,
        factories::CallDescriptionManager const& callProvider);

    /**
     * No copy-construction. This can only be a legal operation, if we allow deep-copy of Modules in graph.
     */
    MegaMolGraph(MegaMolGraph const& rhs) = delete;

    /**
     * Same argument as for copy-construction.
     */
    MegaMolGraph& operator=(MegaMolGraph const& rhs) = delete;

    /**
     * A move of the graph should be OK, even without changing state of Modules in graph.
     */
    MegaMolGraph(MegaMolGraph&& rhs) noexcept;

    /**
     * Same is true for move-assignment.
     */
    MegaMolGraph& operator=(MegaMolGraph&& rhs) noexcept;

    /**
     * Construction from serialized string.
     */
    // MegaMolGraph(std::string const& descr);

    /** dtor */
    virtual ~MegaMolGraph();

    //////////////////////////// END ctor / dtor ///////////////////////////////


    //////////////////////////// Satisfy some abstract requirements ///////////////////////////////


    // TODO: the 'serializable' and 'deferrable construction' concepts result in basically the same implementation?
    // serializable

    // deferrable_construction

    //////////////////////////// Modules and Calls loaded from DLLs ///////////////////////////////

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

    megamol::core::param::ParamSlot* FindParameterSlot(std::string const& paramName) const;

    std::vector<megamol::core::param::AbstractParam*> EnumerateModuleParameters(std::string const& moduleName) const;

    std::vector<megamol::core::param::ParamSlot*> EnumerateModuleParameterSlots(std::string const& moduleName) const;

    CallList_t const& ListCalls() const;

    ModuleList_t const& ListModules() const;

    std::vector<megamol::core::param::AbstractParam*> ListParameters() const;

    std::vector<megamol::core::param::ParamSlot*> ListParameterSlots() const;

    bool SetGraphEntryPoint(std::string moduleName);

    bool RemoveGraphEntryPoint(std::string moduleName);

    void RenderNextFrame();

    bool AddFrontendResources(std::vector<megamol::frontend::FrontendResource> const& resources);

    // shut down all calls, modules, graph entry points
    void Clear();

    MegaMolGraph_Convenience& Convenience();

    // Create View ?

    // Create Chain Call ?

    // int ListInstatiations(lua_State* L);

private:
    // get invalidated and the user is helpless
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

    std::vector<megamol::frontend::FrontendResource> get_requested_resources(std::vector<std::string> resource_requests);


    // the dummy_namespace must be above the call_list_ and module_list_ because it needs to be destroyed AFTER all
    // calls and modules during ~MegaMolGraph()
    std::shared_ptr<RootModuleNamespace> dummy_namespace; // serves as parent object for stupid fat modules

    /** List of modules that this graph owns */
    ModuleList_t module_list_;

    /** List of call that this graph owns */
    CallList_t call_list_;

    std::vector<megamol::frontend::FrontendResource> provided_resources;

    // for each View in the MegaMol graph we create a GraphEntryPoint
    // that entry point is used by the Image Presentation Service to
    // poke the rendering, collect the resulting View renderings and present them to the user appropriately
    std::list<Module::ptr_type> graph_entry_points;
    megamol::frontend_resources::ImagePresentationEntryPoints* m_image_presentation = nullptr;

    MegaMolGraph_Convenience convenience_functions;

    ////////////////////////// old interface stuff //////////////////////////////////////////////
public:
    // TODO: pull necessary 'old' functions to active section above
    /*
        bool QueueChainCallInstantiation(const std::string className, const std::string chainStart, const
    std::string to); bool QueueParamValueChange(const std::string id, const std::string value);
        // a JSON serialization of all the requests as above (see updateListener)
        bool QueueGraphChange(const std::string changes);
        /// @return group id 0 is invalid and means failure
        uint32_t CreateParamGroup(const std::string name, int size);
        bool QueueParamGroupValue(const std::string groupName, const std::string id, const std::string value);
        bool QueueParamGroupValue(uint32_t groupId, const std::string id, const std::string value);
        bool QueueUpdateFlush();
        bool AnythingQueued();

        // todo: for everything below, RO version AND RW version? or would we just omit the const and imply the user
    needs
        // to lock?
        ////////////////////////////

        // vislib::SmartPtr<param::AbstractParam> FindParameter(const std::string name, bool quiet = false) const;

        // todo: optionally ask for the parameters of a specific module (name OR module pointer?)
        inline void EnumerateParameters(std::function<void(const Module&, param::ParamSlot&)> cb) const;

        // todo: optionally pass Module instead of name
        template <class A, class C>
        typename std::enable_if<std::is_convertible<A*, Module*>::value && std::is_convertible<C*, Call*>::value,
            bool>::type
        EnumerateCallerSlots(std::string module_name, std::function<void(C&)> cb) const;

        // WHY??? this is just EnumerateParameters(FindModule()...) GET RID OF IT!
        template <class A>
        typename std::enable_if<std::is_convertible<A*, Module*>::value, bool>::type EnumerateParameterSlots(
            std::string module_name, std::function<void(param::ParamSlot&)> cb) const;

        size_t GetGlobalParameterHash(void) const;

        // probably just throws everything away?
        void Cleanup(void);

        // serialize into... JSON? WHY THE FUCK IS THIS IN THE ROOTMODULENAMESPACE!
        std::string SerializeGraph(void) const;

        // replace the whole graph with whatever is in serialization
        void Deserialize(std::string othergraph);

        // nope, see below!
        // void NotifyParameterValueChange(param::ParamSlot& slot) const;
        // void RegisterParamUpdateListener(param::ParamUpdateListener* pul);
        // void UnregisterParamUpdateListener(param::ParamUpdateListener* pul);

        // accumulate the stuff the queues ask from the graph and give out a JSON diff right afterwards
        // bitfield says what (params, modules, whatnot) - and also reports whether something did not happen
        /// @return some ID to allow for removal of the listener later
        uint32_t RegisterGraphUpdateListener(
            std::function<void(std::string, uint32_t field)> func, int32_t serviceBitfield);
        void UnregisterGraphUpdateListener(uint32_t id);

    private:
        // todo: signature is weird, data structure might be as well
        void computeGlobalParameterHash(ModuleNamespace::const_ptr_type path, ParamHashMap_t& map) const;
        static void compareParameterHash(ParamHashMap_t& one, ParamHashMap_t& other) const;

        void updateFlushIdxList(size_t const processedCount, std::vector<size_t>& list);
        bool checkForFlushEvent(size_t const eventIdx, std::vector<size_t>& list) const;
        void shortenFlushIdxList(size_t const eventCount, std::vector<size_t>& list);
    */

    //    [[nodiscard]] std::shared_ptr<param::AbstractParam> FindParameter(
    //        std::string const& name, bool quiet = false) const;
};


} /* namespace core */
} // namespace megamol
