#pragma once

#include <functional>
#include <future>
#include <list>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>
#include "mmcore/Call.h"
#include "mmcore/Module.h"
#include "mmcore/api/MegaMolCore.h"
#include "mmcore/factories/CallDescription.h"
#include "mmcore/factories/CallDescriptionManager.h"
#include "mmcore/factories/ModuleDescription.h"
#include "mmcore/factories/ModuleDescriptionManager.h"
#include "mmcore/param/AbstractParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/ParamUpdateListener.h"
#include "vislib/sys/Log.h"

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/deferrable_construction.h"
#include "mmcore/serializable.h"

#include "AbstractRenderAPI.hpp"
#include "mmcore/RootModuleNamespace.h"

#include "mmcore/view/AbstractView.h"
#include "RenderResource.h"

namespace megamol {
namespace core {

class MEGAMOLCORE_API MegaMolGraph : public serializable, public deferrable_construction {

    // todo: where do the descriptionmanagers go?
    // todo: what about the view / job descriptions?
public:
    // todo: constructor(s)? see serialization
    // todo: probably get rid of RootModuleNamespace altogether

    ///////////////////////////// types ////////////////////////////////////////
    using ModuleDeletionRequest_t = std::string;

    struct ModuleInstantiationRequest {
        std::string className;
        std::string id;
    };

    using ModuleInstantiationRequest_t = ModuleInstantiationRequest;

    struct CallDeletionRequest {
        std::string from;
        std::string to;
    };

    using CallDeletionRequest_t = CallDeletionRequest;

    struct CallInstantiationRequest {
        std::string className;
        std::string from;
        std::string to;
    };

    using CallInstantiationRequest_t = CallInstantiationRequest;

	struct ModuleInstance_t {
        Module::ptr_type modulePtr = nullptr;
        ModuleInstantiationRequest request;
        bool isGraphEntryPoint = false;
        std::vector<std::string> lifetime_dependencies_requests;
        std::vector<megamol::render_api::RenderResource> lifetime_dependencies;
	};

    using ModuleList_t = std::list<ModuleInstance_t>;

    using CallInstance_t = std::pair<Call::ptr_type, CallInstantiationRequest>;

    using CallList_t = std::list<CallInstance_t>;

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
    std::string Serialize() const override { return ""; };
    void Deserialize(std::string const& descr) override{};

	// deferrable_construction 
    bool create() override { return false; };
    void release() override{};

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

	using EntryPointExecutionCallback =
        std::function<void(Module::ptr_type, std::vector<megamol::render_api::RenderResource>)>;

	bool SetGraphEntryPoint(std::string moduleName, std::vector<std::string> execution_dependencies, EntryPointExecutionCallback callback);

	bool RemoveGraphEntryPoint(std::string moduleName);

    void RenderNextFrame();

	void AddModuleDependencies(std::vector<megamol::render_api::RenderResource> const& dependencies);

	// Create View ?

	// Create Chain Call ?

    //int ListInstatiations(lua_State* L);

private:
    // get invalidated and the user is helpless
    [[nodiscard]] ModuleList_t::iterator find_module(std::string const& name);

    [[nodiscard]] ModuleList_t::const_iterator find_module(std::string const& name) const;

    [[nodiscard]] CallList_t::iterator find_call(std::string const& from, std::string const& to);

    [[nodiscard]] CallList_t::const_iterator find_call(std::string const& from, std::string const& to) const;

    [[nodiscard]] bool add_module(ModuleInstantiationRequest_t const& request);

    [[nodiscard]] bool add_call(CallInstantiationRequest_t const& request);

    bool delete_module(ModuleDeletionRequest_t const& request);

    bool delete_call(CallDeletionRequest_t const& request);

    std::vector<megamol::render_api::RenderResource> get_requested_dependencies(std::vector<std::string> dependency_requests);


    /** List of modules that this graph owns */
    ModuleList_t module_list_;

    // the dummy_namespace must be above the call_list_ because it needs to be destroyed AFTER all calls during
    // ~MegaMolGraph()
    std::shared_ptr<RootModuleNamespace> dummy_namespace; // serves as parent object for stupid fat modules

    /** List of call that this graph owns */
    CallList_t call_list_;

	std::vector<megamol::render_api::RenderResource> provided_dependencies;

	// for each View in the MegaMol graph we create a GraphEntryPoint with corresponding callback for resource/input consumption
	// the graph makes sure that the (lifetime and rendering) dependencies requested by the module are satisfied,
	// which means that the execute() callback for the entry point is provided the requested dependencies/resources for rendering
	// and the Create() and Release() mehods of all modules receive the dependencies/resources they request for their lifetime
	struct GraphEntryPoint {
        std::string moduleName;
        Module::ptr_type modulePtr = nullptr;
        std::vector<megamol::render_api::RenderResource> entry_point_dependencies;
		
		EntryPointExecutionCallback execute;
	};
    std::list<GraphEntryPoint> graph_entry_points;


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

