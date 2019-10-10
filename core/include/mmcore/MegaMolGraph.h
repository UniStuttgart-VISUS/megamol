#ifndef MEGAMOLCORE_MEGAMOLGRAPH_H_INCLUDED
#define MEGAMOLCORE_MEGAMOLGRAPH_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <functional>
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
#include "mmcore/lockable.h"
#include "mmcore/serializable.h"

#include "AbstractUpdateQueue.h" // TODO: why can't we use std::list? why is this class called abstract when, in fact, its implementation is very concrete?

//#include "AbstractRenderAPI.hpp"
#include "../../console/src/RenderAPI/AbstractRenderAPI.hpp" // temporary hack
#include "mmcore/view/AbstractView.h"

namespace megamol {
namespace core {

class MegaMolGraph : public serializable, public deferrable_construction {

    // todo: where do the descriptionmanagers go?
    // todo: what about the view / job descriptions?
public:
    // todo: constructor(s)? see serialization

    // todo: the lock!
    // todo: probably get rid of RootModuleNamespace altogether

    ///////////////////////////// types ////////////////////////////////////////
    using ModuleDeletionRequest_t = std::string;

    using ModuleDeletionQueue_t = AbstractUpdateQueue<ModuleDeletionRequest_t>;

    struct ModuleInstantiationRequest {
        std::string className;
        std::string id;
    };

    using ModuleInstantiationRequest_t = ModuleInstantiationRequest;

    using ModuleInstantiationQueue_t = AbstractUpdateQueue<ModuleInstantiationRequest_t>;

    struct CallDeletionRequest {
        std::string from;
        std::string to;
    };

    using CallDeletionRequest_t = CallDeletionRequest;

    using CallDeletionQueue_t = AbstractUpdateQueue<CallDeletionRequest_t>;

    struct CallInstantiationRequest {
        std::string className;
        std::string from;
        std::string to;
    };

    using CallInstantiationRequest_t = CallInstantiationRequest;

    using CallInstantiationQueue_t = AbstractUpdateQueue<CallInstantiationRequest_t>;

    using ModuleInstance_t = std::pair<Module::ptr_type, ModuleInstantiationRequest>;

    using ModuleList_t = std::list<ModuleInstance_t>;

    using CallInstance_t = std::pair<Call::ptr_type, CallInstantiationRequest>;

    using CallList_t = std::list<CallInstance_t>;


    //////////////////////////// ctor / dtor ///////////////////////////////

    /**
     * Bare construction as stub for deserialization
     */
    MegaMolGraph(factories::ModuleDescriptionManager const& moduleProvider,
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

private :
	const factories::ModuleDescriptionManager& ModuleProvider();
    const factories::CallDescriptionManager& CallProvider();
    const factories::ModuleDescriptionManager* moduleProvider_ptr;
    const factories::CallDescriptionManager* callProvider_ptr;

public:
    //////////////////////////// serialization ////////////////////////////////

    /*
     * Each module should be serializable, i.e. the modules capture their entire state.
     * As a result, an entire MegaMolGraph can basically be copied by reinitializing the serialized descriptor.
     * Therefore the MegaMolGraph creates its descriptor by iterating through all modules and calls in the graph.
     *
     * Maybe the ModuleGraph should even allow external objects to iterate through a linearized array of containing
     * modules and calls.
     */

    //////////////////////////// END serialization ////////////////////////////////

    //////////////////////////// queue methods ////////////////////////////////////

    // TODO: squash NoLock variants into non-NoLock variant if possible?
    bool QueueModuleDeletion(std::string const& id);

    bool QueueModuleDeletionNoLock(std::string const& id);

    bool QueueModuleInstantiation(std::string const& className, std::string const& id);

    bool QueueModuleInstantiationNoLock(std::string const& className, std::string const& id);

    bool QueueCallDeletion(std::string const& from, std::string const& to);

    bool QueueCallDeletionNoLock(std::string const& from, std::string const& to);

    bool QueueCallInstantiation(std::string const& className, std::string const& from, std::string const& to);

    bool QueueCallInstantiationNoLock(std::string const& className, std::string const& from, std::string const& to);

    bool HasPendingRequests();

    void ExecuteGraphUpdates();

    void RenderNextFrame();

    //////////////////////////// find methods ////////////////////////////////////
    template <class A>
    typename std::enable_if<std::is_convertible<A*, Module*>::value, bool>::type FindModule(
        std::string const& module_name, std::function<void(A const&)> const& cb) const;

    //////////////////////////// enumerators /////////////////////////////////////
    /*
     * ModulGraph braucht einen ReadWriterLock. Enumerate const Operatoren k�nnen endlich sinnvoll implementiert
     * werden.
     *
     */

private:
    //////////////////////////// locking //////////////////////////////////////////
    std::scoped_lock<std::unique_lock<std::mutex>, std::unique_lock<std::mutex>, std::unique_lock<std::mutex>,
        std::unique_lock<std::mutex>>
    AcquireQueueLocks();

    // get invalidated and the user is helpless
    [[nodiscard]] ModuleList_t::iterator find_module(std::string const& name);

    [[nodiscard]] ModuleList_t::const_iterator find_module(std::string const& name) const;

    [[nodiscard]] CallList_t::iterator find_call(std::string const& from, std::string const& to);

    [[nodiscard]] CallList_t::const_iterator find_call(std::string const& from, std::string const& to) const;

    [[nodiscard]] bool add_module(ModuleInstantiationRequest_t const& request);

    [[nodiscard]] bool add_call(CallInstantiationRequest_t const& request);

    bool delete_module(ModuleDeletionRequest_t const& request);

    bool delete_call(CallDeletionRequest_t const& request);


    /** Queue for module deletions */
    ModuleDeletionQueue_t module_deletion_queue_;

    /** Queue for module instantiation */
    ModuleInstantiationQueue_t module_instantiation_queue_;

    /** Queue for call deletions */
    CallDeletionQueue_t call_deletion_queue_;

    /** Queue for call instantiations */
    CallInstantiationQueue_t call_instantiation_queue_;

    /** List of modules that this graph owns */
    ModuleList_t module_list_;

    /** List of call that this graph owns */
    CallList_t call_list_;

    /** Reader/Writer mutex for the graph */
    mutable std::shared_mutex graph_lock_;

    std::unique_ptr<console::AbstractRenderAPI> rapi_;
    std::string rapi_root_name;
    std::list<std::function<bool()>> rapi_commands;
    std::list<view::AbstractView*> views_;

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


template <class A>
typename std::enable_if<std::is_convertible<A*, megamol::core::Module*>::value, bool>::type
megamol::core::MegaMolGraph::FindModule(std::string const& module_name, std::function<void(A const&)> const& cb) const {
    auto const mod = find_module(module_name);

    if (mod == module_list_.cend() || mod->first != nullptr) {
        vislib::sys::Log::DefaultLog.WriteInfo("MegaMolGraph: Could not find module %s\n", module_name.c_str());

        return false;
    }

    cb(*(mod->first));
    return true;
}


} /* namespace core */
} // namespace megamol

#endif /* MEGAMOLCORE_MEGAMOLGRAPH_H_INCLUDED */