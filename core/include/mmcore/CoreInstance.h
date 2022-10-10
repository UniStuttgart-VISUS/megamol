/*
 * CoreInstance.h
 *
 * Copyright (C) 2008, 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_COREINSTANCE_H_INCLUDED
#define MEGAMOLCORE_COREINSTANCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/AbstractSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/JobDescription.h"
#include "mmcore/JobInstance.h"
#include "mmcore/JobInstanceRequest.h"
#include "mmcore/LuaState.h"
#include "mmcore/ParamValueSetRequest.h"
#include "mmcore/RootModuleNamespace.h"
#include "mmcore/ViewDescription.h"
#include "mmcore/ViewInstance.h"
#include "mmcore/ViewInstanceRequest.h"
#include "mmcore/factories/CallDescription.h"
#include "mmcore/factories/CallDescriptionManager.h"
#include "mmcore/factories/ModuleDescription.h"
#include "mmcore/factories/ModuleDescriptionManager.h"
#include "mmcore/factories/ObjectDescription.h"
#include "mmcore/factories/ObjectDescriptionManager.h"
#include "mmcore/factories/PluginDescriptor.h"
#include "mmcore/param/AbstractParam.h"
#include "mmcore/param/ParamUpdateListener.h"
#include "mmcore/utility/Configuration.h"

#include "mmcore/utility/log/Log.h"
#include "vislib/Array.h"
#include "vislib/IllegalStateException.h"
#include "vislib/Map.h"
#include "vislib/Pair.h"
#include "vislib/PtrArray.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/SmartPtr.h"
#include "vislib/String.h"
#include "vislib/sys/AutoLock.h"
#include "vislib/sys/CriticalSection.h"
#include "vislib/sys/DynamicLinkLibrary.h"
#include "vislib/sys/Lockable.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#define GOES_INTO_GRAPH
#define GOES_INTO_TRASH
#define REMOVE_GRAPH

namespace megamol {
namespace core {

/* forward declaration */
class AbstractService;

namespace utility {

/* forward declaration */
class ServiceManager;

} /* end namespace utility */

/**
 * class of core instances.
 */
class CoreInstance {
public:
    friend class megamol::core::LuaState;

    typedef std::unordered_map<std::string, size_t> ParamHashMap_t;

    /** ctor */
    CoreInstance(void);

    /** dtor */
    virtual ~CoreInstance(void);

    /**
     * Answer the call description manager of the assembly.
     *
     * @return The call description manager of the assembly.
     */
    virtual const factories::CallDescriptionManager& GetCallDescriptionManager(void) const;

    /**
     * Answer the module description manager of the assembly.
     *
     * @return The module description manager of the assembly.
     */
    virtual const factories::ModuleDescriptionManager& GetModuleDescriptionManager(void) const;

    /** return the contained LuaState */
    inline LuaState* GetLuaState(void) {
        return this->lua;
    }

    /** return whether loaded project files are Lua-based or legacy */
    inline bool IsLuaProject() const {
        return this->loadedLuaProjects.Count() > 0;
    }

    vislib::StringA GetMergedLuaProject() const;

    /**
     * Initialises the instance. This method must only be called once!
     *
     * @throws vislib::IllegalStateException if the instance already is
     *         initialised.
     */
    void Initialise();

    /**
     * Returns the configuration object of this instance.
     *
     * @return The configuration object of this instance.
     */
    inline const megamol::core::utility::Configuration& Configuration(void) const {
        return this->config;
    }

    /**
     * Searches for an view description object with the given name.
     *
     * @param name The name to search for.
     *
     * @return The found view description object or NULL if the name is
     *         not found.
     */
    std::shared_ptr<const ViewDescription> FindViewDescription(const char* name);

    /**
     * Searches for an view description object with the given name.
     *
     * @param name The name to search for.
     *
     * @return The found view description object or NULL if the name is
     *         not found.
     */
    std::shared_ptr<const JobDescription> FindJobDescription(const char* name);

    /**
     * Requests the instantiation of the view defined by the given
     * description.
     *
     * @param desc The description of the view to be instantiated.
     * @param id The identifier to be used for the new instance.
     * @param param The parameters to be set
     */
    void RequestViewInstantiation(
        const ViewDescription* desc, const vislib::StringA& id, const ParamValueSetRequest* param = NULL);

    /**
     * Requests the instantiation of the job defined by the given
     * description.
     *
     * @param desc The description of the job to be instantiated.
     * @param id The identifier to be used for the new instance.
     * @param param The parameters to be set
     */
    void RequestJobInstantiation(
        const JobDescription* desc, const vislib::StringA& id, const ParamValueSetRequest* param = NULL);

    /**
     * Request deletion of the module with the given id.
     */
#ifdef REMOVE_GRAPH
    bool GOES_INTO_GRAPH RequestModuleDeletion(const vislib::StringA& id);
#endif


    /**
     * Request deletion of call connecting callerslot from
     * to calleeslot to.
     */
#ifdef REMOVE_GRAPH
    bool GOES_INTO_GRAPH RequestCallDeletion(const vislib::StringA& from, const vislib::StringA& to);
#endif

    /**
     * Request instantiation of a module of class className
     * with the name id.
     */
#ifdef REMOVE_GRAPH
    bool GOES_INTO_GRAPH RequestModuleInstantiation(const vislib::StringA& className, const vislib::StringA& id);
#endif

    /**
     * Request instantiation of a call of class className, connecting
     * Callerslot from to Calleeslot to.
     */
#ifdef REMOVE_GRAPH
    bool GOES_INTO_GRAPH RequestCallInstantiation(
        const vislib::StringA& className, const vislib::StringA& from, const vislib::StringA& to);
#endif

    /**
     * Request instantiation of a call at the end of a chain of calls of
     * type className. See Daisy-Chaining Paradigm.
     */
#ifdef REMOVE_GRAPH
    bool GOES_INTO_GRAPH RequestChainCallInstantiation(
        const vislib::StringA& className, const vislib::StringA& chainStart, const vislib::StringA& to);
#endif

    /**
     * Request setting the parameter id to the value.
     */
#ifdef REMOVE_GRAPH
    bool GOES_INTO_GRAPH RequestParamValue(const vislib::StringA& id, const vislib::StringA& value);

    bool GOES_INTO_GRAPH CreateParamGroup(const vislib::StringA& name, const int size);
    bool GOES_INTO_GRAPH RequestParamGroupValue(
        const vislib::StringA& group, const vislib::StringA& id, const vislib::StringA& value);

    /**
     * Inserts a flush event into the graph update queues
     */
    bool GOES_INTO_GRAPH FlushGraphUpdates();

    /**
     * Returns a pointer to the parameter with the given name.
     *
     * @param name The name of the parameter to find.
     * @param quiet Flag controlling the error output if the parameter is
     *              not found.
     * @param create create a StringParam if name is not found
     *
     * @return The found parameter or NULL if no parameter with this name
     *         exists.
     */
    vislib::SmartPtr<param::AbstractParam> GOES_INTO_GRAPH FindParameter(
        const vislib::StringA& name, bool quiet = false, bool create = false);

    /**
     * Returns a pointer to the parameter with the given name.
     *
     * @param name The name of the parameter to find.
     * @param quiet Flag controlling the error output if the parameter is
     *              not found.
     * @param create create a StringParam if name is not found
     *
     * @return The found parameter or NULL if no parameter with this name
     *         exists.
     */
    inline vislib::SmartPtr<param::AbstractParam> GOES_INTO_GRAPH FindParameter(
        const vislib::StringW& name, bool quiet = false, bool create = false) {
        // absolutly sufficient, since module namespaces use ANSI strings
        return this->FindParameter(vislib::StringA(name), quiet, create);
    }
#endif

#ifdef REMOVE_GRAPH
    /**
     * Serializes the current graph into lua commands.
     *
     * @param serInstances The serialized instances.
     * @param serModules   The serialized modules.
     * @param serCalls     The serialized calls.
     * @param serParams    The serialized parameters.
     */
    std::string SerializeGraph();

    /**
     * Enumerates all modules of the graph, calling cb for each encountered module.
     * If entry_point is specified, the graph is traversed starting from that module or namespace,
     * otherwise, it is traversed from the root.
     *
     * @param entry_point the name of the module/namespace for traversal start
     * @param cb the lambda
     *
     */
    inline void EnumModulesNoLock(const std::string& entry_point, std::function<void(Module*)> cb) {
        auto thingy = this->namespaceRoot->FindNamedObject(entry_point.c_str());
        bool success = false;
        if (thingy) {
            auto mod = dynamic_cast<Module*>(thingy.get());
            auto ns = dynamic_cast<ModuleNamespace*>(thingy.get());
            if (mod) {
                success = true;
                this->EnumModulesNoLock(mod, cb);
            } else if (ns) {
                success = true;
                this->EnumModulesNoLock(ns, cb);
            }
        }
        if (!success) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "EnumModulesNoLock: Unable to find module nor namespace \"%s\" as entry point", entry_point.c_str());
        }
    }

    /**
     * Enumerates all modules of the graph, calling cb for each encountered module.
     * If entry_point is specified, the graph is traversed starting from that module or namespace,
     * otherwise, it is traversed from the root.
     *
     * @param entry_point traversal start or nullptr
     * @param cb the lambda
     *
     */
    void EnumModulesNoLock(core::AbstractNamedObject* entry_point, std::function<void(Module*)> cb);

    /**
     * Searches for a specific module called module_name of type A and
     * then executes a lambda.
     *
     * @param module_name name of the module
     * @param cb the lambda
     *
     * @returns true, if the module is found and of type A, false otherwise.
     */
    template<class A>
    typename std::enable_if<std::is_convertible<A*, Module*>::value, bool>::type GOES_INTO_GRAPH FindModuleNoLock(
        std::string module_name, std::function<void(A*)> cb) {
        auto ano_container = AbstractNamedObjectContainer::dynamic_pointer_cast(this->namespaceRoot);
        auto ano = ano_container->FindNamedObject(module_name.c_str());
        auto vi = dynamic_cast<A*>(ano.get());
        if (vi != nullptr) {
            cb(vi);
            return true;
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to find module \"%s\" for processing", module_name.c_str());
            return false;
        }
    }

    /**
     * Enumerates all ParamSlots of a Module of type A and executes a lambda on them.
     *
     * @param module_name name of the module
     * @param cb the lambda
     *
     * @returns true, if the module is found and of type A, false otherwise.
     */
    template<class A>
    typename std::enable_if<std::is_convertible<A*, Module*>::value, bool>::type GOES_INTO_GRAPH
    EnumerateParameterSlotsNoLock(std::string module_name, std::function<void(param::ParamSlot&)> cb) {
        auto ano_container = AbstractNamedObjectContainer::dynamic_pointer_cast(this->namespaceRoot);
        auto ano = ano_container->FindNamedObject(module_name.c_str());
        auto vi = dynamic_cast<A*>(ano.get());
        bool found = false;
        if (vi != nullptr) {
            for (auto c = vi->ChildList_Begin(); c != vi->ChildList_End(); c++) {
                auto ps = dynamic_cast<param::ParamSlot*>((*c).get());
                if (ps != nullptr) {
                    cb(*ps);
                    found = true;
                }
            }
            if (!found) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Unable to find a ParamSlot in module \"%s\" for processing", module_name.c_str());
            }
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to find module \"%s\" for processing", module_name.c_str());
        }
        return found;
    }

    /**
     * Enumerates all CallerSlots of a Module of type A where a Call of type C is connected and executes a lambda on
     * the Call.
     *
     * @param module_name name of the module
     * @param cb the lambda
     *
     * @returns true, if the module is found and of type A and has a Call of type C, false otherwise.
     */
    template<class A, class C>
    typename std::enable_if<std::is_convertible<A*, Module*>::value && std::is_convertible<C*, Call*>::value,
        bool>::type GOES_INTO_GRAPH
    EnumerateCallerSlotsNoLock(std::string module_name, std::function<void(C&)> cb) {
        auto ano_container = AbstractNamedObjectContainer::dynamic_pointer_cast(this->namespaceRoot);
        auto ano = ano_container->FindNamedObject(module_name.c_str());
        auto vi = dynamic_cast<A*>(ano.get());
        bool found = false;
        if (vi != nullptr) {
            for (auto c = vi->ChildList_Begin(); c != vi->ChildList_End(); c++) {
                auto sl = dynamic_cast<megamol::core::CallerSlot*>((*c).get());
                if (sl != nullptr) {
                    auto call = sl->template CallAs<C>();
                    if (call != nullptr) {
                        cb(*(call));
                        found = true;
                    }
                }
            }
            if (!found) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Unable to find a CallerSlot in module \"%s\" for processing", module_name.c_str());
            }
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to find module \"%s\" for processing", module_name.c_str());
        }
        return found;
    }

#endif

#ifdef REMOVE_GRAPH
    /**
     * Removes all obsolete modules from the module graph
     */
    void GOES_INTO_GRAPH CleanupModuleGraph(void);
#endif

    /**
     * Shuts down the application by terminating all jobs and closing all views
     */
    void Shutdown(void);

#ifdef REMOVE_GRAPH

    /**
     * Fired whenever a parameter updates it's value
     *
     * @param slot The parameter slot
     */
    void GOES_INTO_GRAPH ParameterValueUpdate(param::ParamSlot& slot);

    /**
     * Adds a ParamUpdateListener to the list of registered listeners
     *
     * @param pul The ParamUpdateListener to add
     */
    inline void GOES_INTO_GRAPH RegisterParamUpdateListener(param::ParamUpdateListener* pul) {
        if (!this->paramUpdateListeners.Contains(pul)) {
            this->paramUpdateListeners.Add(pul);
        }
    }

    /**
     * Removes a ParamUpdateListener from the list of registered listeners
     *
     * @param pul The ParamUpdateListener to remove
     */
    inline void GOES_INTO_GRAPH UnregisterParamUpdateListener(param::ParamUpdateListener* pul) {
        this->paramUpdateListeners.RemoveAll(pul);
    }
#endif

#ifdef REMOVE_GRAPH
    /**
     * Answer the root object of the module graph.
     * Used for internal computations only
     *
     * @return The root object of the module graph
     */
    inline RootModuleNamespace::const_ptr_type GOES_INTO_TRASH ModuleGraphRoot(void) const {
        return this->namespaceRoot;
    }
#endif

    /**
     * Access to the plugin manager
     *
     * @return The plugin manager
     */
    inline const std::vector<factories::AbstractPluginInstance::ptr_type>& GetPlugins() const {
        return plugins;
    }

    /**
     * Callback to delete service objects
     *
     * @param The service to be deleted
     */
    typedef void (*ServiceDeletor)(AbstractService*&);

    /**
     * Installs a service object. The service object is initialized and potentially enabled
     *
     * @param Tp class of the service. Must be derived from AbstractService
     *
     * @return 0 in case of an error. Larger zero is the service ID.
     */
    template<class Tp>
    inline unsigned int InstallService() {
        AbstractService* s = new Tp(*this);
        int retval = InstallServiceObject(s, [](AbstractService*& s) {
            delete s;
            s = nullptr;
        });
        if (retval == 0) {
            delete s;
        }
        return retval;
    }

    /**
     * Installs a service object. The service object is initialized and potentially enabled
     *
     * @param service The service object to be installed
     *
     * @return 0 in case of an error. Larger zero is the service ID. If zero
     *         is returned, the caller is responsible for deleting the
     *         service object. Otherwise the core instance takes control of
     *         the memory.
     */
    unsigned int InstallServiceObject(AbstractService* service, ServiceDeletor deletor);

    /**
     * Answer the installed service object by it's ID.
     *
     * @param ID The id of the service object to be returned
     *
     * @return The installed service object with the provided ID or null if no such service exists.
     */
    AbstractService* GetInstalledService(unsigned int id);

    /**
     * sets the current ImGui context, i.e. the one that was last touched when traversing the graph
     */
    inline void SetCurrentImGuiContext(void* ctx) {
        this->lastImGuiContext = ctx;
    }

    /**
     * gets the current ImGui context, i.e. the one that was last touched when traversing the graph
     */
    inline void* GetCurrentImGuiContext() const {
        return this->lastImGuiContext;
    }

    /**
     * get the number of the currently rendered frame
     */
    inline uint32_t GetFrameID(void) {
        return this->frameID;
    }

    /**
     * Set the number of the currently rendered frame. Whatever you think you are doing, don't: it's wrong.
     * This method is for use by the frontend only.
     */
    inline void SetFrameID(uint32_t frameID) {
        this->frameID = frameID;
    }

    /**
     * Getter for shader paths
     */
    std::vector<std::filesystem::path> GetShaderPaths() const;

    inline void SetConfigurationPaths_Frontend3000Compatibility(
        std::string app_dir, std::vector<std::string> shader_dirs, std::vector<std::string> resource_dirs) {
        this->config.SetApplicationDirectory(app_dir.c_str());

        for (auto& sd : shader_dirs) {
            this->config.AddShaderDirectory(sd.c_str());
        }

        for (auto& rd : resource_dirs) {
            this->config.AddResourceDirectory(rd.c_str());
        }
    }

private:
    /**
     * Utility struct for quickstart configuration
     */
    typedef struct _quickstepinfo_t {

        /** The name of the slot of the previous mod */
        vislib::StringA prevSlot;

        /** The name of the slot of the next mod */
        vislib::StringA nextSlot;

        /** module one step upward */
        factories::ModuleDescription::ptr nextMod;

        /** call connecting 'nextMod' to previous mod */
        factories::CallDescription::ptr call;

        /**
         * Assignment operator
         *
         *
         * @return A reference to this
         */
        struct _quickstepinfo_t& operator=(const struct _quickstepinfo_t& rhs) {
            this->prevSlot = rhs.prevSlot;
            this->nextSlot = rhs.nextSlot;
            this->nextMod = rhs.nextMod;
            this->call = rhs.call;
            return *this;
        }

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return True if this and rhs are equal
         */
        bool operator==(const struct _quickstepinfo_t& rhs) {
            return (this->prevSlot == rhs.prevSlot) && (this->nextSlot == rhs.nextSlot) &&
                   (this->nextMod == rhs.nextMod) && (this->call == rhs.call);
        }

    } quickStepInfo;

    /**
     * Closes a view or job handle (the corresponding instance object will
     * be deleted by the caller.
     *
     * @param obj The object to be removed from the module namespace.
     */
    void closeViewJob(ModuleNamespace::ptr_type obj);

    /**
     * Loads the plugin 'filename'
     *
     * @param filename The plugin to load
     */
    void loadPlugin(const std::shared_ptr<factories::AbstractPluginDescriptor>& plugin);

    /**
     * Translates shader paths to include paths in the compiler options for the ShaderFactory.
     *
     * @param config Configuration instance
     */
    void translateShaderPaths(megamol::core::utility::Configuration const& config);

#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */

    /** the cores configuration */
    megamol::core::utility::Configuration config;

    /** The paths to the shaders */
    std::vector<std::filesystem::path> shaderPaths;

    /** The Lua state */
    megamol::core::LuaState* lua;

    /** illegal hack: the last ImGui context */
    void* lastImGuiContext = nullptr;

    /**
     * All of the verbatim loaded project files. We need to keep them to send them
     * to interested parties, like the simpleclusterclient, so they can interpret
     * them THEMSELVES. All control flow must be retained to allow for asymmetric
     * MegaMol execution.
     */
    vislib::Array<vislib::Pair<vislib::StringA, vislib::StringA>> loadedLuaProjects;

    /** The manager of the builtin view descriptions */
    megamol::core::factories::ObjectDescriptionManager<megamol::core::ViewDescription> builtinViewDescs;

    /** The manager of the view descriptions load from projects */
    megamol::core::factories::ObjectDescriptionManager<megamol::core::ViewDescription> projViewDescs;

    /** The manager of the builtin job descriptions */
    megamol::core::factories::ObjectDescriptionManager<megamol::core::JobDescription> builtinJobDescs;

    /** The manager of the builtin job descriptions */
    megamol::core::factories::ObjectDescriptionManager<megamol::core::JobDescription> projJobDescs;

    /** The list of pending views to be instantiated */
    vislib::SingleLinkedList<ViewInstanceRequest> pendingViewInstRequests;

    /** The list of pending jobs to be instantiated */
    vislib::SingleLinkedList<JobInstanceRequest> pendingJobInstRequests;

#ifdef REMOVE_GRAPH
    /** the list of calls to be instantiated: (class,(from,to))* */
    vislib::SingleLinkedList<core::InstanceDescription::CallInstanceRequest> GOES_INTO_GRAPH pendingCallInstRequests;

    /** the list of calls to be instantiated: (class,(from == chainStart,to))* */
    vislib::SingleLinkedList<core::InstanceDescription::CallInstanceRequest> GOES_INTO_GRAPH
        pendingChainCallInstRequests;

    /** the list of modules to be instantiated: (class, id)* */
    vislib::SingleLinkedList<core::InstanceDescription::ModuleInstanceRequest> GOES_INTO_GRAPH
        pendingModuleInstRequests;

    /** the list of calls to be deleted: (from,to)* */
    vislib::SingleLinkedList<vislib::Pair<vislib::StringA, vislib::StringA>> GOES_INTO_GRAPH pendingCallDelRequests;

    /** the list of modules to be deleted: (id)* */
    vislib::SingleLinkedList<vislib::StringA> GOES_INTO_GRAPH pendingModuleDelRequests;

    /** the list of (parameter = value) pairs that need to be set */
    vislib::SingleLinkedList<vislib::Pair<vislib::StringA, vislib::StringA>> GOES_INTO_GRAPH pendingParamSetRequests;

    struct GOES_INTO_GRAPH ParamGroup {
        int GroupSize;
        vislib::StringA Name;
        vislib::Map<vislib::StringA, vislib::StringA> Requests;

        bool operator==(const ParamGroup& other) const {
            return this->Name.Equals(other.Name);
        }
    };

    vislib::Map<vislib::StringA, ParamGroup> GOES_INTO_GRAPH pendingGroupParamSetRequests;

    /** list of indices into view instantiation requests pointing to flush events */
    std::vector<size_t> GOES_INTO_GRAPH viewInstRequestsFlushIndices;

    /** list of indices into job instantiation requests pointing to flush events */
    std::vector<size_t> GOES_INTO_GRAPH jobInstRequestsFlushIndices;

    /** list of indices into call instantiation requests pointing to flush events */
    std::vector<size_t> GOES_INTO_GRAPH callInstRequestsFlushIndices;

    /** list of indices into chain call instantiation requests pointing to flush events */
    std::vector<size_t> GOES_INTO_GRAPH chainCallInstRequestsFlushIndices;

    /** list of indices into module instantiation requests pointing to flush events */
    std::vector<size_t> GOES_INTO_GRAPH moduleInstRequestsFlushIndices;

    /** list of indices into call deletion requests pointing to flush events */
    std::vector<size_t> GOES_INTO_GRAPH callDelRequestsFlushIndices;

    /** list of indices into module deletion requests pointing to flush events */
    std::vector<size_t> GOES_INTO_GRAPH moduleDelRequestsFlushIndices;

    /** list of indices into param set requests pointing to flush events */
    std::vector<size_t> GOES_INTO_GRAPH paramSetRequestsFlushIndices;

    /** list of indices into group param set requests pointing to flush events */
    std::vector<size_t> GOES_INTO_GRAPH groupParamSetRequestsFlushIndices;

    /**
     * You need to lock this if you manipulate any pending* lists. The lists
     * are designed to be manipulated from the Lua interface which CAN be
     * invoked from another thread (the LuaRemoteHost, for example).
     */
    mutable vislib::sys::CriticalSection GOES_INTO_GRAPH graphUpdateLock;
#endif

    /** The module namespace root */
    RootModuleNamespace::ptr_type namespaceRoot;

    /** the count of rendered frames */
    uint32_t frameID;

#ifdef REMOVE_GRAPH
    /** List of registered param update listeners */
    vislib::SingleLinkedList<param::ParamUpdateListener*> GOES_INTO_GRAPH paramUpdateListeners;
#endif

    /** Vector storing param updates per frame */
    param::ParamUpdateListener::param_updates_vec_t paramUpdates;

    /** The loaded plugins */
    std::vector<factories::AbstractPluginInstance::ptr_type> plugins;

    /** The manager of registered services */
    utility::ServiceManager* services;

#ifdef REMOVE_GRAPH
    /** Map of all parameter hashes (as requested by GetFullParameterHash)*/
    ParamHashMap_t GOES_INTO_GRAPH lastParamMap;

    /** Global hash of all parameters (is increased if any parameter defintion changes) */
    size_t GOES_INTO_GRAPH parameterHash;
#endif

#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */

    /**
     * Factory referencing all call descriptions from core and all loaded
     * plugins.
     */
    factories::CallDescriptionManager all_call_descriptions;

    /**
     * Factory referencing all module descriptions from core and all loaded
     * plugins.
     */
    factories::ModuleDescriptionManager all_module_descriptions;
};

} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_COREINSTANCE_H_INCLUDED */
