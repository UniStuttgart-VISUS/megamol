/*
 * CoreInstance.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_COREINSTANCE_H_INCLUDED
#define MEGAMOLCORE_COREINSTANCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/AbstractSlot.h"
#include "mmcore/ApiHandle.h"
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
#include "mmcore/api/MegaMolCore.h"
#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/factories/AbstractAssemblyInstance.h"
#include "mmcore/factories/CallDescription.h"
#include "mmcore/factories/CallDescriptionManager.h"
#include "mmcore/factories/ModuleDescription.h"
#include "mmcore/factories/ModuleDescriptionManager.h"
#include "mmcore/factories/ObjectDescription.h"
#include "mmcore/factories/ObjectDescriptionManager.h"
#include "mmcore/param/AbstractParam.h"
#include "mmcore/param/ParamUpdateListener.h"
#include "mmcore/utility/Configuration.h"
#include "mmcore/utility/LogEchoTarget.h"
#include "mmcore/utility/ShaderSourceFactory.h"

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
#include "vislib/sys/Log.h"

#include <functional>
#include <memory>
#include <unordered_map>

namespace megamol {
namespace core {

/* forward declaration */
class AbstractService;

namespace utility {

/* forward declaration */
class ServiceManager;

namespace plugins {

/* forward declaration */
class PluginManager;

} /* end namespace plugins */
} /* end namespace utility */

/**
 * class of core instances.
 */
class MEGAMOLCORE_API CoreInstance : public ApiHandle, public factories::AbstractAssemblyInstance {
public:
    friend class megamol::core::LuaState;

    typedef std::unordered_map<std::string, size_t> ParamHashMap_t;

    /**
     * Deallocator for view handles.
     *
     * @param data Must point to the CoreInstance which created this object.
     * @param obj A view object.
     */
    static void ViewJobHandleDalloc(void* data, ApiHandle* obj);

    /** ctor */
    CoreInstance(void);

    /** dtor */
    virtual ~CoreInstance(void);

    /**
     * Answer the (machine-readable) name of the assembly. This usually is
     * The name of the plugin dll/so without prefix and extension.
     *
     * @return The (machine-readable) name of the assembly
     */
    virtual const std::string& GetAssemblyName(void) const;

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
    inline LuaState* GetLuaState(void) { return this->lua; }

    /** return whether loaded project files are Lua-based or legacy */
    inline bool IsLuaProject() const { return this->loadedLuaProjects.Count() > 0; }

    vislib::StringA GetMergedLuaProject() const;

    /**
     * Answers the log object of the instance.
     *
     * @return The log object of the instance.
     */
    inline vislib::sys::Log& Log(void) { return this->log; }

    /**
     * Answer whether this instance is initialised or not.
     *
     * @return 'true' if this instance already is initialised, 'false'
     *         otherwise.
     */
    inline bool IsInitialised(void) const { return (this->preInit == NULL); }

    /**
     * Initialises the instance. This method must only be called once!
     *
     * @throws vislib::IllegalStateException if the instance already is
     *         initialised.
     */
    void Initialise(void);

    /**
     * Sets an initialisation value.
     *
     * @param key Specifies which value to set.
     * @param type Specifies the value type of 'value'.
     * @param value The value to set the initialisation value to. The type
     *              of the variable specified depends on 'type'.
     *
     * @return 'MMC_ERR_NO_ERROR' on success or an nonzero error code if
     *         the function fails.
     *
     * @throw vislib::IllegalStateException if the instance already is
     *        initialised.
     */
    mmcErrorCode SetInitValue(mmcInitValue key, mmcValueType type, const void* value);

    /**
     * Returns the configuration object of this instance.
     *
     * @return The configuration object of this instance.
     */
    inline const megamol::core::utility::Configuration& Configuration(void) const { return this->config; }

    /**
     * Returns the ShaderSourceFactory object of this inatcne.
     *
     * @return The ShaderSourceFactory object of this inatcne.
     */
    inline utility::ShaderSourceFactory& ShaderSourceFactory(void) { return this->shaderSourceFactory; }

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
     * Enumerates all view descriptions. The callback function is called for each
     * view description.
     *
     * @param func The callback function.
     * @param data The user specified pointer to be passed to the callback
     *             function.
     * @param getBuiltinToo true to also retreive the builtin view descriptions
     *					    else false
     */
    void EnumViewDescriptions(mmcEnumStringAFunction func, void* data, bool getBuiltinToo = false);

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
     * Requests all available instantiations.
     */
    void RequestAllInstantiations();

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
    bool RequestModuleDeletion(const vislib::StringA& id);

    /**
     * Request deletion of call connecting callerslot from
     * to calleeslot to.
     */
    bool RequestCallDeletion(const vislib::StringA& from, const vislib::StringA& to);

    /**
     * Request instantiation of a module of class className
     * with the name id.
     */
    bool RequestModuleInstantiation(const vislib::StringA& className, const vislib::StringA& id);

    /**
     * Request instantiation of a call of class className, connecting
     * Callerslot from to Calleeslot to.
     */
    bool RequestCallInstantiation(
        const vislib::StringA& className, const vislib::StringA& from, const vislib::StringA& to);

    /**
     * Request instantiation of a call at the end of a chain of calls of
     * type className. See Daisy-Chaining Paradigm.
     */
    bool RequestChainCallInstantiation(
        const vislib::StringA& className, const vislib::StringA& chainStart, const vislib::StringA& to);

    /**
     * Request setting the parameter id to the value.
     */
    bool RequestParamValue(const vislib::StringA& id, const vislib::StringA& value);

    bool CreateParamGroup(const vislib::StringA& name, const int size);
    bool RequestParamGroupValue(const vislib::StringA& group, const vislib::StringA& id, const vislib::StringA& value);

    /**
     * Inserts a flush event into the graph update queues
     */
    bool FlushGraphUpdates();

    //** do everything that is queued w.r.t. modules and calls */
    void PerformGraphUpdates();

    /**
     * Answer whether the core has pending requests of instantiations of
     * views.
     *
     * @return 'true' if there are pending view instantiation requests,
     *         'false' otherwise.
     */
    inline bool HasPendingViewInstantiationRequests(void) {
        vislib::sys::AutoLock l(this->graphUpdateLock);
        return !this->pendingViewInstRequests.IsEmpty();
    }

    /**
     * Answer whether the core has pending requests of instantiations of
     * jobs.
     *
     * @return 'true' if there are pending job instantiation requests,
     *         'false' otherwise.
     */
    inline bool HasPendingJobInstantiationRequests(void) {
        vislib::sys::AutoLock l(this->graphUpdateLock);
        return !this->pendingJobInstRequests.IsEmpty();
    }

    inline bool HasPendingRequests(void) {
        vislib::sys::AutoLock l(this->graphUpdateLock);
        return !this->pendingViewInstRequests.IsEmpty() || !this->pendingJobInstRequests.IsEmpty() ||
               !this->pendingCallDelRequests.IsEmpty() || !this->pendingCallInstRequests.IsEmpty() ||
               !this->pendingChainCallInstRequests.IsEmpty() || !this->pendingModuleDelRequests.IsEmpty() ||
               !this->pendingModuleInstRequests.IsEmpty() || !this->pendingParamSetRequests.IsEmpty();
    }

    vislib::StringA GetPendingViewName(void);

    /**
     * Instantiates the next pending view, if there is one.
     *
     * @return The newly created view object or 'NULL' in case of an error.
     */
    ViewInstance::ptr_type InstantiatePendingView(void);

    /**
     * Instantiates a view description filled with full names! This method
     * is for internal use by the framework. Do not call it directly
     *
     * @return The instantiated view module
     */
    view::AbstractView* instantiateSubView(ViewDescription* vd);

    /**
     * Instantiates the next pending job, if there is one.
     *
     * @return The newly created job object or 'NULL' in case of an error.
     */
    JobInstance::ptr_type InstantiatePendingJob(void);

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
    vislib::SmartPtr<param::AbstractParam> FindParameter(
        const vislib::StringA& name, bool quiet = false, bool create = false);

    /**
     * Returns the project lua contained in the exif data of a PNG file.
     * 
     * @param filename the png file name
     * 
     * @return the lua project
     */
    static std::string GetProjectFromPNG(std::string filename);

    /**
     * Returns a pointer to the parameter with the given name.
     * If the parameter value is the name of a valid parameter, it follows the path..
     *
     * @param name The name of the parameter to find.
     * @param quiet Flag controlling the error output if the parameter is
     *              not found.
     *
     * @return The found parameter or NULL if no parameter with this name
     *         exists.
     */
    vislib::SmartPtr<param::AbstractParam> FindParameterIndirect(const vislib::StringA& name, bool quiet = false);

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
    inline vislib::SmartPtr<param::AbstractParam> FindParameter(
        const vislib::StringW& name, bool quiet = false, bool create = false) {
        // absolutly sufficient, since module namespaces use ANSI strings
        return this->FindParameter(vislib::StringA(name), quiet, create);
    }

    /**
     * Loads a project into the core.
     *
     * @param filename The path to the project file to load.
     */
    void LoadProject(const vislib::StringA& filename);

    /**
     * Loads a project into the core.
     *
     * @param filename The path to the project file to load.
     */
    void LoadProject(const vislib::StringW& filename);

    /**
     * Serializes the current graph into lua commands.
     *
     * @param serInstances The serialized instances.
     * @param serModules   The serialized modules.
     * @param serCalls     The serialized calls.
     * @param serParams    The serialized parameters.
     */
    void SerializeGraph(std::string& serInstances, std::string& serModules, std::string& serCalls, std::string& serParams);

    /**
     * Enumerates all parameters. The callback function is called for each
     * parameter slot.
     *
     * @param cb The callback function.
     */
    inline void EnumParameters(std::function<void(const Module&, param::ParamSlot&)> cb) const {
        this->enumParameters(this->namespaceRoot, cb);
    }

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
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
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
    template <class A>
    typename std::enable_if<std::is_convertible<A*, Module*>::value, bool>::type FindModuleNoLock(
        std::string module_name, std::function<void(A*)> cb) {
        auto ano_container = AbstractNamedObjectContainer::dynamic_pointer_cast(this->namespaceRoot);
        auto ano = ano_container->FindNamedObject(module_name.c_str());
        auto vi = dynamic_cast<A*>(ano.get());
        if (vi != nullptr) {
            cb(vi);
            return true;
        } else {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, "Unable to find module \"%s\" for processing", module_name.c_str());
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
    template <class A>
    typename std::enable_if<std::is_convertible<A*, Module*>::value, bool>::type EnumerateParameterSlotsNoLock(
        std::string module_name, std::function<void(param::ParamSlot&)> cb) {
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
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                    "Unable to find a ParamSlot in module \"%s\" for processing", module_name.c_str());
            }
        } else {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, "Unable to find module \"%s\" for processing", module_name.c_str());
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
    template <class A, class C>
    typename std::enable_if<std::is_convertible<A*, Module*>::value && std::is_convertible<C*, Call*>::value,
        bool>::type
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
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                    "Unable to find a CallerSlot in module \"%s\" for processing", module_name.c_str());
            }
        } else {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, "Unable to find module \"%s\" for processing", module_name.c_str());
        }
        return found;
    }

    /**
     * Enumerates all parameters. The callback function is called for each
     * parameter name.
     *
     * @param func The callback function.
     * @param data The user specified pointer to be passed to the callback
     *             function.
     */
    inline void EnumParameters(mmcEnumStringAFunction func, void* data) const {
        auto toStringFunction = [func, data](const Module& mod, const param::ParamSlot& slot) {
            vislib::StringA name(mod.FullName());
            name.Append("::");
            name.Append(slot.Name());
            func(name.PeekBuffer(), data);
        };
        this->enumParameters(this->namespaceRoot, toStringFunction);
    }

    /**
     * Updates global parameter hash and returns it.
     * Comparison of parameter info is expensive.
     *
     * @return Updated parameter hash.
     */
    size_t GetGlobalParameterHash(void);

    /**
     * Answer the full name of the paramter 'param' if it is bound to a
     * parameter slot of an active module.
     *
     * @param param The parameter to search for.
     *
     * @return The full name of the parameter, or an empty string if the
     *         parameter is not found
     */
    inline vislib::StringA FindParameterName(const vislib::SmartPtr<param::AbstractParam>& param) const {
        return this->findParameterName(this->namespaceRoot, param);
    }

    /**
     * Answer the time of this instance in seconds.
     *
     * DO NOT USE THIS FUNCTION in Renderer Modules.
     * Use 'instTime' parameter in method 'Render' instead.
     *
     * @return The time of this instance.
     */
    double GetCoreInstanceTime(void) const;

    /**
     * Adds an offset to the instance time.
     *
     * @param offset The offset to be added
     */
    void OffsetInstanceTime(double offset);

    /**
     * Removes all obsolete modules from the module graph
     */
    void CleanupModuleGraph(void);

    /**
     * Closes a view or job handle (the corresponding instance object will
     * be deleted by the caller.
     *
     * @param obj The object to be removed from the module namespace.
     */
    inline void CloseViewJob(ModuleNamespace::ptr_type obj) { this->closeViewJob(obj); }

    /**
     * Shuts down the application by terminating all jobs and closing all views
     */
    void Shutdown(void);

    /**
     * Sets up the module graph based on the serialized graph description
     * from the head node of the network rendering cluster.
     *
     * @param data The serialized graph description (Pointer to an
     *             vislib::net::AbstractSimpleMessage)
     */
    void SetupGraphFromNetwork(const void* data);

    /**
     * Instantiates a call.
     *
     * @param fromPath The full namespace path of the caller slot
     * @param toPath The full namespace path of the callee slot
     * @param desc The call description
     *
     * @return The new call or 'NULL' in case of an error
     */
    Call* InstantiateCall(
        const vislib::StringA fromPath, const vislib::StringA toPath, factories::CallDescription::ptr desc);

    /**
     * Instantiates a module
     *
     * @param path The full namespace path
     * @param desc The module description
     *
     * @return The new module or 'NULL' in case of an error
     */
    Module::ptr_type instantiateModule(const vislib::StringA path, factories::ModuleDescription::ptr desc);

    /**
     * Fired whenever a parameter updates it's value
     *
     * @param slot The parameter slot
     */
    void ParameterValueUpdate(param::ParamSlot& slot);

    /**
     * Adds a ParamUpdateListener to the list of registered listeners
     *
     * @param pul The ParamUpdateListener to add
     */
    inline void RegisterParamUpdateListener(param::ParamUpdateListener* pul) {
        if (!this->paramUpdateListeners.Contains(pul)) {
            this->paramUpdateListeners.Add(pul);
        }
    }

    /**
     * Removes a ParamUpdateListener from the list of registered listeners
     *
     * @param pul The ParamUpdateListener to remove
     */
    inline void UnregisterParamUpdateListener(param::ParamUpdateListener* pul) {
        this->paramUpdateListeners.RemoveAll(pul);
    }

    /**
     * Tries to perform a quickstart with the given data file
     *
     * @param filename The file to quickstart
     */
    void Quickstart(const vislib::TString& filename);

    /**
     * Registers a file type for quickstart if supported by the OS
     *
     * @param frontend Path to the front end to be called
     * @param feparams The parameter string to be used when calling the frontend.
     *                 use '$(FILENAME)' to specify the position of the data file name.
     * @param filetype Semicolor separated list of file type extensions to register
     *                 or "*" if all known file type extensions should be used
     * @param unreg If true, the file types will be removed from the quickstart registry instead of added
     * @param overwrite If true, any previous registration will be overwritten.
     *                  If false, previous registrations will be placed as alternative start commands.
     *                  When unregistering and true, all registrations will be removed,
     *                  if false only registrations to this binary will be removed.
     */
    void QuickstartRegistry(const vislib::TString& frontend, const vislib::TString& feparams,
        const vislib::TString& filetype, bool unreg, bool overwrite);

    /**
     * Answer the root object of the module graph.
     * Used for internal computations only
     *
     * @return The root object of the module graph
     */
    inline RootModuleNamespace::const_ptr_type ModuleGraphRoot(void) const { return this->namespaceRoot; }

    /**
     * Writes the current state of the call graph to an xml file.
     *
     * @param outFilename The output file name.
     * @return 'True' on success, 'false' otherwise.
     */
    bool WriteStateToXML(const char* outFilename);

    /**
     * Access to the plugin manager
     *
     * @return The plugin manager
     */
    inline const utility::plugins::PluginManager& Plugins() const { return *plugins; }

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
    template <class Tp> inline unsigned int InstallService() {
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

private:
    /**
     * Nested class with pre initialisation values.
     */
    class PreInit {
    public:
        /** Default Ctor */
        PreInit(void);

        /**
         * Answer the config file to load.
         *
         * @return The config file to load.
         */
        inline const vislib::StringW& GetConfigFile(void) const { return this->cfgFile; }

        /**
         * Answer the config file overrides.
         *
         * @return a '\b'-separated list of '\a'-separated key-value pairs
         */
        inline const vislib::StringW& GetConfigFileOverrides(void) const { return this->cfgOverrides; }

        /**
         * Answer the log file to use.
         *
         * @return The log file to use.
         */
        inline const vislib::StringW& GetLogFile(void) const { return this->logFile; }

        /**
         * Answer the log level to use.
         *
         * @return The log level to use.
         */
        inline const unsigned int GetLogLevel(void) const { return this->logLevel; }

        /**
         * Answer the log echo level to use.
         *
         * @return The log echo level to use.
         */
        inline const unsigned int GetLogEchoLevel(void) const { return this->logEchoLevel; }

        /**
         * Answer whether the config file has been set.
         *
         * @return 'true' if the config file has been set.
         */
        inline bool IsConfigFileSet(void) const { return this->cfgFileSet; }

        /**
         * Answer whether the config file overrides have been set.
         *
         * @return 'true' if the config file overrides have been set.
         */
        inline bool IsConfigOverrideSet(void) const { return this->cfgOverridesSet; }

        /**
         * Answer whether the log file has been set.
         *
         * @return 'true' if the log file has been set.
         */
        inline bool IsLogFileSet(void) const { return this->logFileSet; }

        /**
         * Answer whether the log level has been set.
         *
         * @return 'true' if the log level has been set.
         */
        inline bool IsLogLevelSet(void) const { return this->logLevelSet; }

        /**
         * Answer whether the log echo level has been set.
         *
         * @return 'true' if the log echo level has been set.
         */
        inline bool IsLogEchoLevelSet(void) const { return this->logEchoLevelSet; }

        /**
         * Sets the config file to load.
         *
         * @param cfgFile The config file to load.
         */
        inline void SetConfigFile(const vislib::StringW& cfgFile) {
            this->cfgFile = cfgFile;
            this->cfgFileSet = true;
        }

        /**
         * Sets the config file overrides.
         *
         * @param cfgOverrides a '\b'-separated list of '\a'-separated
         *                     key-value pairs
         */
        inline void SetConfigFileOverrides(const vislib::StringW& cfgOverrides) {
            this->cfgOverrides = cfgOverrides;
            this->cfgOverridesSet = true;
        }

        /**
         * Sets the log file to use.
         *
         * @param logFile The log file to use.
         */
        inline void SetLogFile(const vislib::StringW& logFile) {
            this->logFile = logFile;
            this->logFileSet = true;
        }

        /**
         * Sets the log level to use.
         *
         * @param level The log level to use.
         */
        inline void SetLogLevel(unsigned int level) {
            this->logLevel = level;
            this->logLevelSet = true;
        }

        /**
         * Sets the log echo level to use.
         *
         * @param level The log echo level to use.
         */
        inline void SetLogEchoLevel(unsigned int level) {
            this->logEchoLevel = level;
            this->logEchoLevelSet = true;
        }

    private:
        /** Flag whether the config file has been set. */
        bool cfgFileSet : 1;

        /** Flag whether the config file overrides have been set. */
        bool cfgOverridesSet : 1;

        /** Flag whether the log file has been set. */
        bool logFileSet : 1;

        /** Flag whether the log level has been set. */
        bool logLevelSet : 1;

        /** Flag whether the log echo level has been set. */
        bool logEchoLevelSet : 1;

        /** The config file name. */
        vislib::StringW cfgFile;

        /** The log file name. */
        vislib::StringW logFile;

        /** A serialized list of config key-value overrides */
        vislib::StringW cfgOverrides;

        /** The log level. */
        unsigned int logLevel;

        /** The log echo level. */
        unsigned int logEchoLevel;
    };

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
     * Adds a project to the instance.
     *
     * @param reader The xml reader to load the project from.
     */
    void addProject(megamol::core::utility::xml::XmlReader& reader);

    /**
     * Enumerates all parameters to collect parameter hashes.
     *
     * @param path The current module namespace
     * @param map  Stores association between parameter name
     *             and parameter hash
     */
    void getGlobalParameterHash(ModuleNamespace::const_ptr_type path, ParamHashMap_t& map) const;

    /**
     * Enumerates all parameters. The callback function is called for each
     * parameter name.
     *
     * @param path The current module namespace
     * @param func The callback function.
     * @param data The user specified pointer to be passed to the callback
     *             function.
     */
    void enumParameters(
        ModuleNamespace::const_ptr_type path, std::function<void(const Module&, param::ParamSlot&)> cb) const;

    /**
     * Answer the full name of the paramter 'param' if it is bound to a
     * parameter slot of an active module.
     *
     * @param path The current module namespace
     * @param param The parameter to search for.
     *
     * @return The full name of the parameter, or an empty string if the
     *         parameter is not found
     */
    vislib::StringA findParameterName(
        ModuleNamespace::const_ptr_type path, const vislib::SmartPtr<param::AbstractParam>& param) const;

    /**
     * Closes a view or job handle (the corresponding instance object will
     * be deleted by the caller.
     *
     * @param obj The object to be removed from the module namespace.
     */
    void closeViewJob(ModuleNamespace::ptr_type obj);

    /**
     * Apply parameters from configuration file
     *
     * @param root The root namespace
     * @param id The instance description
     */
    void applyConfigParams(
        const vislib::StringA& root, const InstanceDescription* id, const ParamValueSetRequest* params);

    /**
     * Loads the plugin 'filename'
     *
     * @param filename The plugin to load
     */
    void loadPlugin(const vislib::TString& filename);

    /**
     * Compares two maps storing the association between
     * parameter names and hashes.
     *
     * @param one   First map for comparison
     * @param other Second map for comparison
     *
     * @return      True, if map "one" and "other" are the same
     */
    bool mapCompare(ParamHashMap_t& one, ParamHashMap_t& other);

    /**
     * Auto-connects a view module graph from 'from' to 'to' upwards
     *
     * @param view The view description object to receive the graph
     * @param from The name of the module to connect from (upwards)
     * @param to The optional module to connect to (upwards)
     *
     * @return True on success
     */
    bool quickConnectUp(ViewDescription& view, const char* from, const char* to);

    /**
     * Collects information on possible upwards connections for the module
     * 'from'
     *
     * @param from The module to connect from (upwards)
     * @param step List of possible upward connections
     */
    void quickConnectUpStepInfo(factories::ModuleDescription::ptr from, vislib::Array<quickStepInfo>& step);

    /**
     * Registers a single file type for quickstarting
     *
     * @param frontend The full path to the frontend to call
     * @param feparams The frontend command line parameter string
     * @param fnext The data file name extension
     * @param fnname The data file type name
     * @param keepothers If true, other open options will not be overwritten.
     */
    void registerQuickstart(const vislib::TString& frontend, const vislib::TString& feparams,
        const vislib::TString& fnext, const vislib::TString& fnname, bool keepothers);

    /**
     * Removes the registration for a single file type for quickstarting
     *
     * @param frontend The full path to the frontend to call
     * @param feparams The frontend command line parameter string
     * @param fnext The data file name extension
     * @param fnname The data file type name
     * @param keepothers If true, other open options will not be deleted.
     */
    void unregisterQuickstart(const vislib::TString& frontend, const vislib::TString& feparams,
        const vislib::TString& fnext, const vislib::TString& fnname, bool keepothers);

    /**
     * Updates flush index list after flush has been performed
     *
     * @param processedCount Number of processed events
     * @param list Index list to be updated
     */
    void updateFlushIdxList(size_t const processedCount, std::vector<size_t>& list);

    /**
     * Check if current event is after flush event
     *
     * @param eventIdx Index in queue of current event
     * @param list List of flush events
     *
     * @return True, if current event is after flush event
     */
    bool checkForFlushEvent(size_t const eventIdx, std::vector<size_t>& list) const;

    /**
     * Removes all unreachable flushes
     *
     * @param eventCount Number of events in the respective queue
     * @param list The flush index list to update
     */
    void shortenFlushIdxList(size_t const eventCount, std::vector<size_t>& list);

#ifdef _WIN32
#    pragma warning(disable : 4251)
#endif /* _WIN32 */
    /** the pre initialisation values. */
    PreInit* preInit;

    /** the cores configuration */
    megamol::core::utility::Configuration config;

    /** The shader source factory */
    utility::ShaderSourceFactory shaderSourceFactory;

    /** The log object */
    vislib::sys::Log log;

    /** The Lua state */
    megamol::core::LuaState* lua;

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

    /** the list of calls to be instantiated: (class,(from,to))* */
    vislib::SingleLinkedList<core::InstanceDescription::CallInstanceRequest> pendingCallInstRequests;

    /** the list of calls to be instantiated: (class,(from == chainStart,to))* */
    vislib::SingleLinkedList<core::InstanceDescription::CallInstanceRequest> pendingChainCallInstRequests;

    /** the list of modules to be instantiated: (class, id)* */
    vislib::SingleLinkedList<core::InstanceDescription::ModuleInstanceRequest> pendingModuleInstRequests;

    /** the list of calls to be deleted: (from,to)* */
    vislib::SingleLinkedList<vislib::Pair<vislib::StringA, vislib::StringA>> pendingCallDelRequests;

    /** the list of modules to be deleted: (id)* */
    vislib::SingleLinkedList<vislib::StringA> pendingModuleDelRequests;

    /** the list of (parameter = value) pairs that need to be set */
    vislib::SingleLinkedList<vislib::Pair<vislib::StringA, vislib::StringA>> pendingParamSetRequests;

    struct ParamGroup {
        int GroupSize;
        vislib::StringA Name;
        vislib::Map<vislib::StringA, vislib::StringA> Requests;

        bool operator==(const ParamGroup& other) const { return this->Name.Equals(other.Name); }
    };

    vislib::Map<vislib::StringA, ParamGroup> pendingGroupParamSetRequests;

    /** list of indices into view instantiation requests pointing to flush events */
    std::vector<size_t> viewInstRequestsFlushIndices;

    /** list of indices into job instantiation requests pointing to flush events */
    std::vector<size_t> jobInstRequestsFlushIndices;

    /** list of indices into call instantiation requests pointing to flush events */
    std::vector<size_t> callInstRequestsFlushIndices;

    /** list of indices into chain call instantiation requests pointing to flush events */
    std::vector<size_t> chainCallInstRequestsFlushIndices;

    /** list of indices into module instantiation requests pointing to flush events */
    std::vector<size_t> moduleInstRequestsFlushIndices;

    /** list of indices into call deletion requests pointing to flush events */
    std::vector<size_t> callDelRequestsFlushIndices;

    /** list of indices into module deletion requests pointing to flush events */
    std::vector<size_t> moduleDelRequestsFlushIndices;

    /** list of indices into param set requests pointing to flush events */
    std::vector<size_t> paramSetRequestsFlushIndices;

    /** list of indices into group param set requests pointing to flush events */
    std::vector<size_t> groupParamSetRequestsFlushIndices;

    /**
     * You need to lock this if you manipulate any pending* lists. The lists
     * are designed to be manipulated from the Lua interface which CAN be
     * invoked from another thread (the LuaRemoteHost, for example).
     */
    mutable vislib::sys::CriticalSection graphUpdateLock;

    /** The module namespace root */
    RootModuleNamespace::ptr_type namespaceRoot;

    /** the time offset */
    double timeOffset;

    /** List of registered param update listeners */
    vislib::SingleLinkedList<param::ParamUpdateListener*> paramUpdateListeners;

    /** The manager of loaded plugins */
    utility::plugins::PluginManager* plugins;

    /** The manager of registered services */
    utility::ServiceManager* services;

    /** Map of all parameter hashes (as requested by GetFullParameterHash)*/
    ParamHashMap_t lastParamMap;

    /** Global hash of all parameters (is increased if any parameter defintion changes) */
    size_t parameterHash;

#ifdef _WIN32
#    pragma warning(default : 4251)
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
