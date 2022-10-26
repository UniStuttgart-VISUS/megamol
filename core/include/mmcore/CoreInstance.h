/*
 * CoreInstance.h
 *
 * Copyright (C) 2008, 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "mmcore/AbstractSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/JobInstance.h"
#include "mmcore/RootModuleNamespace.h"
#include "mmcore/ViewInstance.h"
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


#define GOES_INTO_GRAPH
#define GOES_INTO_TRASH
#define REMOVE_GRAPH

namespace megamol {
namespace core {

/**
 * class of core instances.
 */
class CoreInstance {
public:
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
     * Request setting the parameter id to the value.
     */
#ifdef REMOVE_GRAPH
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
    std::shared_ptr<param::AbstractParam> GOES_INTO_GRAPH FindParameter(
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
    inline std::shared_ptr<param::AbstractParam> GOES_INTO_GRAPH FindParameter(
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

    /** the cores configuration */
    megamol::core::utility::Configuration config;

    /** The paths to the shaders */
    std::vector<std::filesystem::path> shaderPaths;

    /** illegal hack: the last ImGui context */
    void* lastImGuiContext = nullptr;

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
