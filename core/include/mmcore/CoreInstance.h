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
#include "mmcore/RootModuleNamespace.h"
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
