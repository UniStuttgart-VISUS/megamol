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

    /** the count of rendered frames */
    uint32_t frameID;
};

} /* end namespace core */
} /* end namespace megamol */
