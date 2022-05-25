/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <glowl/BufferObject.hpp>
#include <glowl/GLSLProgram.hpp>
#include <tbb/tbb.h>

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/flags/FlagStorage.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_gl/flags/FlagCollectionGL.h"

namespace megamol::mmstd_gl {

/**
 * Class holding a GL buffer of uints which contain flags that say something
 * about a synchronized other piece of data (index equality).
 * Can be used for storing selection etc. Should be kept in sync with the normal
 * FlagStorage, which resides on CPU.
 */
class UniFlagStorage : public core::FlagStorage {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "UniFlagStorageGL";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Module representing an index-synced array of flag uints as a CPU or GL buffer";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    std::vector<std::string> requested_lifetime_resources() override {
        std::vector<std::string> resources = Module::requested_lifetime_resources();
        resources.emplace_back("OpenGL_Context"); // GL modules should request the GL context resource
        return resources;
    }

    /** Ctor. */
    UniFlagStorage();

    /** Dtor. */
    ~UniFlagStorage() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * Access the flags provided by the UniFlagStorage
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool readCPUDataCallback(core::Call& caller) override;

    /**
     * Write/update the flags provided by the UniFlagStorage
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool writeCPUDataCallback(core::Call& caller) override;

    /**
     * Access the flags provided by the UniFlagStorage
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool readGLDataCallback(core::Call& caller);

    /**
     * Write/update the flags provided by the UniFlagStorage
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool writeGLDataCallback(core::Call& caller);

    void serializeGLData();

    bool onJSONChanged(core::param::ParamSlot& slot) override;

    /**
     * Helper to copy CPU flags to GL flags
     */
    void CPU2GLCopy();

    /**
     * Helper to copy GL flags to CPU flags
     */
    void GL2CPUCopy();

    /** The slot for reading the data */
    core::CalleeSlot readGLFlagsSlot;

    /** The slot for writing the data */
    core::CalleeSlot writeGLFlagsSlot;

    std::unique_ptr<glowl::GLSLProgram> compressGPUFlagsProgram;
    std::shared_ptr<mmstd_gl::FlagCollection_GL> theGLData;
    bool cpu_stale = true;
    bool gpu_stale = true;
};

} // namespace megamol::mmstd_gl
