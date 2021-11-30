/*
 * UniFlagStorage.h
 *
 * Copyright (C) 2019-2021 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "tbb/tbb.h"

#include "glowl/BufferObject.hpp"
#include "glowl/GLSLProgram.hpp"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/FlagStorage.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace core_gl {

class FlagCollection_GL;

/**
 * Class holding a GL buffer of uints which contain flags that say something
 * about a synchronized other piece of data (index equality).
 * Can be used for storing selection etc. Should be kept in sync with the normal
 * FlagStorage, which resides on CPU.
 */
class MEGAMOLCORE_API UniFlagStorage : public core::FlagStorage {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "UniFlagStorageGL";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Module representing an index-synced array of flag uints as a CPU or GL buffer";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    std::vector<std::string> requested_lifetime_resources() override {
        std::vector<std::string> resources = Module::requested_lifetime_resources();
        resources.emplace_back("OpenGL_Context"); // GL modules should request the GL context resource
        return resources;
    }

    /** Ctor. */
    UniFlagStorage(void);

    /** Dtor. */
    virtual ~UniFlagStorage(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

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
    std::shared_ptr<core_gl::FlagCollection_GL> theGLData;
    bool gpu_stale = true;
};

class FlagCollection_GL {
public:
    std::shared_ptr<glowl::BufferObject> flags;

    void validateFlagCount(core::FlagStorageTypes::index_type num) {
        if (flags->getByteSize() / sizeof(uint32_t) < num) {
            std::vector<core::FlagStorageTypes::index_type> temp_data(
                num, core::FlagStorageTypes::to_integral(core::FlagStorageTypes::flag_bits::ENABLED));
            std::shared_ptr<glowl::BufferObject> temp_buffer =
                std::make_shared<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, temp_data, GL_DYNAMIC_DRAW);
            glowl::BufferObject::copy(flags.get(), temp_buffer.get(), 0, 0, flags->getByteSize());
            flags = temp_buffer;
        }
    }
};

} // namespace core_gl
} /* end namespace megamol */
