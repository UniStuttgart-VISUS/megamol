/*
 * UniFlagStorage.h
 *
 * Copyright (C) 2019-2021 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "vislib_gl/graphics/gl/IncludeAllGL.h"

#include "mmcore/UniFlagStorage.h"
#include "mmcore_gl/FlagCollectionsGL.h"

namespace megamol {
namespace core_gl {

/**
 * Class holding a GL buffer of uints which contain flags that say something
 * about a synchronized other piece of data (index equality).
 * Can be used for storing selection etc. Should be kept in sync with the normal
 * FlagStorage, which resides on CPU.
 */
class MEGAMOLCORE_API UniFlagStorageGL : public core::UniFlagStorage {
public:
    // enum { ENABLED = 1 << 0, FILTERED = 1 << 1, SELECTED = 1 << 2, SOFTSELECTED = 1 << 3 };

    std::vector<std::string> requested_lifetime_resources() override {
        std::vector<std::string> resources = Module::requested_lifetime_resources();
        resources.emplace_back("OpenGL_Context"); // GL modules should request the GL context resource
        return resources;
    }

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
        return "Module representing an index-synced array of flag uints as a GPU buffer";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor. */
    UniFlagStorageGL(void);

    /** Dtor. */
    virtual ~UniFlagStorageGL(void);

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

private:
    /**
     * Access the flags provided by the UniFlagStorage
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool readDataCallback(core::Call& caller);

    /**
     * Write/update the flags provided by the UniFlagStorage
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool writeDataCallback(core::Call& caller);

    /**
     * Access the flags provided by the UniFlagStorage
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool readCPUDataCallback(core::Call& caller);

    /**
     * Write/update the flags provided by the UniFlagStorage
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool writeCPUDataCallback(core::Call& caller);

    /**
     * Helper to copy CPU flags to GL flags
     */
    void CPU2GLCopy() {
        theData->validateFlagCount(theCPUData->flags->size());
        theData->flags->bufferSubData(*(theCPUData->flags));
    }

    /**
     * Helper to copy GL flags to CPU flags
     */
    void GL2CPUCopy() {
        auto const num = theData->flags->getByteSize() / sizeof(uint32_t);
        theCPUData->validateFlagCount(num);
        glGetNamedBufferSubData(theData->flags->getName(), 0, theData->flags->getByteSize(), theCPUData->flags->data());
    }

    /** The slot for reading the data */
    core::CalleeSlot readFlagsSlot;

    /** The slot for writing the data */
    core::CalleeSlot writeFlagsSlot;

    std::shared_ptr<FlagCollection_GL> theData;
};

} // namespace core_gl
} /* end namespace megamol */
