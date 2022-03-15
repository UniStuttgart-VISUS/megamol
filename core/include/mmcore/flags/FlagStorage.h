/*
 * FlagStorage.h
 *
 * Copyright (C) 2019-2021 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/FlagStorageBitsChecker.h"
#include "mmcore/FlagStorageTypes.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace core {

class FlagCollection_CPU;

/**
 * Class holding a buffer of uints which contain flags that say something
 * about a synchronized other piece of data (index equality).
 * Can be used for storing selection etc.
 */
class MEGAMOLCORE_API FlagStorage : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "FlagStorage";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Module representing an index-synced array of flag uints as a CPU buffer";
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
    FlagStorage(void);

    /** Dtor. */
    virtual ~FlagStorage(void);

    /**
     * Access the metadata provided by the UniFlagStorage
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool readMetaDataCallback(core::Call& caller);

    /**
     * Write/update the metadata provided by the UniFlagStorage
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool writeMetaDataCallback(core::Call& caller);

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
    virtual bool readCPUDataCallback(core::Call& caller);

    /**
     * Write/update the flags provided by the UniFlagStorage
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    virtual bool writeCPUDataCallback(core::Call& caller);

    static nlohmann::json make_bit_array(
        const FlagStorageTypes::index_vector& bit_starts, const FlagStorageTypes::index_vector& bit_ends);
    void array_to_bits(const nlohmann::json& json, FlagStorageTypes::flag_bits flag_bit);
    static FlagStorageTypes::index_type array_max(const nlohmann::json& json);
    void serializeCPUData();
    void deserializeCPUData();
    virtual bool onJSONChanged(param::ParamSlot& slot);

    /** The slot for reading the data */
    core::CalleeSlot readCPUFlagsSlot;

    /** The slot for writing the data */
    core::CalleeSlot writeCPUFlagsSlot;

    core::param::ParamSlot serializedFlags;

    std::shared_ptr<FlagCollection_CPU> theCPUData;
    bool cpu_stale = true;
    uint32_t version = 0;
};

class FlagCollection_CPU {
public:
    std::shared_ptr<FlagStorageTypes::flag_vector_type> flags;

    void validateFlagCount(FlagStorageTypes::index_type num) {
        if (flags->size() < num) {
            flags->resize(num);
            std::fill(
                flags->begin(), flags->end(), FlagStorageTypes::to_integral(FlagStorageTypes::flag_bits::ENABLED));
        }
    }
};

} // namespace core
} /* end namespace megamol */
