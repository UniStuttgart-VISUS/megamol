/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/flags/FlagCollection.h"
#include "mmstd/flags/FlagStorageTypes.h"

namespace megamol::core {

/**
 * Class holding a buffer of uints which contain flags that say something
 * about a synchronized other piece of data (index equality).
 * Can be used for storing selection etc.
 */
class FlagStorage : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "FlagStorage";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Module representing an index-synced array of flag uints as a CPU buffer";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    FlagStorage();

    /** Dtor. */
    ~FlagStorage() override;

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

    core::param::ParamSlot skipFlagsSerializationParam;
    core::param::ParamSlot serializedFlags;

    std::shared_ptr<FlagCollection_CPU> theCPUData;

    // set initial version (in first frame, with temp data) != 0 so call data comes out as "has Update"
    // and eventually triggers a reload of actual user-provided data via validateFlagCount()
    uint32_t version = 1;
};

} // namespace megamol::core
