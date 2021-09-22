/*
 * UniFlagStorage.h
 *
 * Copyright (C) 2019-2021 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#define GLOWL_OPENGL_INCLUDE_GLAD

#include <glowl/GLSLProgram.hpp>

#include "vislib/graphics/gl/IncludeAllGL.h"

#include "FlagCollections.h"
#include "FlagStorage.h"
#include "glowl/BufferObject.hpp"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

namespace megamol {
namespace core {

    /**
     * Class holding a GL buffer of uints which contain flags that say something
     * about a synchronized other piece of data (index equality).
     * Can be used for storing selection etc. Should be kept in sync with the normal
     * FlagStorage, which resides on CPU.
     */
    class MEGAMOLCORE_API UniFlagStorage : public core::Module {
    public:
        // enum { ENABLED = 1 << 0, FILTERED = 1 << 1, SELECTED = 1 << 2 };

        using index_type = int32_t;
        using index_vector = std::vector<index_type>;

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char* ClassName(void) {
            return "UniFlagStorage";
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
            return ogl_IsVersionGEQ(4, 3);
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

        void serializeData();
        static void check_bits(FlagStorage::FlagItemType flag_bit, index_vector& bit_starts, index_vector& bit_ends,
            index_type& curr_bit_start, index_type x, const std::shared_ptr<FlagStorage::FlagVectorType>& flags);
        static void terminate_bit(const std::shared_ptr<FlagStorage::FlagVectorType>& flags, index_vector& bit_ends,
            index_type curr_bit_start);
        static nlohmann::json make_bit_array(const index_vector& bit_starts, const index_vector& bit_ends);
        void array_to_bits(const nlohmann::json& json, FlagStorage::FlagItemType flag_bit);
        void serializeCPUData();
        void deserializeCPUData();
        bool onJSONChanged(param::ParamSlot& slot);

        /*class BitsChecker {
            UniFlagStorage::index_vector enabled_starts, enabled_ends;
            UniFlagStorage::index_vector filtered_starts, filtered_ends;
            UniFlagStorage::index_vector selected_starts, selected_ends;
            void operator()(const tbb::blocked_range<int32_t>& r) {
                for (int32_t i = r.begin(); i != r.end(); ++i) {
                    check_bits(FlagStorage::ENABLED, enabled_starts, enabled_ends, curr_enabled_start, i, cdata);
                    check_bits(FlagStorage::FILTERED, filtered_starts, filtered_ends, curr_filtered_start, i, cdata);
                    check_bits(FlagStorage::SELECTED, selected_starts, selected_ends, curr_selected_start, i, cdata);
                }
            }
        };*/

        /**
         * Helper to copy CPU flags to GL flags
         */
        void CPU2GLCopy();

        /**
         * Helper to copy GL flags to CPU flags
         */
        void GL2CPUCopy();

        /** The slot for reading the data */
        core::CalleeSlot readFlagsSlot;

        /** The slot for writing the data */
        core::CalleeSlot writeFlagsSlot;

        /** The slot for reading the data */
        core::CalleeSlot readCPUFlagsSlot;

        /** The slot for writing the data */
        core::CalleeSlot writeCPUFlagsSlot;

        core::param::ParamSlot serializedFlags;

        std::unique_ptr<glowl::GLSLProgram> compressGPUFlagsProgram;

        std::shared_ptr<FlagCollection_GL> theData;
        std::shared_ptr<FlagCollection_CPU> theCPUData;
        bool cpu_stale = true, gpu_stale = true;
        uint32_t version = 0;
    };

} // namespace core
} /* end namespace megamol */
