/*
 * DifferenceVolume.h
 *
 * Copyright (C) 2019 by Visualisierungsinstitut der Universität Stuttgart.
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <array>
#include <vector>

#include "mmcore/misc/VolumetricDataCall.h"
#include "mmcore/misc/VolumetricMetadataStore.h"

#include "mmcore/param/ParamSlot.h"

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"


namespace megamol {
namespace stdplugin {
namespace volume {

    /**
     * Computes the difference between two time steps of a volumetric data
     * source.
     */
    class DifferenceVolume : public core::Module {

    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static inline constexpr const char* ClassName(void) {
            return "DifferenceVolume";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static inline constexpr const char* Description(void) {
            return "Computes the difference between volumes.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static inline constexpr bool IsAvailable(void) {
            return true;
        }

        /**
         * Initialises a new instance.
         */
        DifferenceVolume(void);

        /**
         * Finalises an instance.
         */
        virtual ~DifferenceVolume(void);

    protected:

        /**
         * Compute the size of a single frame in bytes.
         */
        static std::size_t getFrameSize(
            const core::misc::VolumetricMetadata_t& md);

        /**
         * Gets the scalar type to be used for the difference volume.
         */
        static core::misc::ScalarType_t getDifferenceType(
            const core::misc::VolumetricMetadata_t& md);

        /**
         * Increment the cache ring index.
         */
        static inline std::size_t increment(std::size_t frameIdx) {
            return (++frameIdx % 2);
        }

        /**
         * Compute the difference from 'prev' to 'cur' into 'dst'.
         */
        template<class D, class S>
        void calcDifference(D *dst, const S *cur, const S *prev,
            const std::size_t cnt);

        /**
         * Check whether the given metadata are compatible with the cache state
         * of the module.
         */
        bool checkCompatibility(const core::misc::VolumetricMetadata_t& md) const;

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Computes the hash of 'data'.
         *
         * @return The hash of the currently available data.
         */
        inline std::size_t getHash(void) {
            auto retval = this->hashData;
            retval ^= this->hashState + 0x9e3779b9 + (retval << 6)
                + (retval >> 2);
            return retval;
        }

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool onGetData(core::Call& call);

        /**
         * Gets the data extents.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool onGetExtents(core::Call& call);

        /**
         * Gets the meta data.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool onGetMetadata(core::Call& call);

        /**
         * Callback for all unsupported operations.
         */
        bool onUnsupported(core::Call& call);

        /**
         * Clean up module.
         */
        virtual void release(void) override;

    private:

        std::array<std::vector<std::uint8_t>, 2> cache;
        std::vector<std::uint8_t> data;
        unsigned int frameID;
        std::size_t frameIdx;
        std::size_t hashData;
        std::size_t hashState;
        core::misc::VolumetricMetadataStore metadata;
        core::param::ParamSlot paramIgnoreInputHash;
        core::CallerSlot slotIn;
        core::CalleeSlot slotOut;

    };

} /* end namespace volume */
} /* namespace stdplugin */
} /* namespace megamol */

#include "DifferenceVolume.inl"
