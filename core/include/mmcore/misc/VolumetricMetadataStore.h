/*
 * VolumetricMetadataStore.h
 *
 * Copyright (C) 2019 by Visualisierungsinstitut der Universität Stuttgart.
 * Alle rechte vorbehalten.
 */

#pragma once

#include <array>
#include <vector>

#include "mmcore/api/MegaMolCore.std.h"

#include "mmcore/misc/VolumetricDataCallTypes.h"


namespace megamol {
namespace core {
namespace misc {

    /**
     * A self-contained storage class for VolumetricMetadata_t to be used if the
     * metadata are not obtained from the datRaw library.
     */
    class MEGAMOLCORE_API VolumetricMetadataStore : public VolumetricMetadata_t {

    public:

        VolumetricMetadataStore(void) = default;

        VolumetricMetadataStore(const VolumetricMetadataStore& rhs);

        VolumetricMetadataStore(VolumetricMetadataStore&& rhs) noexcept;

        VolumetricMetadataStore(const VolumetricMetadata_t& rhs);

        VolumetricMetadataStore& operator =(const VolumetricMetadataStore& rhs);

        VolumetricMetadataStore& operator =(VolumetricMetadataStore&& rhs) noexcept;

        VolumetricMetadataStore& operator =(const VolumetricMetadata_t& rhs);

    private:

        void rewrire(void);

        std::vector<double >maxValues;
        std::vector<double> minValues;
        std::array<std::vector<float>, 3> sliceDists;

    };

} /* end namespace misc */
} /* end namespace core */
} /* end namespace megamol */
