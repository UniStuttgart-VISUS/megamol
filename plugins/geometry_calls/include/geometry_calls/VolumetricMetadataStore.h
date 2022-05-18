/*
 * VolumetricMetadataStore.h
 *
 * Copyright (C) 2019 by Visualisierungsinstitut der Universität Stuttgart.
 * Alle rechte vorbehalten.
 */

#pragma once

#include <array>
#include <vector>

#include "geometry_calls/VolumetricDataCallTypes.h"


namespace megamol::geocalls {

/**
 * A self-contained storage class for VolumetricMetadata_t to be used if the
 * metadata are not obtained from the datRaw library.
 */
class VolumetricMetadataStore : public VolumetricMetadata_t {

public:
    VolumetricMetadataStore(void) = default;

    VolumetricMetadataStore(const VolumetricMetadataStore& rhs);

    VolumetricMetadataStore(VolumetricMetadataStore&& rhs) noexcept;

    VolumetricMetadataStore(const VolumetricMetadata_t& rhs);

    VolumetricMetadataStore& operator=(const VolumetricMetadataStore& rhs);

    VolumetricMetadataStore& operator=(VolumetricMetadataStore&& rhs) noexcept;

    VolumetricMetadataStore& operator=(const VolumetricMetadata_t& rhs);

private:
    void rewrire(void);

    std::vector<double> maxValues;
    std::vector<double> minValues;
    std::array<std::vector<float>, 3> sliceDists;
};

} // namespace megamol::geocalls
