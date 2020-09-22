/*
 * VolumetricMetadataStore.cpp
 *
 * Copyright (C) 2019 by Visualisierungsinstitut der Universität Stuttgart.
 * Alle rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/misc/VolumetricMetadataStore.h"

#include <cassert>
#include <iterator>


/*
 * megamol::core::misc::VolumetricMetadataStore::VolumetricMetadataStore
 */
megamol::core::misc::VolumetricMetadataStore::VolumetricMetadataStore(
        const VolumetricMetadataStore& rhs)
    : maxValues(rhs.maxValues), minValues(rhs.minValues),
        sliceDists(rhs.sliceDists) {
    this->rewrire();
}


/*
 * megamol::core::misc::VolumetricMetadataStore::VolumetricMetadataStore
 */
megamol::core::misc::VolumetricMetadataStore::VolumetricMetadataStore(
        VolumetricMetadataStore&& rhs) noexcept
    : maxValues(std::move(rhs.maxValues)), minValues(std::move(rhs.minValues)),
        sliceDists(std::move(rhs.sliceDists)) {
    this->rewrire();
    rhs.rewrire();
    assert(rhs.MinValues == nullptr);
    assert(rhs.MaxValues == nullptr);
}


/*
 * megamol::core::misc::VolumetricMetadataStore::VolumetricMetadataStore
 */
megamol::core::misc::VolumetricMetadataStore::VolumetricMetadataStore(
        const VolumetricMetadata_t& rhs) {
    *this = rhs;
}


/*
 * megamol::core::misc::VolumetricMetadataStore::operator =
 */
megamol::core::misc::VolumetricMetadataStore&
megamol::core::misc::VolumetricMetadataStore::operator =(
        const VolumetricMetadataStore& rhs) {
    if (this != std::addressof(rhs)) {
        ::memcpy(this, std::addressof(rhs), sizeof(VolumetricMetadata_t));
        this->rewrire();
    }
    return *this;
}


/*
 * megamol::core::misc::VolumetricMetadataStore::operator =
 */
megamol::core::misc::VolumetricMetadataStore&
megamol::core::misc::VolumetricMetadataStore::operator =(
        VolumetricMetadataStore&& rhs) noexcept {
    if (this != std::addressof(rhs)) {
        ::memcpy(this, std::addressof(rhs), sizeof(VolumetricMetadata_t));
        this->rewrire();
        rhs.rewrire();
        assert(rhs.MinValues == nullptr);
        assert(rhs.MaxValues == nullptr);
    }
    return *this;
}


/*
 * megamol::core::misc::VolumetricMetadataStore::operator =
 */
megamol::core::misc::VolumetricMetadataStore&
megamol::core::misc::VolumetricMetadataStore::operator =(
        const VolumetricMetadata_t& rhs) {
    if (this != std::addressof(rhs)) {
        ::memcpy(this, std::addressof(rhs), sizeof(VolumetricMetadata_t));

        this->maxValues.clear();
        this->maxValues.resize(this->Components);
        for (std::size_t i = 0; i < this->maxValues.size(); ++i) {
            this->maxValues[i] = rhs.MaxValues[i];
        }

        this->minValues.clear();
        this->minValues.resize(this->Components);
        for (std::size_t i = 0; i < this->minValues.size(); ++i) {
            this->minValues[i] = rhs.MinValues[i];
        }

        switch (this->GridType) {
            case core::misc::CARTESIAN:
                for (std::size_t i = 0; i < this->sliceDists.size(); ++i) {
                    this->sliceDists[i].resize(1);
                    this->sliceDists[i][0] = rhs.SliceDists[i][0];
                }
                break;

            case core::misc::RECTILINEAR:
                for (std::size_t i = 0; i < this->sliceDists.size(); ++i) {
                    this->sliceDists[i].clear();
                    auto cnt = rhs.Resolution[i] - 1;
                    this->sliceDists[i].reserve(cnt);
                    std::copy(rhs.SliceDists[i], rhs.SliceDists[i] + cnt,
                        std::back_inserter(this->sliceDists[i]));
                }
                break;

            default:
                for (auto& s : this->sliceDists) {
                    s.clear();
                }
                break;
        }

        this->rewrire();
    }

    return *this;
}


/*
 * megamol::core::misc::VolumetricMetadataStore::rewrire
 */
void megamol::core::misc::VolumetricMetadataStore::rewrire(void) {
    this->MaxValues = this->maxValues.empty()
        ? nullptr
        : this->maxValues.data();
    this->MinValues = this->minValues.empty()
        ? nullptr
        : this->minValues.data();
    for (std::size_t i = 0; i < this->sliceDists.size(); ++i) {
        this->SliceDists[i] = this->sliceDists[i].empty()
            ? nullptr :
            this->sliceDists[i].data();
    }
}
