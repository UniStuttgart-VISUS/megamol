/*
 * DifferenceVolume.inl
 *
 * Copyright (C) 2019 by Visualisierungsinstitut der Universität Stuttgart.
 * Alle Rechte vorbehalten.
 */


/*
 * megamol::volume::DifferenceVolume::calcDifference
 */
template<class D, class S>
void megamol::volume::DifferenceVolume::calcDifference(D* dst, const S* cur, const S* prev, const std::size_t cnt) {
    //#pragma omp parallel for
    //    for (std::size_t i = 0; i < cnt; ++i) {
    //        dst[i] = static_cast<D>(cur[i]) - static_cast<D>(prev[i]);
    //    }

    const auto components = this->metadata.Components;

    for (std::size_t i = 0; i < components; ++i) {
        this->metadata.MinValues[i] = (std::numeric_limits<double>::max)();
        this->metadata.MaxValues[i] = std::numeric_limits<double>::lowest();
    }

    for (std::size_t i = 0; i < cnt; ++i) {
        dst[i] = static_cast<D>(cur[i]) - static_cast<D>(prev[i]);
        if (dst[i] < this->metadata.MinValues[i % components]) {
            this->metadata.MinValues[i % components] = dst[i];
        }
        if (dst[i] > this->metadata.MaxValues[i % components]) {
            this->metadata.MaxValues[i % components] = dst[i];
        }
    }
}
