/*
 * ProbeCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#ifndef PROBE_COLLECTION_H_INCLUDED
#define PROBE_COLLECTION_H_INCLUDED

#include <array>
#include <random>
#include <string>
#include <variant>

namespace megamol {
namespace probe {

struct BaseProbe {
    /** time at which this probes samples the data */
    size_t m_timestamp;
    /** semantic name of the values/field that this probe samples */
    std::string m_value_name;
    /** position of probe head on surface */
    std::array<float, 3> m_position;
    /** probe insertion/sampling direction */
    std::array<float, 3> m_direction;
    /** "sample from" offset from position */
    float m_begin;
    /** "sample to" offset from position */
    float m_end;
    // std::vector<size_t>m_sample_idxs; ///< indices of samples relevant to this
    /** sample radius used by this probe */
    float m_sample_radius;
    /** for clustered samples */
    int m_cluster_id;
    /** true, if clustering considers this probe to be a representant */
    bool m_representant = false;
    /** string id of the meshes that the probe goes through */
    std::vector<std::string> m_geo_ids;
    /** string id of the meshes that the probe goes through */
    std::vector<uint64_t> m_vert_ids;

    // virtual void probe() = 0;
};

struct FloatProbe : public BaseProbe {
public:
    struct SamplingResult {
        std::vector<float> samples;
        float min_value;
        float max_value;
        float average_value;
    };

    FloatProbe() : m_result(std::make_shared<SamplingResult>()) {}

    template<typename DatafieldType>
    void probe(DatafieldType const& datafield) { /* ToDo*/
    }

    std::shared_ptr<SamplingResult> getSamplingResult() const {
        return m_result;
    }

private:
    std::shared_ptr<SamplingResult> m_result;
};

struct FloatDistributionProbe : public BaseProbe {
public:
    struct SampleValue {
        float mean;
        float lower_bound;
        float upper_bound;
    };

    struct SamplingResult {
        std::vector<SampleValue> samples;
        float min_value;
        float max_value;
        float average_value;
    };

    FloatDistributionProbe() : m_result(std::make_shared<SamplingResult>()) {}

    template<typename DatafieldType>
    void probe(DatafieldType const& datafield) { /* ToDo*/
    }

    std::shared_ptr<SamplingResult> getSamplingResult() const {
        return m_result;
    }

private:
    std::shared_ptr<SamplingResult> m_result;
};

struct IntProbe : public BaseProbe {
    void probe() { /*ToDo*/
    }
};

struct Vec4Probe : public BaseProbe {
public:
    struct SamplingResult {
        std::vector<std::array<float, 4>> samples;
    };

    Vec4Probe() : m_result(std::make_shared<SamplingResult>()) {}

    template<typename DatafieldType>
    void probe(DatafieldType const& datafield) { /* ToDo*/
    }

    std::shared_ptr<SamplingResult> getSamplingResult() const {
        return m_result;
    }

private:
    std::shared_ptr<SamplingResult> m_result;
};

using GenericProbe = std::variant<FloatProbe, IntProbe, Vec4Probe, BaseProbe, FloatDistributionProbe>;
using GenericMinMax = std::variant<std::array<float, 2>, std::array<int, 2>>;

class ProbeCollection {
public:
    ProbeCollection() = default;
    ~ProbeCollection() = default;

    template<typename ProbeType>
    void addProbe(ProbeType const& probe) {
        m_probes.push_back(probe);
    }

    template<typename ProbeType>
    void setProbe(size_t idx, ProbeType const& probe) {
        m_probes[idx] = probe;
    }

    template<typename ProbeType>
    ProbeType getProbe(size_t idx) const {
        return std::get<ProbeType>(m_probes[idx]);
    }

    GenericProbe getGenericProbe(size_t idx) const {
        return m_probes[idx];
    }

    const BaseProbe& getBaseProbe(size_t idx) const {
        const BaseProbe& x = std::visit([](const auto& x) -> const BaseProbe& { return x; }, m_probes[idx]);
        return x;
    }

    uint32_t getProbeCount() const {
        return m_probes.size();
    }

    template<typename T>
    void setGlobalMinMax(T min_, T max_) {
        m_global_min_max = std::array<T, 2>({min_, max_});
    }

    template<typename T>
    std::array<T, 2> getGlobalMinMax() {
        return std::get<std::array<T, 2>>(m_global_min_max);
    }

    GenericMinMax getGenericGlobalMinMax() {
        return m_global_min_max;
    }

    void erase_probes(std::vector<char> const& indicator) {
        if (indicator.size() != m_probes.size())
            return;
        auto const num_el = std::count_if(indicator.begin(), indicator.end(), [](auto const el) { return el == 0; });
        std::vector<GenericProbe> tmp;
        tmp.reserve(num_el);
        for (std::vector<GenericProbe>::size_type idx = 0; idx < m_probes.size(); ++idx) {
            if (indicator[idx] == 0) {
                tmp.push_back(m_probes[idx]);
            }
        }
        m_probes = tmp;
    }

    void shuffle_probes() {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(m_probes.begin(), m_probes.end(), g);
    }

private:
    std::vector<GenericProbe> m_probes;
    GenericMinMax m_global_min_max;
};


} // namespace probe
} // namespace megamol

#endif // !PROBE_COLLECTION_H_INCLUDED
