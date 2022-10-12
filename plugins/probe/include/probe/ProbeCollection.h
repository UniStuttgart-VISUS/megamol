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
#include <mmcore/utility/log/Log.h>

namespace megamol {
namespace probe {

struct BaseProbe {
    enum PlacementMethod {CENTERLINE, CENTERPOINT, VERTEX_NORMAL, FACE_NORMAL, UNKNOWN};
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
    /** original end of probe in case the probe was shortened */
    float m_orig_end;
    // std::vector<size_t>m_sample_idxs; ///< indices of samples relevant to this
    /** sample radius used by this probe */
    float m_sample_radius;
    /** for clustered samples */
    int m_cluster_id;
    /** true, if clustering considers this probe to be a representant */
    bool m_representant = false;
    /** string id of the meshes that the probe goes through */
    std::vector<std::string> m_geo_ids;
    /** vertex id on the rendered mesh the probe is placed */
    std::vector<uint64_t> m_vert_ids;
    /** vertex id on the original mesh the probe is placed */
    std::vector<uint64_t> m_face_vert_ids;
    /** saves the placement method used to create probe */
    PlacementMethod m_placement_method = UNKNOWN;

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
        std::vector<float> values;
        std::vector<float> value_depth;
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

struct VectorDistributionProbe : public BaseProbe {
public:
    struct SampleValue {
        std::vector<std::array<float,3>> norm_directions;
        std::vector<float> magnitudes;
        std::vector<std::array<float,3>> positions;
    };

    struct SamplingResult {
        std::vector<SampleValue> samples;
    };

    VectorDistributionProbe() : m_result(std::make_shared<SamplingResult>()) {}

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
using GenericMinMax = std::variant<std::array<double, 2>, std::array<float, 2>, std::array<int, 2>>;

class ProbeCollection {
public:
    struct ProbeLevel {
        // indices of sub_probes correspond to previous level indices
        std::vector<std::vector<size_t>> sub_probes;
        // indices of super_probes correspond to next level indices
        std::vector<size_t> super_probes;
        std::vector<GenericProbe> probes;
        int level_index = -1;
    };

    ProbeCollection() = default;
    ~ProbeCollection() = default;

    template<typename ProbeType>
    void addProbe(ProbeType const& probe) {
        lod_collection[max_level].probes.push_back(probe);
    }

    template<typename ProbeType>
    void setProbe(size_t idx, ProbeType const& probe) {
        lod_collection[max_level].probes[idx] = probe;
    }

    template<typename ProbeType>
    ProbeType getProbe(size_t idx) const {
        return std::get<ProbeType>(lod_collection[max_level].probes[idx]);
    }

    GenericProbe getGenericProbe(size_t idx) const {
        return lod_collection[max_level].probes[idx];
    }

    BaseProbe const& getBaseProbe(size_t idx) const {
        const BaseProbe& x = std::visit([](const auto& x) -> const BaseProbe& {
            return x; }, lod_collection[max_level].probes[idx]);
        return x;
    }

    uint32_t getProbeCount() const {
        return lod_collection[max_level].probes.size();
    }

    template<typename T>
    void setGlobalMinMax(T min_, T max_) {
        m_global_min_max = std::array<T, 2>({min_, max_});
    }

    template<typename T>
    std::array<T, 2> getGlobalMinMax() {
        return std::get<std::array<T, 2>>(m_global_min_max);
    }

    GenericMinMax getGenericGlobalMinMax() const {
        return m_global_min_max;
    }

    void eraseProbes(std::vector<char> const& indicator) {
        if (indicator.size() != lod_collection[max_level].probes.size())
            return;
        auto const num_el = std::count_if(indicator.begin(), indicator.end(), [](auto const el) { return el == 0; });
        std::vector<GenericProbe> tmp;
        tmp.reserve(num_el);
        for (std::vector<GenericProbe>::size_type idx = 0; idx < lod_collection[max_level].probes.size(); ++idx) {
            if (indicator[idx] == 0) {
                tmp.emplace_back(lod_collection[max_level].probes[idx]);
            }
        }
        lod_collection[max_level].probes = tmp;
    }

    void shuffleProbes() {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(lod_collection[max_level].probes.begin(), lod_collection[max_level].probes.end(), g);
    }

    bool setLevel(int const idx, ProbeLevel const level) {
        if (idx >= lod_collection.size()) {
            core::utility::log::Log::DefaultLog.WriteError("[ProbeLoDCollection] Level index too large.");
            return false;
        }
        if (level.super_probes.empty() && level.sub_probes.empty()) {
            core::utility::log::Log::DefaultLog.WriteError("[ProbeLoDCollection] Level not complete.");
            return false;
        }
        lod_collection[idx] = level;
        return true;
    }

    ProbeLevel getLevel(int const idx) {
        if (idx >= lod_collection.size()) {
            core::utility::log::Log::DefaultLog.WriteError("[ProbeLoDCollection] Probe level does not exist.");
            return ProbeLevel{};
        }
        return lod_collection[idx];
    }

    template<typename ProbeType>
    ProbeType getSuperProbe(int const level, size_t const idx) const {
        if (level - 1 < 0) {
            core::utility::log::Log::DefaultLog.WriteError("[ProbeLoDCollection] Level for get super probe too low.");
            return ProbeType();
        }
        auto const super_index = lod_collection[level].super_probes[idx];
        return std::get<ProbeType>(lod_collection[level - 1].probes[super_index]);
    }

    template<typename ProbeType>
    std::vector<ProbeType> getSubProbes(int const level, size_t const idx) const {
        if (level + 1 >= lod_collection.size()) {
            core::utility::log::Log::DefaultLog.WriteError("[ProbeLoDCollection] Level for get sub probe too high.");
            return nullptr;
        }
        auto const sub_indices = lod_collection[level].sub_probes[idx];
        std::vector<ProbeType> sub_probes;
        sub_probes.reserve(sub_indices.size());
        for (auto const index : sub_indices) {
            sub_probes.emplace_back(std::get<ProbeType>(lod_collection[level + 1].probes[index]));
        }
        return sub_probes;
    }

    int getNumLevels() const {
        return lod_collection.size();
    }

    int getActiveLevels() {
        int i = 0;
        for (auto const& level : lod_collection) {
            if (!level.probes.empty()) {
                i++;
            }
        }
        return i;
    }


private:
    std::array<ProbeLevel, 4> lod_collection;
    int max_level = 3;
    GenericMinMax m_global_min_max;

};

} // namespace probe
} // namespace megamol

#endif // !PROBE_COLLECTION_H_INCLUDED
