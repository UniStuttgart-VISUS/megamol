/*
 * ProbeCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#ifndef PROBE_COLLECTION_H_INCLUDED
#define PROBE_COLLECTION_H_INCLUDED

#include "probe/probe.h"

#include <array>
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

    template <typename DatafieldType> void probe(DatafieldType const& datafield) { /* ToDo*/ }

    std::shared_ptr<SamplingResult> getSamplingResult() const { return m_result; }

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
        std::vector<std::array<float,4>> samples;
    };

    Vec4Probe() : m_result(std::make_shared<SamplingResult>()) {}

    template <typename DatafieldType> void probe(DatafieldType const& datafield) { /* ToDo*/ }

    std::shared_ptr<SamplingResult> getSamplingResult() const { return m_result; }

private:
    std::shared_ptr<SamplingResult> m_result;
};

using GenericProbe = std::variant<FloatProbe, IntProbe, Vec4Probe, BaseProbe>;


class ProbeCollection {
public:
    ProbeCollection() = default;
    ~ProbeCollection() = default;

    template <typename ProbeType> void addProbe(ProbeType const& probe) { m_probes.push_back(probe); }

    template <typename ProbeType> void setProbe(size_t idx, ProbeType const& probe) { m_probes[idx] = probe; }

    template <typename ProbeType> ProbeType getProbe(size_t idx) const { return std::get<ProbeType>(m_probes[idx]); }

    GenericProbe getGenericProbe(size_t idx) const { return m_probes[idx]; }

    uint32_t getProbeCount() const { return m_probes.size(); }

private:
    std::vector<GenericProbe> m_probes;
};


} // namespace probe
} // namespace megamol

#endif // !PROBE_COLLECTION_H_INCLUDED
