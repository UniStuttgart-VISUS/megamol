#pragma once

#include <vector>

#include "mmcore/CallGeneric.h"

namespace megamol::stdplugin::datatools {
struct StatisticsMetaData {
    unsigned int m_frame_cnt = 0;
    unsigned int m_frame_ID = 0;
};

struct StatisticsData {
    float avg_val;
    float med_val;
    float min_val;
    float max_val;
    std::vector<float> histo;
};

class StatisticsCall : public core::GenericVersionedCall<std::vector<StatisticsData>, StatisticsMetaData> {
public:
    StatisticsCall() = default;
    ~StatisticsCall() = default;

    static const char* ClassName(void) {
        return "StatisticsCall";
    }
    static const char* Description(void) {
        return "Call transporting statistical data";
    }
};

using StatisticsCallDescription = megamol::core::factories::CallAutoDescription<StatisticsCall>;
} // namespace megamol::stdplugin::datatools
