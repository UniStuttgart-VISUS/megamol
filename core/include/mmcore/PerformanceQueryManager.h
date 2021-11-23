#pragma once

#include <array>
#include <vector>

namespace megamol {
namespace core {
class Call;

class PerformanceQueryManager {
public:
    PerformanceQueryManager();
    ~PerformanceQueryManager();
    PerformanceQueryManager(const PerformanceQueryManager&);

    bool Start(megamol::core::Call* c, uint32_t frameId, int32_t funcIdx);
    void Stop(uint32_t frameId);
    void Collect();

    void AddCall(megamol::core::Call* c);
    void RemoveCall(megamol::core::Call* c);

    void ResetGLProfiling();
    void AdvanceGLProfiling();

private:
    struct query_info {
        uint32_t id = 0;
        bool started = false;
        int32_t call_idx = -1;
        int32_t func_idx = -1;
    };
    std::array<query_info, 2> query_infos;
    std::vector<megamol::core::Call*> all_calls;
    int32_t starting_call = -1, starting_func = -1;

    int32_t next_query = 0;
    int32_t running_query = -1;
};


} // namespace core
} // namespace megamol
