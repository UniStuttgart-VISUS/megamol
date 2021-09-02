#pragma once

#include <vector>
#include <array>
#include <map>
#include <string>

#include "PerformanceQueryManager.h"
#include "PerformanceHistory.h"

namespace megamol {
namespace core {

    class Call;

    class CallProfiling {
    public:

        CallProfiling();
        ~CallProfiling();

        static uint32_t GetSampleHistoryLength();

        double GetLastCPUTime(uint32_t func) const;
        double GetAverageCPUTime(uint32_t func) const;
        uint32_t GetNumCPUSamples(uint32_t func) const;
        std::array<double, PerformanceHistory::buffer_length> GetCPUHistory(uint32_t func) const;

        double GetLastGPUTime(uint32_t func) const;
        double GetAverageGPUTime(uint32_t func) const;
        uint32_t GetNumGPUSamples(uint32_t func) const;
        std::array<double, PerformanceHistory::buffer_length> GetGPUHistory(uint32_t func) const;

        uint32_t GetFuncCount() const;
        const std::string& GetFuncName(uint32_t i) const;

        static void InitializeQueryManager();
        static void CollectGPUPerformance();

    private:
        friend class Call;
        friend class PerformanceQueryManager;

        void setProfilingInfo(std::vector<std::string> names, Call *parent);

        std::vector<PerformanceHistory> cpu_history;
        std::vector<PerformanceHistory> gpu_history;
        std::vector<std::string> callback_names;
        Call *parent_call = nullptr;
        static PerformanceQueryManager *qm;
        static std::string err_oob;
    };

}
}
