#include "mmcore/CallProfiling.h"
#include "mmcore/Call.h"

using namespace megamol;
using namespace core;

CallProfiling::CallProfiling() {}

CallProfiling::~CallProfiling() {}

uint32_t CallProfiling::GetSampleHistoryLength() {
    return PerformanceHistory::buffer_length;
}

double CallProfiling::GetLastCPUTime(uint32_t func) const {
    if (func < parent_call->GetCallbackCount())
        return cpu_history[func].last_value();
    else
        return -1.0;
}

double CallProfiling::GetAverageCPUTime(uint32_t func) const {
    if (func < parent_call->GetCallbackCount())
        return cpu_history[func].average();
    else
        return -1.0;
}

uint32_t CallProfiling::GetNumCPUSamples(uint32_t func) const {
    if (func < parent_call->GetCallbackCount())
        return cpu_history[func].samples();
    else
        return 0;
}

std::array<double, PerformanceHistory::buffer_length> CallProfiling::GetCPUHistory(uint32_t func) const {
    if (func < parent_call->GetCallbackCount())
        return cpu_history[func].copyHistory();
    else
        return std::array<double, PerformanceHistory::buffer_length>{};
}

double CallProfiling::GetLastGPUTime(uint32_t func) const {
    if (func < parent_call->GetCallbackCount())
        return gpu_history[func].last_value();
    else
        return -1.0;
}

double CallProfiling::GetAverageGPUTime(uint32_t func) const {
    if (func < parent_call->GetCallbackCount())
        return gpu_history[func].average();
    else
        return -1.0;
}

uint32_t CallProfiling::GetNumGPUSamples(uint32_t func) const {
    if (func < parent_call->GetCallbackCount())
        return gpu_history[func].samples();
    else
        return 0;
}

std::array<double, PerformanceHistory::buffer_length> CallProfiling::GetGPUHistory(uint32_t func) const {
    if (func < parent_call->GetCallbackCount())
        return gpu_history[func].copyHistory();
    else
        return std::array<double, PerformanceHistory::buffer_length>{};
}

void CallProfiling::SetParent(Call* parent) {
    cpu_history.resize(parent->GetCallbackCount());
    gpu_history.resize(parent->GetCallbackCount());
    parent_call = parent;
    if (parent_call->GetCapabilities().OpenGLRequired()) {
        InitializeQueryManager();
        qm->AddCall(parent_call);
    }
}

void CallProfiling::InitializeQueryManager() {
    if (qm == nullptr) {
        qm = new PerformanceQueryManager();
    }
}

void CallProfiling::CollectGPUPerformance() {
    if (qm != nullptr) {
        qm->Collect();
    }
}

void CallProfiling::ShutdownProfiling() const {
    if (qm != nullptr) {
        qm->RemoveCall(parent_call);
    }
}
