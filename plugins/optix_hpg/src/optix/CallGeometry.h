#pragma once

#include <tuple>

#include "mmcore/AbstractGetData3DCall.h"

#include "optix.h"

#include "optix/Context.h"

namespace megamol::optix_hpg {
class CallGeometry : public core::AbstractGetData3DCall {
public:
    static const char* ClassName(void) {
        return "CallGeometry";
    }

    static const char* Description(void) {
        return "Transports an OptiX geometry";
    }

    static unsigned int FunctionCount(void) {
        return core::AbstractGetData3DCall::FunctionCount();
    }

    static const char* FunctionName(unsigned int idx) {
        return core::AbstractGetData3DCall::FunctionName(idx);
    }

    Context const* get_ctx() const {
        return _ctx;
    }

    void set_ctx(Context const* ctx) {
        _ctx = ctx;
    }

    OptixTraversableHandle const* get_handle() const {
        return _geo_handle;
    }

    void set_handle(OptixTraversableHandle const* handle) {
        _geo_handle = handle;
    }

    std::tuple<void const*, uint32_t, uint64_t> get_record() const {
        get_sbt_update = set_sbt_update;
        return std::make_tuple(_sbt_record, _sbt_num_records, _sbt_record_stride);
    }

    void set_record(void const* record, uint32_t num_records, uint64_t record_stride, uint64_t version) {
        _sbt_record = record;
        _sbt_num_records = num_records;
        _sbt_record_stride = record_stride;
        set_sbt_update = version;
    }

    bool has_sbt_update() const {
        return get_sbt_update < set_sbt_update;
    }

    std::tuple<OptixProgramGroup const*, uint32_t> get_program_groups() const {
        get_program_update = set_program_update;
        return std::make_tuple(_geo_programs, _num_programs);
    }

    void set_program_groups(OptixProgramGroup const* groups, uint32_t num_progams, uint64_t version) {
        _geo_programs = groups;
        _num_programs = num_progams;
        set_program_update = version;
    }

    bool has_program_update() const {
        return get_program_update < set_program_update;
    }

private:
    Context const* _ctx;

    OptixTraversableHandle const* _geo_handle;

    void const* _sbt_record;

    std::size_t _sbt_record_stride;

    uint32_t _sbt_num_records;

    OptixProgramGroup const* _geo_programs;

    uint32_t _num_programs;

    uint64_t set_program_update;
    mutable uint64_t get_program_update;

    uint64_t set_sbt_update;
    mutable uint64_t get_sbt_update;
};

using CallGeometryDescription = megamol::core::factories::CallAutoDescription<CallGeometry>;

} // namespace megamol::optix_hpg
