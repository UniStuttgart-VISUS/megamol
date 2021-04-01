#pragma once

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

    void const* get_record() const {
        return _sbt_record;
    }

    void set_record(void const* record) {
        _sbt_record = record;
    }

    std::size_t get_record_stride() const {
        return _sbt_record_stride;
    }

    void set_record_stride(std::size_t size) {
        _sbt_record_stride = size;
    }

    int get_num_records() const {
        return _sbt_num_records;
    }

    void set_num_records(int count) {
        _sbt_num_records = count;
    }

    OptixProgramGroup const* get_program_groups() const {
        return _geo_programs;
    }

    void set_program_groups(OptixProgramGroup const* groups) {
        _geo_programs = groups;
    }

    int get_num_programs() const {
        return _num_programs;
    }

    void set_num_programs(int count) {
        _num_programs = count;
    }

private:
    Context const* _ctx;

    OptixTraversableHandle const* _geo_handle;

    void const* _sbt_record;

    std::size_t _sbt_record_stride;

    int _sbt_num_records;

    OptixProgramGroup const* _geo_programs;

    int _num_programs;
};

using CallGeometryDescription = megamol::core::factories::CallAutoDescription<CallGeometry>;

} // namespace megamol::optix_hpg
