/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd/flags/FlagStorage.h"

#include <json.hpp>

#include "FlagStorageBitsChecker.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/StringParam.h"
#include "mmstd/flags/FlagCalls.h"

using namespace megamol;
using namespace megamol::core;


FlagStorage::FlagStorage()
        : readCPUFlagsSlot("readCPUFlags", "Provides flag data to clients.")
        , writeCPUFlagsSlot("writeCPUFlags", "Accepts updated flag data from clients.")
        , skipFlagsSerializationParam("skipFlagsSerialization", "Disable serialization of flags.")
        , serializedFlags("serializedFlags", "persists the flags in projects") {

    this->readCPUFlagsSlot.SetCallback(FlagCallRead_CPU::ClassName(),
        FlagCallRead_CPU::FunctionName(FlagCallRead_CPU::CallGetData), &FlagStorage::readCPUDataCallback);
    this->readCPUFlagsSlot.SetCallback(FlagCallRead_CPU::ClassName(),
        FlagCallRead_CPU::FunctionName(FlagCallRead_CPU::CallGetMetaData), &FlagStorage::readMetaDataCallback);
    this->MakeSlotAvailable(&this->readCPUFlagsSlot);

    this->writeCPUFlagsSlot.SetCallback(FlagCallWrite_CPU::ClassName(),
        FlagCallWrite_CPU::FunctionName(FlagCallWrite_CPU::CallGetData), &FlagStorage::writeCPUDataCallback);
    this->writeCPUFlagsSlot.SetCallback(FlagCallWrite_CPU::ClassName(),
        FlagCallWrite_CPU::FunctionName(FlagCallWrite_CPU::CallGetMetaData), &FlagStorage::writeMetaDataCallback);
    this->MakeSlotAvailable(&this->writeCPUFlagsSlot);

    this->skipFlagsSerializationParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->skipFlagsSerializationParam);

    this->serializedFlags << new core::param::StringParam("");
    this->serializedFlags.SetUpdateCallback(&FlagStorage::onJSONChanged);
    this->MakeSlotAvailable(&this->serializedFlags);
}


FlagStorage::~FlagStorage() {
    this->Release();
};


bool FlagStorage::create() {
    const int num = 1;

    this->theCPUData = std::make_shared<FlagCollection_CPU>();
    this->theCPUData->flags = std::make_shared<FlagStorageTypes::flag_vector_type>(
        num, FlagStorageTypes::to_integral(FlagStorageTypes::flag_bits::ENABLED));

    return true;
}


void FlagStorage::release() {
    // intentionally empty
}


bool FlagStorage::readCPUDataCallback(core::Call& caller) {
    auto fc = dynamic_cast<FlagCallRead_CPU*>(&caller);
    if (fc == nullptr)
        return false;

    fc->setData(this->theCPUData, this->version);
    return true;
}


bool FlagStorage::writeCPUDataCallback(core::Call& caller) {
    auto fc = dynamic_cast<FlagCallWrite_CPU*>(&caller);
    if (fc == nullptr)
        return false;

    if (fc->version() > this->version) {
        this->theCPUData = fc->getData();
        this->version = fc->version();
        serializeCPUData();
    }
    return true;
}


bool FlagStorage::readMetaDataCallback(core::Call& caller) {
    return true;
}

bool FlagStorage::writeMetaDataCallback(core::Call& caller) {
    return true;
}


nlohmann::json FlagStorage::make_bit_array(
    const FlagStorageTypes::index_vector& bit_starts, const FlagStorageTypes::index_vector& bit_ends) {
    auto the_array = nlohmann::json::array();
    for (uint32_t x = 0; x < bit_starts.size(); ++x) {
        const auto& s = bit_starts[x];
        const auto& e = bit_ends[x];
        if (s == e) {
            the_array.push_back(s);
        } else {
            the_array.push_back(nlohmann::json::array({s, e}));
        }
    }
    return the_array;
}

void FlagStorage::array_to_bits(const nlohmann::json& json, FlagStorageTypes::flag_bits flag_bit) {
    for (auto& j : json) {
        if (j.is_array()) {
            FlagStorageTypes::index_type from, to;
            j[0].get_to(from);
            j[1].get_to(to);
            for (FlagStorageTypes::index_type x = from; x <= to; ++x) {
                (*theCPUData->flags)[x] |= FlagStorageTypes::to_integral(flag_bit);
            }
        } else {
            FlagStorageTypes::index_type idx;
            j.get_to(idx);
            (*theCPUData->flags)[idx] |= FlagStorageTypes::to_integral(flag_bit);
        }
    }
}

FlagStorageTypes::index_type FlagStorage::array_max(const nlohmann::json& json) {
    if (json.empty())
        return 0;
    auto& j = json.back();
    FlagStorageTypes::index_type end = 0;
    if (j.is_array()) {
        j[1].get_to(end);
    } else {
        j.get_to(end);
    }
    return end;
}


void FlagStorage::serializeCPUData() {
    if (skipFlagsSerializationParam.Param<core::param::BoolParam>()->Value()) {
        return;
    }

    const auto& cdata = theCPUData->flags;

    FlagStorageTypes::index_vector enabled_starts, enabled_ends;
    FlagStorageTypes::index_vector filtered_starts, filtered_ends;
    FlagStorageTypes::index_vector selected_starts, selected_ends;
    FlagStorageTypes::index_type curr_enabled_start = -1, curr_filtered_start = -1, curr_selected_start = -1;

#if 0 // serial version
    const auto startSerialTime = std::chrono::high_resolution_clock::now();
    for (index_type x = 0; x < cdata->size(); ++x) {
        check_bits(FlagStorage::ENABLED, enabled_starts, enabled_ends, curr_enabled_start, x, cdata);
        check_bits(FlagStorage::FILTERED, filtered_starts, filtered_ends, curr_filtered_start, x, cdata);
        check_bits(FlagStorage::SELECTED, selected_starts, selected_ends, curr_selected_start, x, cdata);
    }
    terminate_bit(cdata, enabled_ends, curr_enabled_start);
    terminate_bit(cdata, filtered_ends, curr_filtered_start);
    terminate_bit(cdata, selected_ends, curr_selected_start);
    const auto endSerialTime = std::chrono::high_resolution_clock::now();
    ASSERT(enabled_starts.size() == enabled_ends.size());
    ASSERT(filtered_starts.size() == filtered_ends.size());
    ASSERT(selected_starts.size() == selected_ends.size());
    nlohmann::json ser_data;
    ser_data["enabled"] = make_bit_array(enabled_starts, enabled_ends);
    ser_data["filtered"] = make_bit_array(filtered_starts, filtered_ends);
    ser_data["selected"] = make_bit_array(selected_starts, selected_ends);
    const std::chrono::duration<double, std::milli> diffSerialMillis = endSerialTime - startSerialTime;
    Log::DefaultLog.WriteInfo("serial reduction: %lf ms", diffSerialMillis.count());
    this->serializedFlags.Param<core::param::StringParam>()->SetValue(ser_data.dump().c_str());
#else
    const auto startParallelTime = std::chrono::high_resolution_clock::now();
    BitsChecker bc(cdata);
    tbb::parallel_reduce(tbb::blocked_range<int32_t>(0, static_cast<int32_t>(cdata->size()), 50000), bc);
    const auto endParallelTime = std::chrono::high_resolution_clock::now();
    ASSERT(bc.enabled_starts.size() == bc.enabled_ends.size());
    ASSERT(bc.filtered_starts.size() == bc.filtered_ends.size());
    ASSERT(bc.selected_starts.size() == bc.selected_ends.size());
    nlohmann::json parallel_data;
    parallel_data["enabled"] = make_bit_array(bc.enabled_starts, bc.enabled_ends);
    parallel_data["filtered"] = make_bit_array(bc.filtered_starts, bc.filtered_ends);
    parallel_data["selected"] = make_bit_array(bc.selected_starts, bc.selected_ends);
    const std::chrono::duration<double, std::milli> diffParallelMillis = endParallelTime - startParallelTime;
    Log::DefaultLog.WriteInfo("parallel reduction: %lf ms", diffParallelMillis.count());
    this->serializedFlags.Param<core::param::StringParam>()->SetValue(parallel_data.dump().c_str());
#endif
    //ASSERT(parallel_data.dump() == ser_data.dump());
}

void FlagStorage::deserializeCPUData() {
    if (skipFlagsSerializationParam.Param<core::param::BoolParam>()->Value()) {
        return;
    }

    try {
        auto j = nlohmann::json::parse(this->serializedFlags.Param<core::param::StringParam>()->Value());
        FlagStorageTypes::index_type max_flag_index = 0;
        // reset all flags
        if (j.contains("enabled")) {
            max_flag_index = std::max(array_max(j["enabled"]), max_flag_index);
        }
        if (j.contains("filtered")) {
            max_flag_index = std::max(array_max(j["filtered"]), max_flag_index);
        }
        if (j.contains("selected")) {
            max_flag_index = std::max(array_max(j["selected"]), max_flag_index);
        }
        theCPUData->validateFlagCount(max_flag_index + 1);
        if (j.contains("enabled")) {
            array_to_bits(j["enabled"], FlagStorageTypes::flag_bits::ENABLED);
        } else {
            utility::log::Log::DefaultLog.WriteWarn("UniFlagStorage: serialized flags do not contain enabled items");
        }
        if (j.contains("filtered")) {
            array_to_bits(j["filtered"], FlagStorageTypes::flag_bits::FILTERED);
        } else {
            utility::log::Log::DefaultLog.WriteWarn("UniFlagStorage: serialized flags do not contain filtered items");
        }
        if (j.contains("selected")) {
            array_to_bits(j["selected"], FlagStorageTypes::flag_bits::SELECTED);
        } else {
            utility::log::Log::DefaultLog.WriteWarn("UniFlagStorage: serialized flags do not contain selected items");
        }
    } catch (nlohmann::detail::parse_error& e) {
        utility::log::Log::DefaultLog.WriteError("UniFlagStorage: failed parsing serialized flags: %s", e.what());
    }
}

bool FlagStorage::onJSONChanged(param::ParamSlot& slot) {
    deserializeCPUData();
    return true;
}
