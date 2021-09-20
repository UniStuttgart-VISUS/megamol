#include "stdafx.h"
#include "mmcore/UniFlagStorage.h"
#include "mmcore/UniFlagCalls.h"
#include "json.hpp"
#include "mmcore/param/StringParam.h"

using namespace megamol;
using namespace megamol::core;


UniFlagStorage::UniFlagStorage(void)
        : readFlagsSlot("readFlags", "Provides flag data to clients.")
        , writeFlagsSlot("writeFlags", "Accepts updated flag data from clients.")
        , readCPUFlagsSlot("readCPUFlags", "Provides flag data to clients.")
        , writeCPUFlagsSlot("writeCPUFlags", "Accepts updated flag data from clients.")
        , serializedFlags("serializedFlags", "persists the flags in projects") {

    this->readFlagsSlot.SetCallback(FlagCallRead_GL::ClassName(),
        FlagCallRead_GL::FunctionName(FlagCallRead_GL::CallGetData), &UniFlagStorage::readDataCallback);
    this->readFlagsSlot.SetCallback(FlagCallRead_GL::ClassName(),
        FlagCallRead_GL::FunctionName(FlagCallRead_GL::CallGetMetaData), &UniFlagStorage::readMetaDataCallback);
    this->MakeSlotAvailable(&this->readFlagsSlot);

    this->writeFlagsSlot.SetCallback(FlagCallWrite_GL::ClassName(),
        FlagCallWrite_GL::FunctionName(FlagCallWrite_GL::CallGetData), &UniFlagStorage::writeDataCallback);
    this->writeFlagsSlot.SetCallback(FlagCallWrite_GL::ClassName(),
        FlagCallWrite_GL::FunctionName(FlagCallWrite_GL::CallGetMetaData), &UniFlagStorage::writeMetaDataCallback);
    this->MakeSlotAvailable(&this->writeFlagsSlot);

    this->readCPUFlagsSlot.SetCallback(FlagCallRead_CPU::ClassName(),
        FlagCallRead_CPU::FunctionName(FlagCallRead_CPU::CallGetData), &UniFlagStorage::readCPUDataCallback);
    this->readCPUFlagsSlot.SetCallback(FlagCallRead_CPU::ClassName(),
        FlagCallRead_CPU::FunctionName(FlagCallRead_CPU::CallGetMetaData), &UniFlagStorage::readMetaDataCallback);
    this->MakeSlotAvailable(&this->readCPUFlagsSlot);

    this->writeCPUFlagsSlot.SetCallback(FlagCallWrite_CPU::ClassName(),
        FlagCallWrite_CPU::FunctionName(FlagCallWrite_CPU::CallGetData), &UniFlagStorage::writeCPUDataCallback);
    this->writeCPUFlagsSlot.SetCallback(FlagCallWrite_CPU::ClassName(),
        FlagCallWrite_CPU::FunctionName(FlagCallWrite_CPU::CallGetMetaData), &UniFlagStorage::writeMetaDataCallback);
    this->MakeSlotAvailable(&this->writeCPUFlagsSlot);

    this->serializedFlags << new core::param::StringParam("");
    this->serializedFlags.SetUpdateCallback(&UniFlagStorage::onJSONChanged);
    this->MakeSlotAvailable(&this->serializedFlags);
}


UniFlagStorage::~UniFlagStorage(void) {
    this->Release();
};


bool UniFlagStorage::create(void) {
    this->theData = std::make_shared<FlagCollection_GL>();
    const int num = 10;
    std::vector<uint32_t> temp_data(num, FlagStorage::ENABLED);
    this->theData->flags =
        std::make_shared<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, temp_data.data(), num, GL_DYNAMIC_DRAW);
    this->theCPUData = std::make_shared<FlagCollection_CPU>();
    this->theCPUData->flags = std::make_shared<FlagStorage::FlagVectorType>(num, FlagStorage::ENABLED);
    return true;
}


void UniFlagStorage::release(void) {
    // intentionally empty
}

bool UniFlagStorage::readDataCallback(core::Call& caller) {
    auto fc = dynamic_cast<FlagCallRead_GL*>(&caller);
    if (fc == nullptr)
        return false;

    if (gpu_stale) {
        CPU2GLCopy();
        gpu_stale = false;
    }

    fc->setData(this->theData, this->version);
    return true;
}

bool UniFlagStorage::writeDataCallback(core::Call& caller) {
    auto fc = dynamic_cast<FlagCallWrite_GL*>(&caller);
    if (fc == nullptr)
        return false;

    if (fc->version() > this->version) {
        this->theData = fc->getData();
        this->version = fc->version();
        cpu_stale = true;

        GL2CPUCopy();
        serializeCPUData();
    }
    return true;
}

bool UniFlagStorage::readCPUDataCallback(core::Call& caller) {
    auto fc = dynamic_cast<FlagCallRead_CPU*>(&caller);
    if (fc == nullptr)
        return false;

    if (cpu_stale) {
        GL2CPUCopy();
        cpu_stale = false;
    }

    fc->setData(this->theCPUData, this->version);
    return true;
}

bool UniFlagStorage::writeCPUDataCallback(core::Call& caller) {
    auto fc = dynamic_cast<FlagCallWrite_CPU*>(&caller);
    if (fc == nullptr)
        return false;

    if (fc->version() > this->version) {
        this->theCPUData = fc->getData();
        this->version = fc->version();
        gpu_stale = true;
        serializeCPUData();
    }
    return true;
}

bool UniFlagStorage::readMetaDataCallback(core::Call& caller) {
    // auto fc = dynamic_cast<FlagCallRead_GL*>(&caller);
    // if (fc == nullptr) return false;

    return true;
}

bool UniFlagStorage::writeMetaDataCallback(core::Call& caller) {
    // auto fc = dynamic_cast<FlagCallWrite_GL*>(&caller);
    // if (fc == nullptr) return false;

    return true;
}

void UniFlagStorage::serializeData() {
    
}

void UniFlagStorage::check_bits(uint32_t flag_bit, std::vector<int32_t>& bit_starts, std::vector<int32_t>& bit_ends,
    int32_t& curr_bit_start, int32_t x, const std::shared_ptr<FlagStorage::FlagVectorType>& flags) {
    auto& f = (*flags)[x];
    if ((f & flag_bit) > 0) {
        if (curr_bit_start == -1) {
            curr_bit_start = x;
            bit_starts.push_back(x);
        }
    } else {
        if (curr_bit_start > -1) {
            bit_ends.push_back(x - 1);
            curr_bit_start = -1;
        }
    }
}

void UniFlagStorage::terminate_bit(const std::shared_ptr<FlagStorage::FlagVectorType>& cdata, std::vector<int32_t>& bit_ends, int32_t curr_bit_start) {
    if (curr_bit_start > -1) {
        bit_ends.push_back(cdata->size() - 1);
    }
}

nlohmann::json UniFlagStorage::make_bit_array(
    const std::vector<int32_t>& bit_starts, const std::vector<int32_t>& bit_ends) {
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

void UniFlagStorage::array_to_bits(const nlohmann::json& json, uint32_t flag_bit) {
    for (auto& j: json) {
        if (j.is_array()) {
            int32_t from, to;
            j[0].get_to(from);
            j[1].get_to(to);
            for (int32_t x = from; x <= to; ++x) {
                (*theCPUData->flags)[x] |= flag_bit;
            }
        } else {
            int32_t idx;
            j.get_to(idx);
            (*theCPUData->flags)[idx] |= flag_bit;
        }
    }
}


void UniFlagStorage::serializeCPUData() {
    const auto& cdata = theCPUData->flags;

    // enum { ENABLED = 1 << 0, FILTERED = 1 << 1, SELECTED = 1 << 2, SOFTSELECTED = 1 << 3 };
    std::vector<int32_t> enabled_starts, enabled_ends;
    std::vector<int32_t> filtered_starts, filtered_ends;
    std::vector<int32_t> selected_starts, selected_ends;
    //std::vector<int32_t> softselected_starts, softselected_ends;
    int32_t curr_enabled_start = -1, curr_filtered_start = -1, curr_selected_start = -1; //, curr_softselected_start = -1;

    for (int32_t x = 0; x < cdata->size(); ++x) {
        check_bits(FlagStorage::ENABLED, enabled_starts, enabled_ends, curr_enabled_start, x, cdata);
        check_bits(FlagStorage::FILTERED, filtered_starts, filtered_ends, curr_filtered_start, x, cdata);
        check_bits(FlagStorage::SELECTED, selected_starts, selected_ends, curr_selected_start, x, cdata);
        //check_bits(FlagStorage::SOFTSELECTED, softselected_starts, softselected_ends, curr_softselected_start, x, cdata);
    }

    terminate_bit(cdata, enabled_ends, curr_enabled_start);
    terminate_bit(cdata, filtered_ends, curr_filtered_start);
    terminate_bit(cdata, selected_ends, curr_selected_start);
    //terminate_bit(cdata, softselected_ends, curr_softselected_start);

    ASSERT(enabled_starts.size() == enabled_ends.size());
    nlohmann::json ser_data;
    ser_data["enabled"] = make_bit_array(enabled_starts, enabled_ends);
    ser_data["filtered"] = make_bit_array(filtered_starts, filtered_ends);
    ser_data["selected"] = make_bit_array(selected_starts, selected_ends);
    //ser_data["softselected"] = make_bit_array(softselected_starts, softselected_ends);

    this->serializedFlags.Param<core::param::StringParam>()->SetValue(ser_data.dump().c_str());
}

void UniFlagStorage::deserializeCPUData() {
    try {
        auto j = nlohmann::json::parse(this->serializedFlags.Param<core::param::StringParam>()->Value().PeekBuffer());
        // reset all flags
        theCPUData->flags->assign(theCPUData->flags->size(), 0);
        if (j.contains("enabled")) {
            array_to_bits(j["enabled"], FlagStorage::ENABLED);
        } else {
            utility::log::Log::DefaultLog.WriteWarn("UniFlagStorage: serialized flags do not contain enabled items");
        }
        if (j.contains("filtered")) {
            array_to_bits(j["filtered"], FlagStorage::FILTERED);
        } else {
            utility::log::Log::DefaultLog.WriteWarn("UniFlagStorage: serialized flags do not contain filtered items");
        }
        if (j.contains("selected")) {
            array_to_bits(j["selected"], FlagStorage::SELECTED);
        } else {
            utility::log::Log::DefaultLog.WriteWarn("UniFlagStorage: serialized flags do not contain selected items");
        }
    } catch (nlohmann::detail::parse_error& e) {
        utility::log::Log::DefaultLog.WriteError("UniFlagStorage: failed parsing serialized flags: %s", e.what());
    }
}

bool UniFlagStorage::onJSONChanged(param::ParamSlot& slot) {
    if (cpu_stale) {
        GL2CPUCopy();
    }
    deserializeCPUData();
    gpu_stale = true;
    return true;
}
