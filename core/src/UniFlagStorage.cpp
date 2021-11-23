#include "mmcore/UniFlagStorage.h"
#include "mmcore/UniFlagCalls.h"
#include "stdafx.h"

using namespace megamol;
using namespace megamol::core;


UniFlagStorage::UniFlagStorage(void)
        : readCPUFlagsSlot("readCPUFlags", "Provides flag data to clients.")
        , writeCPUFlagsSlot("writeCPUFlags", "Accepts updated flag data from clients.") {

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
}


UniFlagStorage::~UniFlagStorage(void) {
    this->Release();
};


bool UniFlagStorage::create(void) {
    const int num = 10;
    std::vector<uint32_t> temp_data(num, FlagStorage::ENABLED);
    this->theCPUData = std::make_shared<FlagCollection_CPU>();
    this->theCPUData->flags = std::make_shared<FlagStorage::FlagVectorType>(num, FlagStorage::ENABLED);
    return true;
}


void UniFlagStorage::release(void) {
    // intentionally empty
}


bool UniFlagStorage::readCPUDataCallback(core::Call& caller) {
    auto fc = dynamic_cast<FlagCallRead_CPU*>(&caller);
    if (fc == nullptr)
        return false;

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
