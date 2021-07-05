#include "stdafx.h"
#include "mmcore/UniFlagStorage.h"
#include "mmcore/UniFlagCalls.h"

using namespace megamol;
using namespace megamol::core;


UniFlagStorage::UniFlagStorage(void)
        : readFlagsSlot("readFlags", "Provides flag data to clients.")
        , writeFlagsSlot("writeFlags", "Accepts updated flag data from clients.")
        , readCPUFlagsSlot("readCPUFlags", "Provides flag data to clients.")
        , writeCPUFlagsSlot("writeCPUFlags", "Accepts updated flag data from clients.") {

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
        GL2CPUCopy();
    }
    return true;
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
        CPU2GLCopy();
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
