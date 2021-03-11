#include "stdafx.h"
#include "mmcore/FlagStorage_GL.h"
#include "mmcore/FlagCall_GL.h"

using namespace megamol;
using namespace megamol::core;


FlagStorage_GL::FlagStorage_GL(void)
        : readFlagsSlot("readFlags", "Provides flag data to clients.")
        , writeFlagsSlot("writeFlags", "Accepts updated flag data from clients.")
        , readCPUFlagsSlot("readCPUFlags", "Provides flag data to clients.")
        , writeCPUFlagsSlot("writeCPUFlags", "Accepts updated flag data from clients.") {

    this->readFlagsSlot.SetCallback(FlagCallRead_GL::ClassName(),
        FlagCallRead_GL::FunctionName(FlagCallRead_GL::CallGetData), &FlagStorage_GL::readDataCallback);
    this->readFlagsSlot.SetCallback(FlagCallRead_GL::ClassName(),
        FlagCallRead_GL::FunctionName(FlagCallRead_GL::CallGetMetaData), &FlagStorage_GL::readMetaDataCallback);
    this->MakeSlotAvailable(&this->readFlagsSlot);

    this->writeFlagsSlot.SetCallback(FlagCallWrite_GL::ClassName(),
        FlagCallWrite_GL::FunctionName(FlagCallWrite_GL::CallGetData), &FlagStorage_GL::writeDataCallback);
    this->writeFlagsSlot.SetCallback(FlagCallWrite_GL::ClassName(),
        FlagCallWrite_GL::FunctionName(FlagCallWrite_GL::CallGetMetaData), &FlagStorage_GL::writeMetaDataCallback);
    this->MakeSlotAvailable(&this->writeFlagsSlot);

    this->readCPUFlagsSlot.SetCallback(FlagCallRead_CPU::ClassName(),
        FlagCallRead_CPU::FunctionName(FlagCallRead_CPU::CallGetData), &FlagStorage_GL::readDataCallback);
    this->readCPUFlagsSlot.SetCallback(FlagCallRead_CPU::ClassName(),
        FlagCallRead_CPU::FunctionName(FlagCallRead_CPU::CallGetMetaData), &FlagStorage_GL::readMetaDataCallback);
    this->MakeSlotAvailable(&this->readCPUFlagsSlot);

    this->writeCPUFlagsSlot.SetCallback(FlagCallWrite_CPU::ClassName(),
        FlagCallWrite_CPU::FunctionName(FlagCallWrite_CPU::CallGetData), &FlagStorage_GL::writeDataCallback);
    this->writeCPUFlagsSlot.SetCallback(FlagCallWrite_CPU::ClassName(),
        FlagCallWrite_CPU::FunctionName(FlagCallWrite_CPU::CallGetMetaData), &FlagStorage_GL::writeMetaDataCallback);
    this->MakeSlotAvailable(&this->writeCPUFlagsSlot);
}


FlagStorage_GL::~FlagStorage_GL(void) { this->Release(); };


bool FlagStorage_GL::create(void) {
    this->theData = std::make_shared<FlagCollection_GL>();
    const int num = 10;
    std::vector<uint32_t> temp_data(num, FlagStorage::ENABLED);
    this->theData->flags = std::make_shared<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, temp_data.data(), num, GL_DYNAMIC_DRAW);
    this->theCPUData->flags->resize(num, FlagStorage::ENABLED);
    return true;
}


void FlagStorage_GL::release(void) {
    // intentionally empty
}

bool FlagStorage_GL::readDataCallback(core::Call& caller) {
    auto fc = dynamic_cast<FlagCallRead_GL*>(&caller);
    if (fc == nullptr) return false;

    fc->setData(this->theData, this->version);
    return true;
}

bool FlagStorage_GL::writeDataCallback(core::Call& caller) {
    auto fc = dynamic_cast<FlagCallWrite_GL*>(&caller);
    if (fc == nullptr) return false;

    if (fc->version() > this->version) {
        this->theData = fc->getData();
        this->version = fc->version();
        GL2CPUCopy();
    }
    return true;
}

bool FlagStorage_GL::readCPUDataCallback(core::Call& caller) {
    auto fc = dynamic_cast<FlagCallRead_CPU*>(&caller);
    if (fc == nullptr)
        return false;

    fc->setData(this->theCPUData, this->version);
    return true;
}

bool FlagStorage_GL::writeCPUDataCallback(core::Call& caller) {
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

bool FlagStorage_GL::readMetaDataCallback(core::Call& caller) {
    //auto fc = dynamic_cast<FlagCallRead_GL*>(&caller);
    //if (fc == nullptr) return false;

    return true;
}

bool FlagStorage_GL::writeMetaDataCallback(core::Call& caller) {
    //auto fc = dynamic_cast<FlagCallWrite_GL*>(&caller);
    //if (fc == nullptr) return false;

    return true;
}
