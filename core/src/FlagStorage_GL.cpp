#include "stdafx.h"
#include "mmcore/FlagStorage_GL.h"
#include "mmcore/FlagCall_GL.h"

using namespace megamol;
using namespace megamol::core;


FlagStorage_GL::FlagStorage_GL(void)
    : readFlagsSlot("readFlags", "Provides flag data to clients."),
    writeFlagsSlot("writeFlags", "Accepts updated flag data from clients."){

    this->readFlagsSlot.SetCallback(
        FlagCallRead_GL::ClassName(), FlagCallRead_GL::FunctionName(FlagCallRead_GL::CallGetData), &FlagStorage_GL::readDataCallback);
    this->readFlagsSlot.SetCallback(
        FlagCallRead_GL::ClassName(), FlagCallRead_GL::FunctionName(FlagCallRead_GL::CallGetMetaData), &FlagStorage_GL::readMetaDataCallback);
    this->MakeSlotAvailable(&this->readFlagsSlot);

    this->writeFlagsSlot.SetCallback(
        FlagCallWrite_GL::ClassName(), FlagCallWrite_GL::FunctionName(FlagCallWrite_GL::CallGetData), &FlagStorage_GL::writeDataCallback);
    this->writeFlagsSlot.SetCallback(
        FlagCallWrite_GL::ClassName(), FlagCallWrite_GL::FunctionName(FlagCallWrite_GL::CallGetMetaData), &FlagStorage_GL::writeMetaDataCallback);
    this->MakeSlotAvailable(&this->writeFlagsSlot);
}


FlagStorage_GL::~FlagStorage_GL(void) { this->Release(); };


bool FlagStorage_GL::create(void) {
    this->theData = std::make_shared<FlagCollection_GL>();
    const int num = 10;
    std::vector<uint32_t> temp_data(num, FlagStorage::ENABLED);
    this->theData->flags = std::make_shared<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, temp_data.data(), num, GL_DYNAMIC_DRAW);
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