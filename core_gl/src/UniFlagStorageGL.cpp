#include "stdafx.h"
#include "mmcore_gl/UniFlagStorageGL.h"

#include "mmcore/UniFlagCalls.h"
#include "mmcore_gl/UniFlagCallsGL.h"

#include "OpenGL_Context.h"

using namespace megamol;
using namespace megamol::core_gl;


UniFlagStorageGL::UniFlagStorageGL(void)
        : core::UniFlagStorage(), readFlagsSlot("readFlags", "Provides flag data to clients.")
        , writeFlagsSlot("writeFlags", "Accepts updated flag data from clients.") {

    this->readFlagsSlot.SetCallback(FlagCallRead_GL::ClassName(),
        FlagCallRead_GL::FunctionName(FlagCallRead_GL::CallGetData), &UniFlagStorageGL::readDataCallback);
    this->readFlagsSlot.SetCallback(FlagCallRead_GL::ClassName(),
        FlagCallRead_GL::FunctionName(FlagCallRead_GL::CallGetMetaData), &core::UniFlagStorage::readMetaDataCallback);
    this->MakeSlotAvailable(&this->readFlagsSlot);

    this->writeFlagsSlot.SetCallback(FlagCallWrite_GL::ClassName(),
        FlagCallWrite_GL::FunctionName(FlagCallWrite_GL::CallGetData), &UniFlagStorageGL::writeDataCallback);
    this->writeFlagsSlot.SetCallback(FlagCallWrite_GL::ClassName(),
        FlagCallWrite_GL::FunctionName(FlagCallWrite_GL::CallGetMetaData), &core::UniFlagStorage::writeMetaDataCallback);
    this->MakeSlotAvailable(&this->writeFlagsSlot);

}


UniFlagStorageGL::~UniFlagStorageGL(void) {
    this->Release();
};


bool UniFlagStorageGL::create(void) {
    auto const& ogl_ctx = frontend_resources.get<frontend_resources::OpenGL_Context>();
    if (!ogl_ctx.isVersionGEQ(4, 3))
        return false;

    this->theData = std::make_shared<FlagCollection_GL>();
    const int num = 10;
    std::vector<uint32_t> temp_data(num, core::FlagStorage::ENABLED);
    this->theData->flags =
        std::make_shared<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, temp_data.data(), num, GL_DYNAMIC_DRAW);
    this->theCPUData = std::make_shared<core::FlagCollection_CPU>();
    this->theCPUData->flags = std::make_shared<core::FlagStorage::FlagVectorType>(num, core::FlagStorage::ENABLED);
    return true;
}


void UniFlagStorageGL::release(void) {
    // intentionally empty
}

bool UniFlagStorageGL::readDataCallback(core::Call& caller) {
    auto fc = dynamic_cast<FlagCallRead_GL*>(&caller);
    if (fc == nullptr)
        return false;

    fc->setData(this->theData, this->version);
    return true;
}

bool UniFlagStorageGL::writeDataCallback(core::Call& caller) {
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
