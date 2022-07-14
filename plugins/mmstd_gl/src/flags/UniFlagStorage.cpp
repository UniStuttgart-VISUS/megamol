/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd_gl/flags/UniFlagStorage.h"

#include <json.hpp>

#include "OpenGL_Context.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmstd/flags/FlagCalls.h"
#include "mmstd_gl/flags/FlagCallsGL.h"

using namespace megamol;
using namespace megamol::mmstd_gl;


UniFlagStorage::UniFlagStorage()
        : FlagStorage()
        , readGLFlagsSlot("readFlags", "Provides flag data to clients.")
        , writeGLFlagsSlot("writeFlags", "Accepts updated flag data from clients.") {

    this->readGLFlagsSlot.SetCallback(FlagCallRead_GL::ClassName(),
        FlagCallRead_GL::FunctionName(FlagCallRead_GL::CallGetData), &UniFlagStorage::readGLDataCallback);
    this->readGLFlagsSlot.SetCallback(FlagCallRead_GL::ClassName(),
        FlagCallRead_GL::FunctionName(FlagCallRead_GL::CallGetMetaData), &core::FlagStorage::readMetaDataCallback);
    this->MakeSlotAvailable(&this->readGLFlagsSlot);

    this->writeGLFlagsSlot.SetCallback(FlagCallWrite_GL::ClassName(),
        FlagCallWrite_GL::FunctionName(FlagCallWrite_GL::CallGetData), &UniFlagStorage::writeGLDataCallback);
    this->writeGLFlagsSlot.SetCallback(FlagCallWrite_GL::ClassName(),
        FlagCallWrite_GL::FunctionName(FlagCallWrite_GL::CallGetMetaData), &core::FlagStorage::writeMetaDataCallback);
    this->MakeSlotAvailable(&this->writeGLFlagsSlot);
}


UniFlagStorage::~UniFlagStorage() {
    this->Release();
};


bool UniFlagStorage::create() {
    const int num = 1;

    auto const& ogl_ctx = frontend_resources.get<frontend_resources::OpenGL_Context>();
    if (!ogl_ctx.isVersionGEQ(4, 3))
        return false;

    // TODO beware this shader only compiles and has never been tested. It will probably release the kraken or something more sinister
    try {
        auto const shaderOptions = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        compressGPUFlagsProgram = core::utility::make_glowl_shader(
            "compress_bitflags", shaderOptions, "mmstd_gl/flags/compress_bitflags.comp.glsl");
    } catch (std::exception& e) {
        Log::DefaultLog.WriteError(
            ("UniFlagStorage: could not compile compute shader: " + std::string(e.what())).c_str());
        return false;
    }

    this->theGLData = std::make_shared<FlagCollection_GL>();
    this->theGLData->validateFlagCount(num);

    core::FlagStorage::create();
    return true;
}


void UniFlagStorage::release() {
    // intentionally empty
}

bool UniFlagStorage::readGLDataCallback(core::Call& caller) {
    auto fc = dynamic_cast<FlagCallRead_GL*>(&caller);
    if (fc == nullptr)
        return false;

    if (gpu_stale) {
        CPU2GLCopy();
        gpu_stale = false;
    }

    // this is for debugging only
    //GL2CPUCopy();
    //serializeCPUData();

    fc->setData(this->theGLData, this->version);
    return true;
}

bool UniFlagStorage::writeGLDataCallback(core::Call& caller) {
    auto fc = dynamic_cast<FlagCallWrite_GL*>(&caller);
    if (fc == nullptr)
        return false;

    if (fc->version() > this->version) {
        this->theGLData = fc->getData();
        this->version = fc->version();
        cpu_stale = true;

        if (!skipFlagsSerializationParam.Param<core::param::BoolParam>()->Value()) {
            // TODO try to avoid this and only fetch the serialization data from the GPU!!!! (if and when it works)
            // see compress_bitflags.comp.glsl (never tested yet!)
            // -> replace the whole block below with serializeData()
            // actually with TBB performance is fine already haha
            GL2CPUCopy();
            cpu_stale = false; // on purpose!
            serializeCPUData();
        }
    }
    return true;
}

bool UniFlagStorage::readCPUDataCallback(core::Call& caller) {
    if (cpu_stale) {
        GL2CPUCopy();
        cpu_stale = false;
    }
    return core::FlagStorage::readCPUDataCallback(caller);
}

bool UniFlagStorage::writeCPUDataCallback(core::Call& caller) {
    auto fc = dynamic_cast<core::FlagCallWrite_CPU*>(&caller);
    if (fc == nullptr)
        return false;

    if (fc->version() > this->version) {
        // all but the GPU stuff happens in parent
        //this->theCPUData = fc->getData();
        //this->version = fc->version();
        gpu_stale = true;
        //serializeCPUData();
    }
    return core::FlagStorage::writeCPUDataCallback(caller);
}

void UniFlagStorage::serializeGLData() {
    this->theGLData->flags->bind(GL_SHADER_STORAGE_BUFFER);
    megamol::core::utility::log::Log::DefaultLog.WriteError(
        "UniFlagStorage::serializeData: not implemented! If you see this, you have a problem.");
    // TODO allocate the buffers
    // TODO bind params
    // foreach bit: call compute shader, pray
    // only build onoff for the first bit!
    // download stuff
}

bool UniFlagStorage::onJSONChanged(core::param::ParamSlot& slot) {
    if (cpu_stale) {
        GL2CPUCopy();
    }
    deserializeCPUData();
    gpu_stale = true;
    return true;
}

void UniFlagStorage::CPU2GLCopy() {
    const auto& flags = *theCPUData->flags;
    theGLData->validateFlagCount(flags.size());
    glNamedBufferSubData(
        theGLData->flags->getName(), 0, flags.size() * sizeof(core::FlagStorageTypes::flag_item_type), flags.data());
}

void UniFlagStorage::GL2CPUCopy() {
    auto const num = theGLData->flags->getByteSize() / sizeof(core::FlagStorageTypes::flag_item_type);
    theCPUData->validateFlagCount(num);
    glGetNamedBufferSubData(theGLData->flags->getName(), 0, theGLData->flags->getByteSize(), theCPUData->flags->data());
}
