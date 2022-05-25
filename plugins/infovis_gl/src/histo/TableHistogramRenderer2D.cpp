/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "TableHistogramRenderer2D.h"

#include "datatools/table/TableDataCall.h"
#include "mmcore/param/IntParam.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmstd_gl/flags/FlagCallsGL.h"

using namespace megamol::infovis_gl;
using namespace megamol::datatools;

using megamol::core::utility::log::Log;

TableHistogramRenderer2D::TableHistogramRenderer2D()
        : BaseHistogramRenderer2D()
        , tableDataCallerSlot_("getData", "Float table input")
        , flagStorageReadCallerSlot_("readFlagStorage", "Flag storage read input")
        , flagStorageWriteCallerSlot_("writeFlagStorage", "Flag storage write input")
        , numRows_(0)
        , currentTableDataHash_(std::numeric_limits<std::size_t>::max())
        , currentTableFrameId_(std::numeric_limits<unsigned int>::max()) {
    tableDataCallerSlot_.SetCompatibleCall<table::TableDataCallDescription>();
    MakeSlotAvailable(&tableDataCallerSlot_);

    flagStorageReadCallerSlot_.SetCompatibleCall<mmstd_gl::FlagCallRead_GLDescription>();
    MakeSlotAvailable(&flagStorageReadCallerSlot_);

    flagStorageWriteCallerSlot_.SetCompatibleCall<mmstd_gl::FlagCallWrite_GLDescription>();
    MakeSlotAvailable(&flagStorageWriteCallerSlot_);
}

TableHistogramRenderer2D::~TableHistogramRenderer2D() {
    this->Release();
}

bool TableHistogramRenderer2D::createImpl(const msf::ShaderFactoryOptionsOpenGL& shaderOptions) {
    try {
        calcHistogramProgram_ = core::utility::make_glowl_shader(
            "histo_table_calc", shaderOptions, "infovis_gl/histo/table_calc.comp.glsl");
        selectionProgram_ = core::utility::make_glowl_shader(
            "histo_table_select", shaderOptions, "infovis_gl/histo/table_select.comp.glsl");
    } catch (std::exception& e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, ("TableHistogramRenderer2D: " + std::string(e.what())).c_str());
        return false;
    }

    glGenBuffers(1, &floatDataBuffer_);

    glGetProgramiv(selectionProgram_->getHandle(), GL_COMPUTE_WORK_GROUP_SIZE, selectionWorkgroupSize_);

    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &maxWorkgroupCount_[0]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &maxWorkgroupCount_[1]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &maxWorkgroupCount_[2]);

    return true;
}

void TableHistogramRenderer2D::releaseImpl() {
    glDeleteBuffers(1, &floatDataBuffer_);
}

bool TableHistogramRenderer2D::handleCall(mmstd_gl::CallRender2DGL& call) {
    auto floatTableCall = tableDataCallerSlot_.CallAs<table::TableDataCall>();
    if (floatTableCall == nullptr) {
        return false;
    }
    auto readFlagsCall = flagStorageReadCallerSlot_.CallAs<mmstd_gl::FlagCallRead_GL>();
    if (readFlagsCall == nullptr) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "TableHistogramRenderer2D requires a read flag storage!");
        return false;
    }
    auto writeFlagsCall = flagStorageWriteCallerSlot_.CallAs<mmstd_gl::FlagCallWrite_GL>();
    if (writeFlagsCall == nullptr) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "TableHistogramRenderer2D requires a write flag storage!");
        return false;
    }

    (*readFlagsCall)(mmstd_gl::FlagCallRead_GL::CallGetData);

    floatTableCall->SetFrameID(static_cast<unsigned int>(call.Time()));
    (*floatTableCall)(1);
    (*floatTableCall)(0);
    call.SetTimeFramesCount(floatTableCall->GetFrameCount());
    auto hash = floatTableCall->DataHash();
    auto frameId = floatTableCall->GetFrameID();
    bool dataChanged = currentTableDataHash_ != hash || currentTableFrameId_ != frameId;
    if (dataChanged) {
        auto numComponents = floatTableCall->GetColumnsCount();

        std::vector<std::string> names(numComponents);
        std::vector<float> minimums(numComponents);
        std::vector<float> maximums(numComponents);

        for (std::size_t i = 0; i < numComponents; ++i) {
            const auto& colInfo = floatTableCall->GetColumnsInfos()[i];
            names[i] = colInfo.Name();
            minimums[i] = colInfo.MinimumValue();
            maximums[i] = colInfo.MaximumValue();
        }

        setComponentHeaders(std::move(names), std::move(minimums), std::move(maximums));

        numRows_ = floatTableCall->GetRowsCount();
        const GLsizeiptr bufSize = numComponents * numRows_ * sizeof(float);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, floatDataBuffer_);
        glBufferData(GL_SHADER_STORAGE_BUFFER, bufSize, floatTableCall->GetData(), GL_STATIC_DRAW);
    }

    if (dataChanged || readFlagsCall->hasUpdate() || binsChanged()) {
        // Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Calculate Histogram");

        resetHistogramBuffers();

        readFlagsCall->getData()->validateFlagCount(numRows_);

        calcHistogramProgram_->use();

        bindCommon(calcHistogramProgram_);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, floatDataBuffer_);
        readFlagsCall->getData()->flags->bindBase(GL_SHADER_STORAGE_BUFFER, 5);

        calcHistogramProgram_->setUniform("numRows", static_cast<GLuint>(numRows_));

        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        glUseProgram(0);

        currentTableDataHash_ = hash;
        currentTableFrameId_ = frameId;
    }

    return true;
}

void TableHistogramRenderer2D::updateSelection(SelectionMode selectionMode, int selectedComponent, int selectedBin) {
    auto readFlagsCall = flagStorageReadCallerSlot_.CallAs<mmstd_gl::FlagCallRead_GL>();
    auto writeFlagsCall = flagStorageWriteCallerSlot_.CallAs<mmstd_gl::FlagCallWrite_GL>();
    if (readFlagsCall != nullptr && writeFlagsCall != nullptr) {
        selectionProgram_->use();

        bindCommon(selectionProgram_);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, floatDataBuffer_);
        readFlagsCall->getData()->flags->bindBase(GL_SHADER_STORAGE_BUFFER, 5);

        selectionProgram_->setUniform("numRows", static_cast<GLuint>(numRows_));
        selectionProgram_->setUniform(
            "selectionMode", static_cast<std::underlying_type_t<SelectionMode>>(selectionMode));
        selectionProgram_->setUniform("selectedComponent", selectedComponent);
        selectionProgram_->setUniform("selectedBin", selectedBin);

        GLuint groupCounts[3];
        computeDispatchSizes(numRows_, selectionWorkgroupSize_, maxWorkgroupCount_, groupCounts);

        glDispatchCompute(groupCounts[0], groupCounts[1], groupCounts[2]);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        glUseProgram(0);

        writeFlagsCall->setData(readFlagsCall->getData(), readFlagsCall->version() + 1);
        (*writeFlagsCall)(mmstd_gl::FlagCallWrite_GL::CallGetData);
    }
}
