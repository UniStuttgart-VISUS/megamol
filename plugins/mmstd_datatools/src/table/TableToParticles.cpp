#include "stdafx.h"
#include "TableToParticles.h"

#include "mmcore/moldyn/EllipsoidalDataCall.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"

#include "vislib/Trace.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/PerformanceCounter.h"
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtc/type_ptr.hpp"

using namespace megamol::stdplugin::datatools;
using namespace megamol;

/*
 * TableToParticles::TableToParticles
 */
TableToParticles::TableToParticles(void)
    : Module()
    , slotCallMultiPart("multidata", "Provides the data as MultiParticle call.")
    , slotCallTable("floattable", "float table input call")
    , slotColumnR("redcolumnname", "The name of the column holding the red colour channel value.")
    , slotColumnG("greencolumnname", "The name of the column holding the green colour channel value.")
    , slotColumnB("bluecolumnname", "The name of the column holding the blue colour channel value.")
    , slotColumnI("intensitycolumnname", "The name of the column holding the intensity colour channel value.")
    , slotGlobalColor("color", "Constant sphere color.")
    , slotColorMode("colormode", "Pass on color as RGB or intensity")
    , slotColumnRadius("radiuscolumnname", "The name of the column holding the particle radius.")
    , slotGlobalRadius("radius", "Constant sphere radius.")
    , slotRadiusMode("radiusmode", "pass on global or per-particle radius.")
    , slotColumnX("xcolumnname", "The name of the column holding the x-coordinate.")
    , slotColumnY("ycolumnname", "The name of the column holding the y-coordinate.")
    , slotColumnZ("zcolumnname", "The name of the column holding the z-coordinate.")
    , slotColumnVX("direction::vxcolumnname", "The name of the column holding the vx-coordinate.")
    , slotColumnVY("direction::vycolumnname", "The name of the column holding the vy-coordinate.")
    , slotColumnVZ("direction::vzcolumnname", "The name of the column holding the vz-coordinate.")
    , slotTensorColumn{{{"tensor::x1columnname", "The name of the column holding the first vector (row) x component"},
          {"tensor::y1columnname", "The name of the column holding the first vector (row) y component"},
          {"tensor::z1columnname", "The name of the column holding the first vector (row) z component"},
          {"tensor::x2columnname", "The name of the column holding the second vector (row) x component"},
          {"tensor::y2columnname", "The name of the column holding the second vector (row) y component"},
          {"tensor::z2columnname", "The name of the column holding the second vector (row) z component"},
          {"tensor::x3columnname", "The name of the column holding the third vector (row) x component"},
          {"tensor::y3columnname", "The name of the column holding the third vector (row) y component"},
          {"tensor::z3columnname", "The name of the column holding the third vector (row) z component"}}}
    , slotTensorMagnitudeColumn{{{"tensor::magnitude1", "the magnitude of the first vector (row)"},
          {"tensor::magnitude2", "the magnitude of the second vector (row)"},
          {"tensor::magnitude3", "the magnitude of the third vector (row)"}}}
    , inputHash(0)
    , myHash(0)
    , columnIndex() {

    /* Register parameters. */
    core::param::FlexEnumParam* rColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnR << rColumnEp;
    this->MakeSlotAvailable(&this->slotColumnR);

    core::param::FlexEnumParam* gColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnG << gColumnEp;
    this->MakeSlotAvailable(&this->slotColumnG);

    core::param::FlexEnumParam* bColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnB << bColumnEp;
    this->MakeSlotAvailable(&this->slotColumnB);

    core::param::FlexEnumParam* iColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnI << iColumnEp;
    this->MakeSlotAvailable(&this->slotColumnI);

    this->slotGlobalColor << new megamol::core::param::StringParam(_T(""));
    this->MakeSlotAvailable(&this->slotGlobalColor);

    core::param::EnumParam* ep = new core::param::EnumParam(2);
    ep->SetTypePair(0, "RGB");
    ep->SetTypePair(1, "Intensity");
    ep->SetTypePair(2, "global RGB");
    this->slotColorMode << ep;
    this->MakeSlotAvailable(&this->slotColorMode);

    core::param::FlexEnumParam* columnRadiusEp = new core::param::FlexEnumParam("undef");
    this->slotColumnRadius << columnRadiusEp;
    this->MakeSlotAvailable(&this->slotColumnRadius);

    this->slotGlobalRadius << new megamol::core::param::FloatParam(0.001f);
    this->MakeSlotAvailable(&this->slotGlobalRadius);

    core::param::EnumParam* ep2 = new core::param::EnumParam(0);
    ep2->SetTypePair(0, "per particle");
    ep2->SetTypePair(1, "global");
    this->slotRadiusMode << ep2;
    this->MakeSlotAvailable(&this->slotRadiusMode);

    // this->slotColumnX << new megamol::core::param::StringParam(_T("x"));
    core::param::FlexEnumParam* xColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnX << xColumnEp;
    this->MakeSlotAvailable(&this->slotColumnX);

    core::param::FlexEnumParam* yColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnY << yColumnEp;
    this->MakeSlotAvailable(&this->slotColumnY);

    core::param::FlexEnumParam* zColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnZ << zColumnEp;
    this->MakeSlotAvailable(&this->slotColumnZ);

    core::param::FlexEnumParam* vxColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnVX << vxColumnEp;
    this->MakeSlotAvailable(&this->slotColumnVX);

    core::param::FlexEnumParam* vyColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnVY << vyColumnEp;
    this->MakeSlotAvailable(&this->slotColumnVY);

    core::param::FlexEnumParam* vzColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnVZ << vzColumnEp;
    this->MakeSlotAvailable(&this->slotColumnVZ);

    for (auto x = 0; x < 9; ++x) {
        auto* p = new core::param::FlexEnumParam("undef");
        this->slotTensorColumn[x] << p;
        this->MakeSlotAvailable(&this->slotTensorColumn[x]);
    }
    for (auto x = 0; x < 3; ++x) {
        auto* p = new core::param::FlexEnumParam("undef");
        this->slotTensorMagnitudeColumn[x] << p;
        this->MakeSlotAvailable(&this->slotTensorMagnitudeColumn[x]);
    }

    /* Register calls. */
    this->slotCallMultiPart.SetCallback(
        core::moldyn::MultiParticleDataCall::ClassName(), "GetData", &TableToParticles::getMultiParticleData);
    this->slotCallMultiPart.SetCallback(
        core::moldyn::MultiParticleDataCall::ClassName(), "GetExtent", &TableToParticles::getMultiparticleExtent);

    this->slotCallMultiPart.SetCallback(
        core::moldyn::EllipsoidalParticleDataCall::ClassName(), "GetData", &TableToParticles::getMultiParticleData);
    this->slotCallMultiPart.SetCallback(
        core::moldyn::EllipsoidalParticleDataCall::ClassName(), "GetExtent", &TableToParticles::getMultiparticleExtent);

    this->MakeSlotAvailable(&this->slotCallMultiPart);

    this->slotCallTable.SetCompatibleCall<table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->slotCallTable);
}


/*
 * TableToParticles::~TableToParticles
 */
TableToParticles::~TableToParticles(void) { this->Release(); }


/*
 * megamol::pcl::PclDataSource::create
 */
bool TableToParticles::create(void) {
    bool retval = true;
    return true;
}

bool TableToParticles::anythingDirty() {
    bool tensorDirty = std::any_of(this->slotTensorColumn.begin(), this->slotTensorColumn.end(),
        [](core::param::ParamSlot& p) { return p.IsDirty(); });
    tensorDirty =
        tensorDirty || std::any_of(this->slotTensorMagnitudeColumn.begin(), this->slotTensorMagnitudeColumn.end(),
                           [](core::param::ParamSlot& p) { return p.IsDirty(); });
    return tensorDirty || this->slotColumnR.IsDirty() || this->slotColumnG.IsDirty() || this->slotColumnB.IsDirty() ||
           this->slotColumnI.IsDirty() || this->slotGlobalColor.IsDirty() || this->slotColorMode.IsDirty() ||
           this->slotColumnRadius.IsDirty() || this->slotGlobalRadius.IsDirty() || this->slotRadiusMode.IsDirty() ||
           this->slotColumnX.IsDirty() || this->slotColumnY.IsDirty() || this->slotColumnZ.IsDirty() ||
           this->slotColumnVX.IsDirty() || this->slotColumnVY.IsDirty() || this->slotColumnVZ.IsDirty();
}

void TableToParticles::resetAllDirty() {
    this->slotColumnR.ResetDirty();
    this->slotColumnG.ResetDirty();
    this->slotColumnB.ResetDirty();
    this->slotColumnI.ResetDirty();
    this->slotGlobalColor.ResetDirty();
    this->slotColorMode.ResetDirty();
    this->slotColumnRadius.ResetDirty();
    this->slotGlobalRadius.ResetDirty();
    this->slotRadiusMode.ResetDirty();
    this->slotColumnX.ResetDirty();
    this->slotColumnY.ResetDirty();
    this->slotColumnZ.ResetDirty();
    this->slotColumnVX.ResetDirty();
    this->slotColumnVY.ResetDirty();
    this->slotColumnVZ.ResetDirty();
    for (auto& x : slotTensorColumn) {
        x.ResetDirty();
    }
    for (auto& x : slotTensorMagnitudeColumn) {
        x.ResetDirty();
    }
}

std::string TableToParticles::cleanUpColumnHeader(const std::string& header) const {
    return this->cleanUpColumnHeader(vislib::TString(header.data()));
}

std::string TableToParticles::cleanUpColumnHeader(const vislib::TString& header) const {
    vislib::TString h(header);
    h.TrimSpaces();
    h.ToLowerCase();
    return std::string(T2A(h.PeekBuffer()));
}

bool TableToParticles::pushColumnIndex(std::vector<uint32_t>& cols, const vislib::TString& colName) {
    std::string c = cleanUpColumnHeader(colName);
    if (this->columnIndex.find(c) != columnIndex.end()) {
        cols.push_back(columnIndex[c]);
        return true;
    } else {
        vislib::sys::Log::DefaultLog.WriteError("unknown column '%s'", c.c_str());
        return false;
    }
}

bool TableToParticles::assertData(table::TableDataCall* ft) {
    if (this->inputHash == ft->DataHash() && this->myTime == ft->GetFrameID() && !anythingDirty()) return true;

    if (this->inputHash != ft->DataHash()) {
        vislib::sys::Log::DefaultLog.WriteInfo("TableToParticles: Dataset changed -> Updating EnumParams\n");
        this->columnIndex.clear();

        this->slotColumnX.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnY.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnZ.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnVX.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnVY.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnVZ.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnR.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnG.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnB.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnI.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnRadius.Param<core::param::FlexEnumParam>()->ClearValues();
        for (auto& x : this->slotTensorColumn) {
            x.Param<core::param::FlexEnumParam>()->ClearValues();
        }
        for (auto& x : this->slotTensorMagnitudeColumn) {
            x.Param<core::param::FlexEnumParam>()->ClearValues();
        }

        for (uint32_t i = 0; i < ft->GetColumnsCount(); i++) {
            std::string n = std::string(this->cleanUpColumnHeader(ft->GetColumnsInfos()[i].Name()));
            columnIndex[n] = i;

            this->slotColumnX.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnY.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnZ.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnVX.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnVY.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnVZ.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnR.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnG.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnB.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnI.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnRadius.Param<core::param::FlexEnumParam>()->AddValue(n);
            for (auto& x : this->slotTensorColumn) {
                x.Param<core::param::FlexEnumParam>()->AddValue(n);
            }
            for (auto& x : this->slotTensorMagnitudeColumn) {
                x.Param<core::param::FlexEnumParam>()->AddValue(n);
            }
        }
    }

    stride = 3;
    switch (this->slotColorMode.Param<core::param::EnumParam>()->Value()) {
    case 0: // RGB
        stride += 3;
        break;
    case 1: // I
        stride += 1;
        break;
    case 2: // global RGB
        break;
    }
    switch (this->slotRadiusMode.Param<core::param::EnumParam>()->Value()) {
    case 0: // per particle
        stride += 1;
        break;
    case 1: // global
        break;
    }

    bool retValue = true;
    haveVelocities = false;

    float radius = 0.0f;

    std::vector<uint32_t> indicesToCollect;
    retValue = retValue &&
               pushColumnIndex(indicesToCollect, this->slotColumnX.Param<core::param::FlexEnumParam>()->ValueString());
    retValue = retValue &&
               pushColumnIndex(indicesToCollect, this->slotColumnY.Param<core::param::FlexEnumParam>()->ValueString());
    retValue = retValue &&
               pushColumnIndex(indicesToCollect, this->slotColumnZ.Param<core::param::FlexEnumParam>()->ValueString());

    if (this->slotRadiusMode.Param<core::param::EnumParam>()->Value() == 0) { // particle
        if (!pushColumnIndex(
                indicesToCollect, this->slotColumnRadius.Param<core::param::FlexEnumParam>()->ValueString())) {
            retValue = false;
        } else {
            radius = ft->GetColumnsInfos()[indicesToCollect.back()].MaximumValue();
        }
    } // global
    else {
        radius = this->slotGlobalRadius.Param<core::param::FloatParam>()->Value();
    }
    switch (this->slotColorMode.Param<core::param::EnumParam>()->Value()) {
    case 0: // RGB
        retValue = retValue && pushColumnIndex(indicesToCollect,
                                   this->slotColumnR.Param<core::param::FlexEnumParam>()->ValueString());
        retValue = retValue && pushColumnIndex(indicesToCollect,
                                   this->slotColumnG.Param<core::param::FlexEnumParam>()->ValueString());
        retValue = retValue && pushColumnIndex(indicesToCollect,
                                   this->slotColumnB.Param<core::param::FlexEnumParam>()->ValueString());
        break;
    case 1: // I
        if (!pushColumnIndex(indicesToCollect, this->slotColumnI.Param<core::param::FlexEnumParam>()->ValueString())) {
            retValue = false;
        } else {
            iMin = ft->GetColumnsInfos()[indicesToCollect[indicesToCollect.size() - 1]].MinimumValue();
            iMax = ft->GetColumnsInfos()[indicesToCollect[indicesToCollect.size() - 1]].MaximumValue();
        }
        break;
    case 2: // global RGB
        break;
    }

    bool vx, vy, vz;
    vx = vy = vz = false;
    std::string c = cleanUpColumnHeader(this->slotColumnVX.Param<core::param::FlexEnumParam>()->ValueString());
    if (this->columnIndex.find(c) != columnIndex.end()) vx = true;
    c = cleanUpColumnHeader(this->slotColumnVY.Param<core::param::FlexEnumParam>()->ValueString());
    if (this->columnIndex.find(c) != columnIndex.end()) vy = true;
    c = cleanUpColumnHeader(this->slotColumnVZ.Param<core::param::FlexEnumParam>()->ValueString());
    if (this->columnIndex.find(c) != columnIndex.end()) vz = true;

    if (vx && vy && vz) {
        retValue = retValue && pushColumnIndex(indicesToCollect,
                                   this->slotColumnVX.Param<core::param::FlexEnumParam>()->ValueString());
        retValue = retValue && pushColumnIndex(indicesToCollect,
                                   this->slotColumnVY.Param<core::param::FlexEnumParam>()->ValueString());
        retValue = retValue && pushColumnIndex(indicesToCollect,
                                   this->slotColumnVZ.Param<core::param::FlexEnumParam>()->ValueString());

        haveVelocities = true;
        stride += 3;
    }

    this->haveTensor = true;
    // this means that we have explicit magnitudes and tensor rows are normalized!
    this->haveTensorMagnitudes = true;
    std::array<uint32_t, 9> tensorIndices{};
    for (uint32_t tensorIdx = 0; tensorIdx < slotTensorColumn.size(); ++tensorIdx) {
        auto& x = slotTensorColumn[tensorIdx];
        c = cleanUpColumnHeader(x.Param<core::param::FlexEnumParam>()->ValueString());
        if (this->columnIndex.find(c) == columnIndex.end()) {
            this->haveTensor = false;
        } else {
            tensorIndices[tensorIdx] = this->columnIndex[c];
        }
    }
    for (auto& x : this->slotTensorMagnitudeColumn) {
        c = cleanUpColumnHeader(x.Param<core::param::FlexEnumParam>()->ValueString());
        if (this->columnIndex.find(c) == columnIndex.end()) this->haveTensorMagnitudes = false;
    }
    if (this->haveTensor) {
        // we are always going to pass magnitudes per 'basis vector' (3) and a quaternion (4)
        stride += 7;
        if (this->haveTensorMagnitudes) {
            // we can copy the radii AKA magnitudes directly
            for (auto& x : this->slotTensorMagnitudeColumn) {
                pushColumnIndex(indicesToCollect, x.Param<core::param::FlexEnumParam>()->ValueString());
            }
        }
    }


    everything.resize(ft->GetRowsCount() * stride);
    uint64_t rows = ft->GetRowsCount();
    uint32_t cols = ft->GetColumnsCount();


    const float* ftData = ft->GetData();
    uint32_t numIndices = indicesToCollect.size();
    if (retValue) {
        if (this->haveTensor) {
            if (!this->haveTensorMagnitudes) {
                ASSERT(numIndices + 7 == stride);
            } else {
                ASSERT(numIndices + 4 == stride);
            }
        } else {
            ASSERT(numIndices == stride);
        }
    }
    uint32_t tensorOffset;
    if (haveTensorMagnitudes) {
        tensorOffset = numIndices;
    } else {
        tensorOffset = numIndices + 3;
    }
    for (uint32_t i = 0; i < ft->GetRowsCount(); i++) {
        float* currOut = &everything[i * stride];
        for (uint32_t j = 0; j < numIndices; j++) {
            currOut[j] = ftData[cols * i + indicesToCollect[j]];
        }
        if (this->haveTensor) {
            glm::mat3 tensor;
            glm::vec3 xvec, yvec, zvec;
            for (int offset = 0; offset < 9; ++offset) {
                tensor[offset % 3][offset / 3] = ftData[cols * i + tensorIndices[offset]];
            }
            if (!this->haveTensorMagnitudes) {
                // TODO compute, save, pass pointer later.
                for (int offset = 0; offset < 3; ++offset) {
                    xvec[offset] = ftData[cols * i + tensorIndices[offset]];
                }
                for (int offset = 3; offset < 6; ++offset) {
                    yvec[offset - 3] = ftData[cols * i + tensorIndices[offset]];
                }
                for (int offset = 6; offset < 9; ++offset) {
                    zvec[offset - 6] = ftData[cols * i + tensorIndices[offset]];
                }
                currOut[numIndices + 0] = glm::length(xvec);
                currOut[numIndices + 1] = glm::length(yvec);
                currOut[numIndices + 2] = glm::length(zvec);

            }
            auto quat = glm::quat_cast(tensor);
            //memcpy(&currOut[numIndices + 3], glm::value_ptr(quat), sizeof(float) * 4);
            currOut[tensorOffset + 0] = quat.x;
            currOut[tensorOffset + 1] = quat.y;
            currOut[tensorOffset + 2] = quat.z;
            currOut[tensorOffset + 3] = quat.w;
            //currOut[tensorOffset - 3] = 3.0f;
            //currOut[tensorOffset - 2] = 2.0f;
            //currOut[tensorOffset - 1] = 1.0f;

            //currOut[tensorOffset + 0] = 0.0f;
            //currOut[tensorOffset + 1] = 0.0f;
            //currOut[tensorOffset + 2] = 0.0f;
            //currOut[tensorOffset + 3] = 1.0f;
            // TODO make mat3, convert to quat, save, pass pointer later.
        }
    }

    for (uint32_t i = 0; i < (numIndices < 3 ? numIndices : 3); i++) {
        this->bboxMin[i] = ft->GetColumnsInfos()[indicesToCollect[i]].MinimumValue() - radius;
        this->bboxMax[i] = ft->GetColumnsInfos()[indicesToCollect[i]].MaximumValue() + radius;
    }

    this->myHash++;
    this->resetAllDirty();
    this->inputHash = ft->DataHash();
    this->myTime = ft->GetFrameID();
    return retValue;
}

/*
 * megamol::pcl::PclDataSource::getMultiParticleData
 */
bool TableToParticles::getMultiParticleData(core::Call& call) {
    try {
        auto* c = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&call);
        auto* e = dynamic_cast<core::moldyn::EllipsoidalParticleDataCall*>(&call);
        auto* ft = this->slotCallTable.CallAs<table::TableDataCall>();
        if (ft == nullptr) return false;
        (*ft)();

        if (!assertData(ft)) return false;

        if (c != nullptr) {
            c->SetFrameCount(1);
            c->SetFrameID(0);
            c->SetDataHash(this->myHash);

            c->SetExtent(1, this->bboxMin[0], this->bboxMin[1], this->bboxMin[2], this->bboxMax[0], this->bboxMax[1],
                this->bboxMax[2]);
            c->SetParticleListCount(1);
            c->AccessParticles(0).SetCount(ft->GetRowsCount());
            c->AccessParticles(0).SetGlobalRadius(this->slotGlobalRadius.Param<core::param::FloatParam>()->Value());
            if (!this->slotGlobalColor.Param<core::param::StringParam>()->Value().IsEmpty()) {
                float r, g, b;
                core::utility::ColourParser::FromString(
                    this->slotGlobalColor.Param<core::param::StringParam>()->Value(), r, g, b);
                c->AccessParticles(0).SetGlobalColour(static_cast<unsigned int>(r * 255.0f),
                    static_cast<unsigned int>(g * 255.0f), static_cast<unsigned int>(b * 255.0f));
            }

            if (ft->GetRowsCount() > 0) {
                uint32_t colOffset = 0;
                switch (this->slotRadiusMode.Param<core::param::EnumParam>()->Value()) {
                case 0: // per particle
                    c->AccessParticles(0).SetVertexData(
                        megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR,
                        this->everything.data(), static_cast<unsigned int>(stride * sizeof(float)));
                    colOffset = 4;
                    break;
                case 1: // global
                    c->AccessParticles(0).SetVertexData(
                        megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ,
                        this->everything.data(), static_cast<unsigned int>(stride * sizeof(float)));
                    colOffset = 3;
                    break;
                }

                switch (this->slotColorMode.Param<core::param::EnumParam>()->Value()) {
                case 0: // RGB
                    c->AccessParticles(0).SetColourData(
                        megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB,
                        this->everything.data() + colOffset, static_cast<unsigned int>(stride * sizeof(float)));
                    break;
                case 1: // I
                    c->AccessParticles(0).SetColourData(
                        megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I,
                        this->everything.data() + colOffset, static_cast<unsigned int>(stride * sizeof(float)));
                    c->AccessParticles(0).SetColourMapIndexValues(iMin, iMax);
                    break;
                case 2: // global RGB
                    c->AccessParticles(0).SetColourData(
                        megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE, nullptr);
                    break;
                }
                if (haveVelocities) {
                    c->AccessParticles(0).SetDirData(core::moldyn::MultiParticleDataCall::Particles::DIRDATA_FLOAT_XYZ,
                        this->everything.data() + (stride - 3), static_cast<unsigned int>(stride * sizeof(float)));
                }
            }
            c->SetUnlocker(nullptr);
        } else if (e != nullptr) {
            e->SetFrameCount(1);
            e->SetFrameID(0);
            e->SetDataHash(this->myHash);

            e->SetExtent(1, this->bboxMin[0], this->bboxMin[1], this->bboxMin[2], this->bboxMax[0], this->bboxMax[1],
                this->bboxMax[2]);
            e->SetParticleListCount(1);
            e->AccessParticles(0).SetCount(ft->GetRowsCount());
            e->AccessParticles(0).SetGlobalRadius(this->slotGlobalRadius.Param<core::param::FloatParam>()->Value());
            if (!this->slotGlobalColor.Param<core::param::StringParam>()->Value().IsEmpty()) {
                float r, g, b;
                core::utility::ColourParser::FromString(
                    this->slotGlobalColor.Param<core::param::StringParam>()->Value(), r, g, b);
                e->AccessParticles(0).SetGlobalColour(static_cast<unsigned int>(r * 255.0f),
                    static_cast<unsigned int>(g * 255.0f), static_cast<unsigned int>(b * 255.0f));
            }

            if (ft->GetRowsCount() > 0) {
                uint32_t colOffset = 0;
                switch (this->slotRadiusMode.Param<core::param::EnumParam>()->Value()) {
                case 0: // per particle
                    e->AccessParticles(0).SetVertexData(
                        megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR,
                        this->everything.data(), static_cast<unsigned int>(stride * sizeof(float)));
                    colOffset = 4;
                    break;
                case 1: // global
                    e->AccessParticles(0).SetVertexData(
                        megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ,
                        this->everything.data(), static_cast<unsigned int>(stride * sizeof(float)));
                    colOffset = 3;
                    break;
                }

                switch (this->slotColorMode.Param<core::param::EnumParam>()->Value()) {
                case 0: // RGB
                    e->AccessParticles(0).SetColourData(
                        megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB,
                        this->everything.data() + colOffset, static_cast<unsigned int>(stride * sizeof(float)));
                    break;
                case 1: // I
                    e->AccessParticles(0).SetColourData(
                        megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I,
                        this->everything.data() + colOffset, static_cast<unsigned int>(stride * sizeof(float)));
                    e->AccessParticles(0).SetColourMapIndexValues(iMin, iMax);
                    break;
                case 2: // global RGB
                    e->AccessParticles(0).SetColourData(
                        megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE, nullptr);
                    break;
                }
                //if (haveVelocities) {
                //    e->AccessParticles(0).SetDirData(core::moldyn::MultiParticleDataCall::Particles::DIRDATA_FLOAT_XYZ,
                //        this->everything.data() + (stride - 3), static_cast<unsigned int>(stride * sizeof(float)));
                //}

                if (haveTensor) {
                    e->AccessParticles(0).SetRadData(
                        this->everything.data() + (stride - 7), static_cast<unsigned int>(stride * sizeof(float)));
                    e->AccessParticles(0).SetQuatData(
                        this->everything.data() + (stride - 4), static_cast<unsigned int>(stride * sizeof(float)));
                } else {
                    e->AccessParticles(0).SetRadData(nullptr, static_cast<unsigned int>(stride * sizeof(float)));
                    e->AccessParticles(0).SetQuatData(nullptr, static_cast<unsigned int>(stride * sizeof(float)));
                }
            }
            e->SetUnlocker(nullptr);
        }
        return true;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteError(1, e.GetMsg());
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError(1, _T("Unexpected exception ")
                                                   _T("in callback getMultiParticleData."));
        return false;
    }
}


/*
 * megamol::pcl::PclDataSource::getMultiparticleExtent
 */
bool TableToParticles::getMultiparticleExtent(core::Call& call) {
    try {
        auto* c = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&call);
        auto* e = dynamic_cast<core::moldyn::EllipsoidalParticleDataCall*>(&call);
        table::TableDataCall* ft = this->slotCallTable.CallAs<table::TableDataCall>();
        if (ft == nullptr) return false;

        if (c != nullptr) {
            ft->SetFrameID(c->FrameID());
            (*ft)();
            if (!assertData(ft)) return false;

            c->SetFrameCount(ft->GetFrameCount());
            c->SetFrameID(ft->GetFrameID());
            c->SetDataHash(this->myHash);

            c->SetExtent(1, this->bboxMin[0], this->bboxMin[1], this->bboxMin[2], this->bboxMax[0], this->bboxMax[1],
                this->bboxMax[2]);
            c->SetUnlocker(NULL);
        } else if (e != nullptr) {
            ft->SetFrameID(e->FrameID());
            (*ft)();
            if (!assertData(ft)) return false;

            e->SetFrameCount(ft->GetFrameCount());
            e->SetFrameID(ft->GetFrameID());
            e->SetDataHash(this->myHash);

            e->SetExtent(1, this->bboxMin[0], this->bboxMin[1], this->bboxMin[2], this->bboxMax[0], this->bboxMax[1],
                this->bboxMax[2]);
            e->SetUnlocker(NULL);
        }
        return true;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteError(1, e.GetMsg());
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError(1, _T("Unexpected exception ")
                                                   _T("in callback getMultiparticleExtent."));
        return false;
    }
}


/*
 * megamol::pcl::PclDataSource::release
 */
void TableToParticles::release(void) {}
