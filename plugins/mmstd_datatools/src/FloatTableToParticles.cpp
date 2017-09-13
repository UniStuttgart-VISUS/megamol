#include "stdafx.h"
#include "FloatTableToParticles.h"

#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/utility/ColourParser.h"

#include "vislib/sys/Log.h"
#include "vislib/sys/PerformanceCounter.h"
#include "vislib/Trace.h"


using namespace megamol::stdplugin::datatools;
using namespace megamol;

/*
 * FloatTableToParticles::FloatTableToParticles
 */
FloatTableToParticles::FloatTableToParticles(void) : Module(),
        slotCallMultiPart("multidata", "Provides the data as MultiParticle call."),
        slotCallFloatTable("floattable", "float table input call"),
        slotColumnR("redcolumnname", "The name of the column holding the red colour channel value."),
        slotColumnG("greencolumnname", "The name of the column holding the green colour channel value."),
        slotColumnB("bluecolumnname", "The name of the column holding the blue colour channel value."),
        slotColumnI("intensitycolumnname", "The name of the column holding the intensity colour channel value."),
        slotGlobalColor("color", "Constant sphere color."),
        slotColorMode("colormode", "Pass on color as RGB or intensity"),
        slotColumnRadius("radiuscolumnname", "The name of the column holding the particle radius."),
        slotGlobalRadius("radius", "Constant sphere radius."),
        slotRadiusMode("radiusmode", "pass on global or per-particle radius."),
        slotColumnX("xcolumnname", "The name of the column holding the x-coordinate."),
        slotColumnY("ycolumnname", "The name of the column holding the y-coordinate."),
        slotColumnZ("zcolumnname", "The name of the column holding the z-coordinate."),
        inputHash(0), myHash(0), columnIndex() {

    /* Register parameters. */
    core::param::FlexEnumParam *rColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnR << rColumnEp;
    this->MakeSlotAvailable(&this->slotColumnR);

    core::param::FlexEnumParam *gColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnG << gColumnEp;
    this->MakeSlotAvailable(&this->slotColumnG);

    core::param::FlexEnumParam *bColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnB << bColumnEp;
    this->MakeSlotAvailable(&this->slotColumnB);

    core::param::FlexEnumParam *iColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnI << iColumnEp;
    this->MakeSlotAvailable(&this->slotColumnI);

    this->slotGlobalColor << new megamol::core::param::StringParam(_T(""));
    this->MakeSlotAvailable(&this->slotGlobalColor);

    core::param::EnumParam *ep = new core::param::EnumParam(2);
    ep->SetTypePair(0, "RGB");
    ep->SetTypePair(1, "Intensity");
    ep->SetTypePair(2, "global RGB");
    this->slotColorMode << ep;
    this->MakeSlotAvailable(&this->slotColorMode);

    core::param::FlexEnumParam *columnRadiusEp = new core::param::FlexEnumParam("undef");
    this->slotColumnRadius << columnRadiusEp;
    this->MakeSlotAvailable(&this->slotColumnRadius);

    this->slotGlobalRadius << new megamol::core::param::FloatParam(0.001f);
    this->MakeSlotAvailable(&this->slotGlobalRadius);

    core::param::EnumParam *ep2 = new core::param::EnumParam(0);
    ep2->SetTypePair(0, "per particle");
    ep2->SetTypePair(1, "global");
    this->slotRadiusMode << ep2;
    this->MakeSlotAvailable(&this->slotRadiusMode);

    //this->slotColumnX << new megamol::core::param::StringParam(_T("x"));
    core::param::FlexEnumParam *xColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnX << xColumnEp;
    this->MakeSlotAvailable(&this->slotColumnX);

    core::param::FlexEnumParam *yColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnY << yColumnEp;
    this->MakeSlotAvailable(&this->slotColumnY);

    core::param::FlexEnumParam *zColumnEp = new core::param::FlexEnumParam("undef");
    this->slotColumnZ << zColumnEp;
    this->MakeSlotAvailable(&this->slotColumnZ);

    /* Register calls. */
    this->slotCallMultiPart.SetCallback(
        core::moldyn::MultiParticleDataCall::ClassName(),
        "GetData",
        &FloatTableToParticles::getMultiParticleData);
    this->slotCallMultiPart.SetCallback(
        core::moldyn::MultiParticleDataCall::ClassName(),
        "GetExtent",
        &FloatTableToParticles::getMultiparticleExtent);
    this->MakeSlotAvailable(&this->slotCallMultiPart);

    this->slotCallFloatTable.SetCompatibleCall<floattable::CallFloatTableDataDescription>();
    this->MakeSlotAvailable(&this->slotCallFloatTable);
}


/*
 * FloatTableToParticles::~FloatTableToParticles
 */
FloatTableToParticles::~FloatTableToParticles(void) {
    this->Release();
}


/*
 * megamol::pcl::PclDataSource::create
 */
bool FloatTableToParticles::create(void) {
    bool retval = true;
    return true;
}

bool FloatTableToParticles::anythingDirty() {
    return this->slotColumnR.IsDirty()
        || this->slotColumnG.IsDirty()
        || this->slotColumnB.IsDirty()
        || this->slotColumnI.IsDirty()
        || this->slotGlobalColor.IsDirty()
        || this->slotColorMode.IsDirty()
        || this->slotColumnRadius.IsDirty()
        || this->slotGlobalRadius.IsDirty()
        || this->slotRadiusMode.IsDirty()
        || this->slotColumnX.IsDirty()
        || this->slotColumnY.IsDirty()
        || this->slotColumnZ.IsDirty();
}

void FloatTableToParticles::resetAllDirty() {
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
}

std::string FloatTableToParticles::cleanUpColumnHeader(const std::string& header) const {
    return this->cleanUpColumnHeader(vislib::TString(header.data()));
}

std::string FloatTableToParticles::cleanUpColumnHeader(const vislib::TString& header) const {
    vislib::TString h(header);
    h.TrimSpaces();
    h.ToLowerCase();
    return std::string(T2A(h.PeekBuffer()));
}

bool FloatTableToParticles::pushColumnIndex(std::vector<size_t>& cols, const vislib::TString& colName) {
    std::string c = cleanUpColumnHeader(colName);
    if (this->columnIndex.find(c) != columnIndex.end()) {
        cols.push_back(columnIndex[c]);
        return true;
    } else {
        vislib::sys::Log::DefaultLog.WriteError("unknown column '%s'", c.c_str());
        return false;
    }
}

bool FloatTableToParticles::assertData(floattable::CallFloatTableData *ft) {
    if (this->inputHash == ft->DataHash() && !anythingDirty()) return true;

    if (this->inputHash != ft->DataHash()) {
        vislib::sys::Log::DefaultLog.WriteInfo("FloatTableToParticles: Dataset changed -> Updating EnumParams\n");
        this->columnIndex.clear();

        this->slotColumnX.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnY.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnZ.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnR.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnG.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnB.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnI.Param<core::param::FlexEnumParam>()->ClearValues();
        this->slotColumnRadius.Param<core::param::FlexEnumParam>()->ClearValues();

        for (size_t i = 0; i < ft->GetColumnsCount(); i++) {
            std::string n = std::string(this->cleanUpColumnHeader(ft->GetColumnsInfos()[i].Name()));
            columnIndex[n] = i;

            this->slotColumnX.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnY.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnZ.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnR.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnG.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnB.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnI.Param<core::param::FlexEnumParam>()->AddValue(n);
            this->slotColumnRadius.Param<core::param::FlexEnumParam>()->AddValue(n);
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

    everything.reserve(ft->GetRowsCount() * stride);
    size_t rows = ft->GetRowsCount();
    size_t cols = ft->GetColumnsCount();

    bool retValue = true;

    std::vector<size_t> indicesToCollect;
    if (!pushColumnIndex(indicesToCollect, this->slotColumnX.Param<core::param::FlexEnumParam>()->ValueString())) {
        retValue = false;
    }
    if (!pushColumnIndex(indicesToCollect, this->slotColumnY.Param<core::param::FlexEnumParam>()->ValueString())) {
        retValue = false;
    }
    if (!pushColumnIndex(indicesToCollect, this->slotColumnZ.Param<core::param::FlexEnumParam>()->ValueString())) {
        retValue = false;
    }
    if (this->slotRadiusMode.Param<core::param::EnumParam>()->Value() == 0) {
        if (!pushColumnIndex(indicesToCollect, this->slotColumnRadius.Param<core::param::FlexEnumParam>()->ValueString())) {
            retValue = false;
        }
    }
    switch (this->slotColorMode.Param<core::param::EnumParam>()->Value()) {
        case 0: // RGB
            if (!pushColumnIndex(indicesToCollect, this->slotColumnR.Param<core::param::FlexEnumParam>()->ValueString())) {
                retValue = false;
            }
            if (!pushColumnIndex(indicesToCollect, this->slotColumnG.Param<core::param::FlexEnumParam>()->ValueString())) {
                retValue = false;
            }
            if (!pushColumnIndex(indicesToCollect, this->slotColumnB.Param<core::param::FlexEnumParam>()->ValueString())) {
                retValue = false;
            }
            break;
        case 1: // I
            if (!pushColumnIndex(indicesToCollect, this->slotColumnI.Param<core::param::FlexEnumParam>()->ValueString())) {
                retValue = false;
            }
            iMin = ft->GetColumnsInfos()[indicesToCollect[indicesToCollect.size() - 1]].MinimumValue();
            iMax = ft->GetColumnsInfos()[indicesToCollect[indicesToCollect.size() - 1]].MaximumValue();
            break;
        case 2: // global RGB
            break;
    }

    const float *ftData = ft->GetData();
    size_t numIndices = indicesToCollect.size();
    for (size_t i = 0; i < ft->GetRowsCount(); i++) {
        float *currOut = &everything.data()[i * stride];
        for (size_t j = 0; j < numIndices; j++) {
            currOut[j] = ftData[cols * i + indicesToCollect[j]];
        }
    }

    for (size_t i = 0; i < (numIndices < 3 ? numIndices : 3); i++) {
        this->bboxMin[i] = ft->GetColumnsInfos()[indicesToCollect[i]].MinimumValue();
        this->bboxMax[i] = ft->GetColumnsInfos()[indicesToCollect[i]].MaximumValue();
    }

    this->myHash++;
    this->resetAllDirty();
    this->inputHash = ft->DataHash();
    return retValue;
}

/*
 * megamol::pcl::PclDataSource::getMultiParticleData
 */
bool FloatTableToParticles::getMultiParticleData(core::Call& call) {
    try {
        core::moldyn::MultiParticleDataCall& c = dynamic_cast<
            core::moldyn::MultiParticleDataCall&>(call);
        floattable::CallFloatTableData *ft = this->slotCallFloatTable.CallAs<floattable::CallFloatTableData>();
        if (ft == NULL) return false;
        (*ft)();

        if (!assertData(ft)) return false;

        c.SetFrameCount(1);
        c.SetFrameID(0);
        c.SetDataHash(this->myHash);

        c.SetExtent(1,
            this->bboxMin[0], this->bboxMin[1], this->bboxMin[2],
            this->bboxMax[0], this->bboxMax[1], this->bboxMax[2]);
        c.SetParticleListCount(1);
        c.AccessParticles(0).SetCount(ft->GetRowsCount());
        c.AccessParticles(0).SetGlobalRadius(this->slotGlobalRadius.Param<
            core::param::FloatParam>()->Value());
        if (!this->slotGlobalColor.Param<core::param::StringParam>()->Value().IsEmpty()) {
            float r, g, b;
            core::utility::ColourParser::FromString(this->slotGlobalColor.Param<core::param::StringParam>()->Value(), r, g, b);
            c.AccessParticles(0).SetGlobalColour(static_cast<unsigned int>(r * 255.0f),
                static_cast<unsigned int>(g * 255.0f), static_cast<unsigned int>(b * 255.0f));
        }

        size_t colOffset = 0;
        switch (this->slotRadiusMode.Param<core::param::EnumParam>()->Value()) {
            case 0: // per particle
                c.AccessParticles(0).SetVertexData(megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR, this->everything.data(), static_cast<unsigned int>(stride * sizeof(float)));
                colOffset = 4;
                break;
            case 1: // global
                c.AccessParticles(0).SetVertexData(megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ, this->everything.data(), static_cast<unsigned int>(stride * sizeof(float)));
                colOffset = 3;
                break;
        }

        switch (this->slotColorMode.Param<core::param::EnumParam>()->Value()) {
            case 0: // RGB
                c.AccessParticles(0).SetColourData(megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB, this->everything.data() + colOffset, static_cast<unsigned int>(stride * sizeof(float)));
                break;
            case 1: // I
                c.AccessParticles(0).SetColourData(megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I, this->everything.data() + colOffset, static_cast<unsigned int>(stride * sizeof(float)));
                c.AccessParticles(0).SetColourMapIndexValues(iMin, iMax);
                break;
            case 2: // global RGB
                c.AccessParticles(0).SetColourData(megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE, nullptr);
                break;
        }

        c.SetUnlocker(NULL);

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
bool FloatTableToParticles::getMultiparticleExtent(core::Call& call) {
    try {
        core::moldyn::MultiParticleDataCall& c = dynamic_cast<
            core::moldyn::MultiParticleDataCall&>(call);
        floattable::CallFloatTableData *ft = this->slotCallFloatTable.CallAs<floattable::CallFloatTableData>();
        if (ft == NULL) return false;
        (*ft)();

        if (!assertData(ft)) return false;

        c.SetFrameCount(1);
        c.SetFrameID(0);
        c.SetDataHash(this->myHash);

        c.SetExtent(1,
            this->bboxMin[0], this->bboxMin[1], this->bboxMin[2],
            this->bboxMax[0], this->bboxMax[1], this->bboxMax[2]);
        c.SetUnlocker(NULL);
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
void FloatTableToParticles::release(void) {
}
