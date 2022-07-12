#include "TableObserverPlane.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmstd/renderer/CallClipPlane.h"

#include "mmcore/utility/log/Log.h"
#include "vislib/Trace.h"
#include "vislib/math/Matrix4.h"
#include "vislib/math/Plane.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/sys/PerformanceCounter.h"
#include <cmath>

using namespace megamol::datatools;
using namespace megamol::datatools::table;
using namespace megamol;

/*
 * TableToParticles::TableObserverPlane
 */
TableObserverPlane::TableObserverPlane(void)
        : Module()
        , slotCallInputTable("table", "table input call")
        , slotCallClipPlane("clipplabe", "clip plane input call")
        , slotCallObservedTable("observation", "resulting table")
        , slotColumnX("xcolumnname", "The name of the column holding the x-coordinate.")
        , slotColumnY("ycolumnname", "The name of the column holding the y-coordinate.")
        , slotColumnZ("zcolumnname", "The name of the column holding the z-coordinate.")
        , slotColumnRadius("radiuscolumnname", "The name of the column holding the particle radius.")
        , slotGlobalRadius("radius", "Constant sphere radius.")
        , slotRadiusMode("radiusmode", "pass on global or per-particle radius.")
        , slotObservationStrategy("strategy", "The way of choosing particles.")
        , slotStartTime("startTime", "when to start observing data")
        , slotEndTime("endTime", "when to stop observing data")
        ,
        //slotTimeIncrement("timeIncrement", "time step"),
        slotSliceOffset("sliceOffset", "offset between observations in resulting stack")
        , inputHash(0)
        , myHash(0)
        , columnIndex()
        , frameID(-1)
        , everything() {

    /* Register parameters. */
    core::param::EnumParam* ep = new core::param::EnumParam(0);
    ep->SetTypePair(0, "intersection");
    ep->SetTypePair(1, "interpolation");
    this->slotObservationStrategy << ep;
    this->MakeSlotAvailable(&this->slotObservationStrategy);

    this->slotStartTime << new megamol::core::param::IntParam(0);
    this->MakeSlotAvailable(&this->slotStartTime);

    this->slotEndTime << new megamol::core::param::IntParam(9999999);
    this->MakeSlotAvailable(&this->slotEndTime);

    //this->slotTimeIncrement << new megamol::core::param::FloatParam(1.0f);
    //this->MakeSlotAvailable(&this->slotTimeIncrement);

    this->slotSliceOffset << new megamol::core::param::FloatParam(0.5f);
    this->MakeSlotAvailable(&this->slotSliceOffset);

    this->slotColumnX << new megamol::core::param::StringParam(_T("x"));
    this->MakeSlotAvailable(&this->slotColumnX);

    this->slotColumnY << new megamol::core::param::StringParam(_T("y"));
    this->MakeSlotAvailable(&this->slotColumnY);

    this->slotColumnZ << new megamol::core::param::StringParam(_T("z"));
    this->MakeSlotAvailable(&this->slotColumnZ);

    this->slotColumnRadius << new megamol::core::param::StringParam(_T(""));
    this->MakeSlotAvailable(&this->slotColumnRadius);

    this->slotGlobalRadius << new megamol::core::param::FloatParam(0.001f);
    this->MakeSlotAvailable(&this->slotGlobalRadius);

    core::param::EnumParam* ep2 = new core::param::EnumParam(0);
    ep2->SetTypePair(0, "per particle");
    ep2->SetTypePair(1, "global");
    this->slotRadiusMode << ep2;
    this->MakeSlotAvailable(&this->slotRadiusMode);

    /* Register calls. */
    this->slotCallObservedTable.SetCallback(
        datatools::table::TableDataCall::ClassName(), "GetData", &TableObserverPlane::getObservedData);
    this->slotCallObservedTable.SetCallback(
        datatools::table::TableDataCall::ClassName(), "GetHash", &TableObserverPlane::getHash);
    this->MakeSlotAvailable(&this->slotCallObservedTable);

    this->slotCallInputTable.SetCompatibleCall<table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->slotCallInputTable);
    this->slotCallClipPlane.SetCompatibleCall<megamol::core::view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->slotCallClipPlane);
}


/*
 * TableToParticles::~TableToParticles
 */
TableObserverPlane::~TableObserverPlane(void) {
    this->Release();
}


/*
 * megamol::pcl::PclDataSource::create
 */
bool TableObserverPlane::create(void) {
    bool retval = true;
    return true;
}

bool TableObserverPlane::anythingDirty() {
    return this->slotColumnX.IsDirty() || this->slotColumnY.IsDirty() || this->slotColumnZ.IsDirty() ||
           this->slotColumnRadius.IsDirty() || this->slotGlobalRadius.IsDirty() || this->slotRadiusMode.IsDirty() ||
           this->slotObservationStrategy.IsDirty() || this->slotStartTime.IsDirty() ||
           this->slotEndTime.IsDirty()
           //|| this->slotTimeIncrement.IsDirty()
           || this->slotSliceOffset.IsDirty();
}

void TableObserverPlane::resetAllDirty() {
    this->slotColumnX.ResetDirty();
    this->slotColumnY.ResetDirty();
    this->slotColumnZ.ResetDirty();
    this->slotColumnRadius.ResetDirty();
    this->slotGlobalRadius.ResetDirty();
    this->slotRadiusMode.ResetDirty();
    this->slotObservationStrategy.ResetDirty();
    this->slotStartTime.ResetDirty();
    this->slotEndTime.ResetDirty();
    //this->slotTimeIncrement.ResetDirty();
    this->slotSliceOffset.ResetDirty();
}

std::string TableObserverPlane::cleanUpColumnHeader(const std::string& header) const {
    return this->cleanUpColumnHeader(vislib::TString(header.data()));
}

std::string TableObserverPlane::cleanUpColumnHeader(const vislib::TString& header) const {
    vislib::TString h(header);
    h.TrimSpaces();
    h.ToLowerCase();
    return std::string(vislib::StringA(h).PeekBuffer());
}

int TableObserverPlane::getColumnIndex(const vislib::TString& colName) {
    std::string c = cleanUpColumnHeader(colName);
    if (this->columnIndex.find(c) != columnIndex.end()) {
        return static_cast<int>(columnIndex[c]);
    } else {
        return -1;
    }
}

bool TableObserverPlane::pushColumnIndex(std::vector<size_t>& cols, const vislib::TString& colName) {
    std::string c = cleanUpColumnHeader(colName);
    if (this->columnIndex.find(c) != columnIndex.end()) {
        cols.push_back(columnIndex[c]);
        return true;
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError("unknown column '%s'", c.c_str());
        return false;
    }
}

bool TableObserverPlane::assertData(
    table::TableDataCall* ft, megamol::core::view::CallClipPlane* cp, table::TableDataCall& out) {
    if (this->inputHash == ft->DataHash() && !anythingDirty())
        return true;
    (*ft)();
    if (this->inputHash != ft->DataHash()) {
        this->columnIndex.clear();
        for (size_t i = 0; i < ft->GetColumnsCount(); i++) {
            std::string n = this->cleanUpColumnHeader(ft->GetColumnsInfos()[i].Name());
            columnIndex[n] = i;
        }
    }

    // count planes
    int start = this->slotStartTime.Param<core::param::IntParam>()->Value();
    int end = this->slotEndTime.Param<core::param::IntParam>()->Value();
    end = std::min<int>(end, ft->GetFrameCount() - 1);
    this->slotEndTime.Param<core::param::IntParam>()->SetValue(end, false); // TODO is this bad?
    //float incr = this->slotTimeIncrement.Param<core::param::FloatParam>()->Value();
    float sliceOff = this->slotSliceOffset.Param<core::param::FloatParam>()->Value();
    int numFrames = end - start + 1;
    int xcol = this->getColumnIndex(this->slotColumnX.Param<core::param::StringParam>()->Value().c_str());
    int ycol = this->getColumnIndex(this->slotColumnY.Param<core::param::StringParam>()->Value().c_str());
    int zcol = this->getColumnIndex(this->slotColumnZ.Param<core::param::StringParam>()->Value().c_str());
    int strat = this->slotObservationStrategy.Param<core::param::EnumParam>()->Value();

    if (xcol == -1 || ycol == -1 || zcol == -1)
        return false;

    int rcol = this->getColumnIndex(this->slotColumnRadius.Param<core::param::StringParam>()->Value().c_str());
    bool useGlobRad = this->slotRadiusMode.Param<core::param::EnumParam>()->Value() == 1;
    if (!useGlobRad && rcol == -1)
        return false;
    float globalRad = this->slotGlobalRadius.Param<core::param::FloatParam>()->Value();

    size_t rows = ft->GetRowsCount();
    size_t cols = ft->GetColumnsCount();

    (*cp)();
    auto p = cp->GetPlane();


    out.SetFrameCount(1);

    ft->SetFrameID(0);
    (*ft)();
    while (ft->GetFrameID() != 0) {
        (*ft)();
    }

    vislib::math::Vector<float, 4> pt;
    vislib::math::Vector<float, 4> resPt;
    vislib::math::ShallowPoint<float, 3> spt(pt.PeekComponents());
    vislib::math::Point<float, 3> planeCenter;
    vislib::math::Point<float, 3> origin(0.0f, 0.0f, 0.0f);
    vislib::math::Vector<float, 3> longNormal(p.Normal());
    longNormal.ScaleToLength(p.D() * 2.0f);
    auto res = p.Intersect(planeCenter, origin, origin + longNormal);
    ASSERT(res == 1);

    vislib::math::Vector<float, 3> reversedNormal(p.Normal());
    reversedNormal.ScaleToLength(-1.0f);
    vislib::math::Vector<float, 3> up(0.0f, 1.0f, 0.0f);
    vislib::math::Vector<float, 3> u = up.Cross(reversedNormal);
    vislib::math::Vector<float, 3> v = reversedNormal.Cross(u);
    vislib::math::Vector<float, 3> w = p.Normal();
    u.Normalise();
    v.Normalise();
    w.Normalise();
    vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> mPlaneCoords(u.X(), v.X(), w.X(), 0.0f, u.Y(), v.Y(),
        w.Y(), 0.0f, u.Z(), v.Z(), w.Z(), 0.0f, -planeCenter.X(), -planeCenter.Y(), -planeCenter.Z(), 1.0f);

    const float* ftData = ft->GetData();
    float r = globalRad;
    everything.clear();
    everything.reserve(rows * cols);
    size_t totalRows = 0;
    size_t localRows;
    for (int frm = 0; frm < numFrames; frm++) {
        int time = frm + start;
        ft->SetFrameID(time);
        (*ft)();
        while (ft->GetFrameID() != time) {
            (*ft)();
        }
        rows = ft->GetRowsCount();
        localRows = 0;
        if (rows == 0)
            continue;

        if (ft->GetColumnsCount() != cols) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "TableObserverPlane cannot cope with changing column count!");
            return false;
        }
        // check whether the headers are the same
        // alternatively just build the index every frame...
        for (size_t i = 0; i < ft->GetColumnsCount(); i++) {
            std::string n = this->cleanUpColumnHeader(ft->GetColumnsInfos()[i].Name());
            if (columnIndex.find(n) == columnIndex.end() || columnIndex[n] != i) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "TableObserverPlane does not trust reordered columns!");
                return false;
            }
        }

        float dist = 0.0f, val = 0.0f;
        for (size_t row = 0; row < rows; row++) {
            pt.Set(ftData[cols * row + xcol], ftData[cols * row + ycol], ftData[cols * row + zcol], 1.0f);
            //spt.SetPointer(pt.PeekCoordinates());
            if (!useGlobRad) {
                r = ftData[cols * row + rcol];
            }
            switch (strat) {
            case 0: // intersection
                dist = p.Distance(spt);
                if (std::abs(dist) <= r) {
                    resPt = mPlaneCoords * pt;
                    resPt.SetZ(resPt.Z() + sliceOff * static_cast<float>(frm));
                    for (size_t col = 0; col < cols; col++) {
                        if (col == xcol) {
                            val = resPt.X();
                        } else if (col == ycol) {
                            val = resPt.Y();
                        } else if (col == zcol) {
                            val = resPt.Z();
                        } else {
                            val = ftData[cols * row + col];
                        }
                        everything.push_back(val);
                    }
                    localRows++;
                }
                break;
            case 1: // interpolation
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "TableObserverPlane has no implementation for interpolation yet!");
                return false;
                break;
            }
        }
        totalRows += localRows;
    }
    ASSERT(everything.size() == cols * totalRows);
    out.Set(cols, totalRows, ft->GetColumnsInfos(), this->everything.data());

    this->myHash++;
    this->resetAllDirty();
    this->inputHash = ft->DataHash();
    return true;
}

/*
 * megamol::pcl::PclDataSource::getMultiParticleData
 */
bool TableObserverPlane::getObservedData(core::Call& call) {
    try {
        table::TableDataCall& out = dynamic_cast<table::TableDataCall&>(call);
        table::TableDataCall* ft = this->slotCallInputTable.CallAs<table::TableDataCall>();
        if (ft == NULL)
            return false;
        megamol::core::view::CallClipPlane* cp = this->slotCallClipPlane.CallAs<megamol::core::view::CallClipPlane>();
        if (cp == NULL)
            return false;

        if (!assertData(ft, cp, out))
            return false;

        //c.SetFrameID(0);
        out.SetDataHash(this->myHash);

        out.SetUnlocker(NULL);

        return true;
    } catch (vislib::Exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(e.GetMsg());
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(_T("Unexpected exception ")
                                                                _T("in callback getObservedData."));
        return false;
    }
}


/*
 * megamol::pcl::PclDataSource::getMultiparticleExtent
 */
bool TableObserverPlane::getHash(core::Call& call) {
    try {
        table::TableDataCall& out = dynamic_cast<table::TableDataCall&>(call);
        table::TableDataCall* ft = this->slotCallInputTable.CallAs<table::TableDataCall>();
        if (ft == NULL)
            return false;
        megamol::core::view::CallClipPlane* cp = this->slotCallClipPlane.CallAs<megamol::core::view::CallClipPlane>();
        if (cp == NULL)
            return false;

        if (!assertData(ft, cp, out))
            return false;

        out.SetDataHash(this->myHash);

        out.SetUnlocker(NULL);
        return true;
    } catch (vislib::Exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(e.GetMsg());
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(_T("Unexpected exception ")
                                                                _T("in callback getMultiparticleExtent."));
        return false;
    }
}


/*
 * megamol::pcl::PclDataSource::release
 */
void TableObserverPlane::release(void) {}
