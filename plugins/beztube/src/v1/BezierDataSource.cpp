/*
 * BezierDataSource.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define VLDEPRECATED
#define VISLIB_DEPRECATED_H_INCLUDED
#include "v1/BezierDataSource.h"
#include "mmcore/Call.h"
#include "v1/BezierDataCall.h"
#include "mmcore/param/FilePathParam.h"
#include "vislib/sys/ASCIIFileBuffer.h"
#include "vislib/math/BezierCurve.h"
#include "vislib/Exception.h"
#include "vislib/sys/Log.h"
#include "vislib/math/Point.h"
#include "vislib/VersionNumber.h"

using namespace megamol;
using namespace megamol::beztube;


/*
 * v1::BezierDataSource::BezierDataSource
 */
v1::BezierDataSource::BezierDataSource(void) : Module(),
        filenameSlot("filename", "The path of the IMD file to read"),
        getDataSlot("getdata", "The slot exposing the loaded data"),
        minX(0.0f), minY(0.0f), minZ(0.0f), maxX(1.0f), maxY(1.0f),
        maxZ(1.0f), curves(), datahash(0) {

    this->filenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);

    this->getDataSlot.SetCallback("v1.BezierDataCall", "GetData",
        &BezierDataSource::getDataCallback);
    this->getDataSlot.SetCallback("v1.BezierDataCall", "GetExtent",
        &BezierDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);
}


/*
 * v1::BezierDataSource::~BezierDataSource
 */
v1::BezierDataSource::~BezierDataSource(void) {
    this->Release();
}


/*
 * v1::BezierDataSource::create
 */
bool v1::BezierDataSource::create(void) {
    // intentionally empty
    return true;
}


/*
 * v1::BezierDataSource::release
 */
void v1::BezierDataSource::release(void) {
    this->curves.Clear();
    this->minX = this->minY = this->minZ = 0.0f;
    this->maxX = this->maxY = this->maxZ = 1.0f;
    this->datahash = 0;
}


/*
 * v1::BezierDataSource::getDataCallback
 */
bool v1::BezierDataSource::getDataCallback(core::Call& caller) {
    v1::BezierDataCall *bdc = dynamic_cast<v1::BezierDataCall*>(&caller);
    if (bdc == NULL) return false;

    this->assertData();

    bdc->SetData(static_cast<unsigned int>(this->curves.Count()),
        this->curves.PeekElements());
    bdc->SetDataHash(this->datahash);
    bdc->SetExtent(1 /* static data */,
        this->minX, this->minY, this->minZ,
        this->maxX, this->maxY, this->maxZ);
    bdc->SetFrameID(0);
    bdc->SetUnlocker(NULL);

    return true;
}


/*
 * v1::BezierDataSource::getExtentCallback
 */
bool v1::BezierDataSource::getExtentCallback(core::Call& caller) {
    v1::BezierDataCall *bdc = dynamic_cast<v1::BezierDataCall*>(&caller);
    if (bdc == NULL) return false;

    this->assertData();

    bdc->SetDataHash(this->datahash);
    bdc->SetExtent(1 /* static data */,
        this->minX, this->minY, this->minZ,
        this->maxX, this->maxY, this->maxZ);

    return true;
}


/*
 * v1::BezierDataSource::assertData
 */
void v1::BezierDataSource::assertData(void) {
    if (!this->filenameSlot.IsDirty()) return;
    this->filenameSlot.ResetDirty();

    this->curves.Clear();
    this->minX = this->minY = this->minZ = 0.0f;
    this->maxX = this->maxY = this->maxZ = 1.0f;
    this->datahash++;

    this->loadBezDat(this->filenameSlot.Param<core::param::FilePathParam>()->Value());

    // calc bounding box (and datahash)
    if (this->curves.Count() > 0) {

        this->minX = this->maxX = this->curves[0].ControlPoint(0).X();
        this->minY = this->maxY = this->curves[0].ControlPoint(0).Y();
        this->minZ = this->maxZ = this->curves[0].ControlPoint(0).Z();

        for (SIZE_T idx = this->curves.Count(); idx > 0;) {
            idx--;
            for (SIZE_T cpi = 0; cpi < 4; cpi++) {
                const v1::BezierDataCall::BezierPoint& pt
                    = this->curves[idx].ControlPoint(
                        static_cast<unsigned int>(cpi));

                if (this->minX > pt.X() - pt.Radius()) {
                    this->minX = pt.X() - pt.Radius();
                }
                if (this->maxX < pt.X() + pt.Radius()) {
                    this->maxX = pt.X() + pt.Radius();
                }
                if (this->minY > pt.Y() - pt.Radius()) {
                    this->minY = pt.Y() - pt.Radius();
                }
                if (this->maxY < pt.Y() + pt.Radius()) {
                    this->maxY = pt.Y() + pt.Radius();
                }
                if (this->minZ > pt.Z() - pt.Radius()) {
                    this->minZ = pt.Z() - pt.Radius();
                }
                if (this->maxZ < pt.Z() + pt.Radius()) {
                    this->maxZ = pt.Z() + pt.Radius();
                }
            }
        }

    } else {
        this->minX = this->minY = this->minZ = -1.0f;
        this->maxX = this->maxY = this->maxZ = 1.0f;

    }

}


/*
 * v1::BezierDataSource::loadBezDat
 */
void v1::BezierDataSource::loadBezDat(const vislib::TString& filename) {
    using vislib::sys::Log;

    vislib::sys::ASCIIFileBuffer bezDat(vislib::sys::ASCIIFileBuffer::PARSING_WORDS);
    bezDat.LoadFile(filename);

    if (bezDat.Count() < 1) {
        Log::DefaultLog.WriteError("BezDat file seems to be empty");
        return;
    }

    if (bezDat[0].Count() < 2) {
        Log::DefaultLog.WriteError("Header line too short");
        return;
    }

    if (!vislib::StringA("BezDatA").Equals(bezDat[0].Word(0))) {
        Log::DefaultLog.WriteError("BezDat header ID wrong");
        return;
    }

    vislib::VersionNumber ver(0, 0, 0);
    try {
        ver.Parse(bezDat[0].Word(1));
    } catch(...) {
    }
    if (ver != vislib::VersionNumber(1, 0)) {
        Log::DefaultLog.WriteError("BezDat version number wrong");
        return;
    }

    SIZE_T pcnt = 0;
    SIZE_T ccnt = 0;
    for (SIZE_T i = 1; i < bezDat.Count(); i++) {
        if (bezDat[i].Count() == 0) continue;

        if (vislib::StringA("PT").Equals(bezDat[i].Word(0)) && (bezDat[i].Count() >= 8)) pcnt++;
        else if (vislib::StringA("BC").Equals(bezDat[i].Word(0)) && (bezDat[i].Count() >= 5)) ccnt++;
    }

    v1::BezierDataCall::BezierPoint *points = new v1::BezierDataCall::BezierPoint[pcnt];
    pcnt = 0;
    for (SIZE_T i = 1; i < bezDat.Count(); i++) {
        if (bezDat[i].Count() == 0) continue;
        if (!vislib::StringA("PT").Equals(bezDat[i].Word(0)) || (bezDat[i].Count() < 8)) continue;
        try {
            points[pcnt].Set(
                static_cast<float>(vislib::CharTraitsA::ParseDouble(bezDat[i].Word(1))),
                static_cast<float>(vislib::CharTraitsA::ParseDouble(bezDat[i].Word(2))),
                static_cast<float>(vislib::CharTraitsA::ParseDouble(bezDat[i].Word(3))),
                static_cast<float>(vislib::CharTraitsA::ParseDouble(bezDat[i].Word(4))),
                static_cast<unsigned char>(vislib::math::Clamp(vislib::CharTraitsA::ParseInt(bezDat[i].Word(5)), 0, 255)),
                static_cast<unsigned char>(vislib::math::Clamp(vislib::CharTraitsA::ParseInt(bezDat[i].Word(6)), 0, 255)),
                static_cast<unsigned char>(vislib::math::Clamp(vislib::CharTraitsA::ParseInt(bezDat[i].Word(7)), 0, 255)));
            pcnt++;
        } catch(vislib::Exception ex) {
            Log::DefaultLog.WriteError("Parse Error in Line %d: %s", static_cast<int>(i), ex.GetMsgA());
            delete[] points;
            return;
        } catch(...) {
            Log::DefaultLog.WriteError("Parse Error in Line %d", static_cast<int>(i));
            delete[] points;
            return;
        }
    }
    this->curves.AssertCapacity(ccnt);
    for (SIZE_T i = 1; i < bezDat.Count(); i++) {
        if (bezDat[i].Count() == 0) continue;
        if (!vislib::StringA("BC").Equals(bezDat[i].Word(0)) || (bezDat[i].Count() < 5)) continue;
        try {
            int i1 = vislib::CharTraitsA::ParseInt(bezDat[i].Word(1));
            int i2 = vislib::CharTraitsA::ParseInt(bezDat[i].Word(2));
            int i3 = vislib::CharTraitsA::ParseInt(bezDat[i].Word(3));
            int i4 = vislib::CharTraitsA::ParseInt(bezDat[i].Word(4));
            if ((i1 >= static_cast<int>(pcnt))
                    || (i2 >= static_cast<int>(pcnt))
                    || (i3 >= static_cast<int>(pcnt))
                    || (i4 >= static_cast<int>(pcnt))) {
                throw vislib::Exception("Point index out of range", __FILE__, __LINE__);
            }
            this->curves.Add(vislib::math::BezierCurve<v1::BezierDataCall::BezierPoint, 3>());
            this->curves.Last().ControlPoint(0) = points[i1];
            this->curves.Last().ControlPoint(1) = points[i2];
            this->curves.Last().ControlPoint(2) = points[i3];
            this->curves.Last().ControlPoint(3) = points[i4];
        } catch(vislib::Exception ex) {
            Log::DefaultLog.WriteError("Parse Error in Line %d: %s", static_cast<int>(i), ex.GetMsgA());
            this->curves.Clear();
            delete[] points;
            return;
        } catch(...) {
            Log::DefaultLog.WriteError("Parse Error in Line %d", static_cast<int>(i));
            this->curves.Clear();
            delete[] points;
            return;
        }
    }

    delete[] points;
}
