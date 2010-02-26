/*
 * BezierDataSource.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "BezierDataSource.h"
#include "BezierDataCall.h"
#include "param/FilePathParam.h"
#include "vislib/BezierCurve.h"
#include "vislib/Point.h"
/*
#include <climits>
#include "MultiParticleDataCall.h"
#include "param/BoolParam.h"
#include "param/EnumParam.h"
#include "param/FloatParam.h"
#include "utility/ColourParser.h"
#include "vislib/forceinline.h"
#include "vislib/Log.h"
#include "vislib/StringTokeniser.h"
#include "vislib/sysfunctions.h"
#include "vislib/SystemMessage.h"
#include "vislib/mathfunctions.h"
#include "vislib/MemmappedFile.h"
#include "vislib/Vector.h"
*/

using namespace megamol::core;


/*
 * misc::BezierDataSource::BezierDataSource
 */
misc::BezierDataSource::BezierDataSource(void) : Module(),
        filenameSlot("filename", "The path of the IMD file to read"),
        getDataSlot("getdata", "The slot exposing the loaded data"),
        minX(0.0f), minY(0.0f), minZ(0.0f), maxX(1.0f), maxY(1.0f),
        maxZ(1.0f), curves(), datahash(0) {

    this->filenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);

    this->getDataSlot.SetCallback("BezierDataCall", "GetData",
        &BezierDataSource::getDataCallback);
    this->getDataSlot.SetCallback("BezierDataCall", "GetExtent",
        &BezierDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);
}


/*
 * misc::BezierDataSource::~BezierDataSource
 */
misc::BezierDataSource::~BezierDataSource(void) {
    this->Release();
}


/*
 * misc::BezierDataSource::create
 */
bool misc::BezierDataSource::create(void) {
    // intentionally empty
    return true;
}


/*
 * misc::BezierDataSource::release
 */
void misc::BezierDataSource::release(void) {
    this->curves.Clear();
    this->minX = this->minY = this->minZ = 0.0f;
    this->maxX = this->maxY = this->maxZ = 1.0f;
    this->datahash = 0;
}


/*
 * misc::BezierDataSource::getDataCallback
 */
bool misc::BezierDataSource::getDataCallback(Call& caller) {
    BezierDataCall *bdc = dynamic_cast<BezierDataCall*>(&caller);
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
 * misc::BezierDataSource::getExtentCallback
 */
bool misc::BezierDataSource::getExtentCallback(Call& caller) {
    BezierDataCall *bdc = dynamic_cast<BezierDataCall*>(&caller);
    if (bdc == NULL) return false;

    this->assertData();

    bdc->SetDataHash(this->datahash);
    bdc->SetExtent(1 /* static data */,
        this->minX, this->minY, this->minZ,
        this->maxX, this->maxY, this->maxZ);

    return true;
}


/*
 * misc::BezierDataSource::assertData
 */
void misc::BezierDataSource::assertData(void) {
    if (!this->filenameSlot.IsDirty()) return;
    this->filenameSlot.ResetDirty();

    this->curves.Clear();
    this->minX = this->minY = this->minZ = 0.0f;
    this->maxX = this->maxY = this->maxZ = 1.0f;
    this->datahash = 0;


    // TODO: What file format ... wtf
    // Dummy curve
    this->curves.Add(vislib::math::BezierCurve<misc::BezierDataCall::BezierPoint, 3>());
    this->curves[0].ControlPoint(0).Set(0.0f, 1.0f, 0.0f, 0.2f, 255, 0, 0);
    this->curves[0].ControlPoint(1).Set(1.0f, -2.0f, 1.0f, 0.3f, 255, 0, 0);
    this->curves[0].ControlPoint(2).Set(2.0f, 2.0f, -1.0f, 0.0125f, 255, 255, 0);
    this->curves[0].ControlPoint(3).Set(3.0f, 0.0f, 2.0f, 0.2f, 0, 255, 0);
    this->curves.Add(vislib::math::BezierCurve<misc::BezierDataCall::BezierPoint, 3>());
    this->curves[1].ControlPoint(0).Set(0.0f, 1.0f, 0.0f, 0.2f, 255, 0, 0);
    this->curves[1].ControlPoint(1).Set(-1.0f, 4.0f, -1.0f, 0.1f, 255, 0, 0);
    this->curves[1].ControlPoint(2).Set(-2.0f, 2.0f, -1.0f, 0.4f, 255, 0, 255);
    this->curves[1].ControlPoint(3).Set(-3.0f, 0.0f, 3.0f, 0.4f, 0, 0, 255);

    this->datahash = 1; // boo


    // calc bounding box (and datahash)
    if (this->curves.Count() > 0) {

        this->minX = this->maxX = this->curves[0].ControlPoint(0).X();
        this->minY = this->maxY = this->curves[0].ControlPoint(0).Y();
        this->minZ = this->maxZ = this->curves[0].ControlPoint(0).Z();

        for (SIZE_T idx = this->curves.Count(); idx > 0;) {
            idx--;
            for (SIZE_T cpi = 0; cpi < 4; cpi++) {
                const misc::BezierDataCall::BezierPoint& pt
                    = this->curves[idx].ControlPoint(
                        static_cast<unsigned int>(cpi));

                //this->datahash ^= *reinterpret_cast<const SIZE_T&>(pt[0]);
                //this->datahash ^= *reinterpret_cast<const SIZE_T&>(pt[1]);
                //this->datahash ^= *reinterpret_cast<const SIZE_T&>(pt[2]);
                //this->datahash ^= *reinterpret_cast<const SIZE_T&>(pt[3]);

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
    }

}
