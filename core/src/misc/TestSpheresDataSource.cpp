/*
 * TestSpheresDataSource.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "mmcore/misc/TestSpheresDataSource.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "vislib/math/Vector.h"
#include "vislib/math/ShallowVector.h"
#include "vislib/math/Quaternion.h"
#include "cmath"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/IntParam.h"

using namespace megamol::core;


/*
 * misc::TestSpheresDataSource::TestSpheresDataSource
 */
misc::TestSpheresDataSource::TestSpheresDataSource(void) : view::AnimDataModule(), getDataSlot("getData", "Gets the data from the data source")
	, numSpheresSlot("numSpheres", "number of spheres to generate")
	, numFramesSlot("numFrames", "number of frames to generate")
#ifdef MMCORE_TEST_DYN_PARAM_SLOTS
    , p1("p1", "Test slot for dynamic parameter slots")
    , p2("p2", "Test slot for dynamic parameter slots")
#endif
        {

    this->getDataSlot.SetCallback(moldyn::MultiParticleDataCall::ClassName(), moldyn::MultiParticleDataCall::FunctionName(0), &TestSpheresDataSource::getDataCallback);
    this->getDataSlot.SetCallback(moldyn::MultiParticleDataCall::ClassName(), moldyn::MultiParticleDataCall::FunctionName(1), &TestSpheresDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

	numSpheresSlot << new param::IntParam(15);
	this->MakeSlotAvailable(&numSpheresSlot);

	numFramesSlot << new param::IntParam(100);
	this->MakeSlotAvailable(&numFramesSlot);

#ifdef MMCORE_TEST_DYN_PARAM_SLOTS
    p1.SetParameter(new param::BoolParam(false));
    MakeSlotAvailable(&p1);
    p1.ForceSetDirty();

    p2.SetParameter(new param::StringParam("Hugo"));
#endif

}


/*
 * misc::TestSpheresDataSource::~TestSpheresDataSource
 */
misc::TestSpheresDataSource::~TestSpheresDataSource(void) {
    this->Release();
}


/*
 * misc::TestSpheresDataSource::constructFrame
 */
view::AnimDataModule::Frame* misc::TestSpheresDataSource::constructFrame(void) const {
    return new Frame(const_cast<misc::TestSpheresDataSource&>(*this));
}


/*
 * misc::BezierDataSource::create
 */
bool misc::TestSpheresDataSource::create(void) {
	auto f = this->numFramesSlot.Param<param::IntParam>()->Value();
    view::AnimDataModule::setFrameCount(f);
    view::AnimDataModule::initFrameCache(f);
    return true;
}


/*
 * misc::TestSpheresDataSource::loadFrame
 */
void misc::TestSpheresDataSource::loadFrame(view::AnimDataModule::Frame *frame, unsigned int idx) {
    Frame *frm = dynamic_cast<Frame *>(frame);
    if (frm == NULL) return;
    frm->SetFrameNumber(idx);
	auto frameCount = this->numFramesSlot.Param<param::IntParam>()->Value();
	auto sphereCount = this->numSpheresSlot.Param<param::IntParam>()->Value();
    frm->data = new float[7 * sphereCount];
    for (unsigned int i = 0; i < sphereCount; i++) {
        vislib::math::ShallowVector<float, 3> pos(&frm->data[i * 7]);
        ::srand(i); // stablize values for particles
        float &r = frm->data[i * 7 + 3];
        float &cr = frm->data[i * 7 + 4];
        float &cg = frm->data[i * 7 + 5];
        float &cb = frm->data[i * 7 + 6];
        vislib::math::Vector<float, 3> X(static_cast<float>((::rand() % 2) * 2 - 1), 0.0f, 0.0f);
        vislib::math::Vector<float, 3> Y(0.0f, static_cast<float>((::rand() % 2) * 2 - 1), 0.0f);
        vislib::math::Vector<float, 3> Z(
            static_cast<float>(1000 - ::rand() % 2001) * 0.001f,
            static_cast<float>(1000 - ::rand() % 2001) * 0.001f,
            static_cast<float>(1000 - ::rand() % 2001) * 0.001f);
        switch (::rand() % 6) {
            case 0: Z.SetX(1.0f); break;
            case 1: Z.SetX(-1.0f); break;
            case 2: Z.SetY(1.0f); break;
            case 3: Z.SetY(-1.0f); break;
            case 4: Z.SetZ(1.0f); break;
            case 5: Z.SetZ(-1.0f); break;
        }
        Z.Normalise();
        vislib::math::Quaternion<float> rot(static_cast<float>((::rand() % 2) * 2 - 1) * static_cast<float>(M_PI) * static_cast<float>(::rand() % 2000) * 0.001f, Z);
        float dist = static_cast<float>(::rand() % 1001) * 0.001f;
        dist = ::pow(dist, 0.333f) * 0.9f;
        float a = (static_cast<float>(2 * idx) / static_cast<float>(frameCount)) * static_cast<float>(M_PI);
        X = rot * X;
        Y = rot * Y;

        X *= sin(a) * dist;
        Y *= cos(a) * dist;
        pos = X;
        pos += Y;

        r = 0.05f + static_cast<float>(::rand() % 501) * 0.0001f;

        Z.Set(
            static_cast<float>(1000 - ::rand() % 2001) * 0.001f,
            static_cast<float>(1000 - ::rand() % 2001) * 0.001f,
            static_cast<float>(1000 - ::rand() % 2001) * 0.001f);
        switch (::rand() % 6) {
            case 0: Z.SetX(1.0f); break;
            case 1: Z.SetX(-1.0f); break;
            case 2: Z.SetY(1.0f); break;
            case 3: Z.SetY(-1.0f); break;
            case 4: Z.SetZ(1.0f); break;
            case 5: Z.SetZ(-1.0f); break;
        }
        Z.Normalise();

        cr = vislib::math::Abs(Z.X());
        cg = vislib::math::Abs(Z.Y());
        cb = vislib::math::Abs(Z.Z());

    }

}


/*
 * misc::TestSpheresDataSource::release
 */
void misc::TestSpheresDataSource::release(void) {
    // intentionally empty
}


/*
 * misc::TestSpheresDataSource::getDataCallback
 */
bool misc::TestSpheresDataSource::getDataCallback(Call& caller) {
    moldyn::MultiParticleDataCall *mpdc = dynamic_cast<moldyn::MultiParticleDataCall *>(&caller);
    if (mpdc == NULL) return false;

	auto frameCount = this->numFramesSlot.Param<param::IntParam>()->Value();
	auto sphereCount = this->numSpheresSlot.Param<param::IntParam>()->Value();

	if (this->numFramesSlot.IsDirty() || this->numSpheresSlot.IsDirty()) {
		this->resetFrameCache();
		view::AnimDataModule::setFrameCount(frameCount);
		view::AnimDataModule::initFrameCache(frameCount);
		this->numFramesSlot.ResetDirty();
		this->numSpheresSlot.ResetDirty();
	}

	view::AnimDataModule::Frame *f = this->requestLockedFrame(mpdc->FrameID());
    if (f == NULL) return false;
    f->Unlock(); // because I know that this data source is simple enough that no locking is required
    Frame *frm = dynamic_cast<Frame*>(f);
    if (frm == NULL) return false;

#ifdef MMCORE_TEST_DYN_PARAM_SLOTS
    if (p1.IsDirty()) {
        p1.ResetDirty();
        if (p1.Param<param::BoolParam>()->Value()) {
            if (p2.GetStatus() == AbstractSlot::STATUS_UNAVAILABLE) {
                MakeSlotAvailable(&p2);
            }
        } else {
            if (p2.GetStatus() != AbstractSlot::STATUS_UNAVAILABLE) {
                SetSlotUnavailable(&p2);
            }
        }
    }
#endif

    mpdc->SetFrameID(f->FrameNumber());
    mpdc->SetDataHash(1);
    mpdc->SetExtent(frameCount,
        -1.0f, -1.0f, -1.0f,
        1.0f, 1.0f, 1.0f);
    mpdc->SetParticleListCount(1);
    mpdc->AccessParticles(0).SetCount(sphereCount);
    mpdc->AccessParticles(0).SetVertexData(moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZR, frm->data, sizeof(float) * 7);
    mpdc->AccessParticles(0).SetColourData(moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGB, frm->data + 4, sizeof(float) * 7);
    mpdc->SetUnlocker(NULL);

    return true;
}


/*
 * misc::TestSpheresDataSource::getExtentCallback
 */
bool misc::TestSpheresDataSource::getExtentCallback(Call& caller) {
    moldyn::MultiParticleDataCall *mpdc = dynamic_cast<moldyn::MultiParticleDataCall *>(&caller);
    if (mpdc == NULL) return false;

	auto frameCount = this->numFramesSlot.Param<param::IntParam>()->Value();

    mpdc->SetDataHash(1);
    mpdc->SetExtent(frameCount,
        -1.0f, -1.0f, -1.0f,
        1.0f, 1.0f, 1.0f);

    return true;
}
