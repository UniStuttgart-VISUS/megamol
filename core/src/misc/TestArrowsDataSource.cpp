/*
 * TestArrowsDataSource.cpp
 *
 * Copyright (C) 2019 by MegaMol Team. Alle Rechte vorbehalten.
 */
// Make crappy clang-format f*** off:
// clang-format off

#include "stdafx.h"

#include "mmcore/misc/TestArrowsDataSource.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"

#include "vislib/sys/Log.h"


/*
 * megamol::core::misc::TestArrowsDataSource::TestArrowsDataSource
 */
megamol::core::misc::TestArrowsDataSource::TestArrowsDataSource(void)
        : slotGetData("GetData", "Gets the data from the data source.") {
    using namespace megamol::core::moldyn;

    this->slotGetData.SetCallback(
        MultiParticleDataCall::ClassName(),
        MultiParticleDataCall::FunctionName(0),
        &TestArrowsDataSource::onGetData);
    this->slotGetData.SetCallback(
        MultiParticleDataCall::ClassName(), 
        MultiParticleDataCall::FunctionName(1),
        &TestArrowsDataSource::onGetExtents);
    this->MakeSlotAvailable(&this->slotGetData);
}


/*
 * megamol::core::misc::TestArrowsDataSource::~TestArrowsDataSource
 */
megamol::core::misc::TestArrowsDataSource::~TestArrowsDataSource(void) {
    this->Release();
}


#if 0
/*
 * misc::TestSpheresDataSource::loadFrame
 */
void misc::TestSpheresDataSource::loadFrame(view::AnimDataModule::Frame *frame, unsigned int idx) {
    Frame *frm = dynamic_cast<Frame *>(frame);
    if (frm == NULL) return;
    frm->SetFrameNumber(idx);
    frm->data = new float[7 * TestSpheresDataSource::sphereCount];
    for (unsigned int i = 0; i < TestSpheresDataSource::sphereCount; i++) {
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
        float a = (static_cast<float>(2 * idx) / static_cast<float>(TestSpheresDataSource::frameCount)) * static_cast<float>(M_PI);
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
#endif


/*
 * megamol::core::misc::TestArrowsDataSource::create
 */
bool megamol::core::misc::TestArrowsDataSource::create(void) {
    this->data.resize(3);

    {
        auto& p = this->data[0];
        p.x = p.vx = p.r = 1.0f;
        p.y = p.vy = p.g = 0.0f;
        p.z = p.vz = p.b = 0.0f;
        p.l = 1.0f;
    }

    {
        auto& p = this->data[1];
        p.x = p.vx = p.r = 0.0f;
        p.y = p.vy = p.g = 1.0f;
        p.z = p.vz = p.b = 0.0f;
        p.l = 1.0f;
    }

    {
        auto& p = this->data[2];
        p.x = p.vx = p.r = 0.0f;
        p.y = p.vy = p.g = 0.0f;
        p.z = p.vz = p.b = 1.0f;
        p.l = 1.0f;
    }

    this->extents[0] = -5.0f;
    this->extents[1] = -5.0f;
    this->extents[2] = -5.0f;

    this->extents[3] = 5.0f;
    this->extents[4] = 5.0f;
    this->extents[5] = 5.0f;

    return true;
}


/*
 * megamol::core::misc::TestArrowsDataSource::onGetData
 */
bool megamol::core::misc::TestArrowsDataSource::onGetData(Call& caller) {
    using namespace megamol::core::moldyn;
    using vislib::sys::Log;

    auto c = dynamic_cast<MultiParticleDataCall *>(&caller);
    if (c == nullptr) {
        Log::DefaultLog.WriteError(_T("Call passed to ")
            _T("TestArrowsDataSource::onGetData is not a valid ")
            _T("MultiParticleDataCall."), nullptr);
        return false;
    }

    c->SetFrameID(0);
    c->SetDataHash(1);
    c->SetExtent(1, this->extents[0], this->extents[1], this->extents[2],
        this->extents[3], this->extents[4], this->extents[5]);

    c->SetParticleListCount(1);
    c->AccessParticles(0).SetCount(this->data.size());
    c->AccessParticles(0).SetVertexData(
        SimpleSphericalParticles::VERTDATA_FLOAT_XYZR,
        this->data.data(),
        sizeof(Particle));
    c->AccessParticles(0).SetDirData(
        SimpleSphericalParticles::DIRDATA_FLOAT_XYZ,
        std::addressof(this->data.data()->vx),
        sizeof(Particle));
    c->AccessParticles(0).SetColourData(
        SimpleSphericalParticles::COLDATA_FLOAT_RGB,
        std::addressof(this->data.data()->r),
        sizeof(Particle));
    c->SetUnlocker(NULL);

    return true;
}


/*
 * megamol::core::misc::TestArrowsDataSource::onGetExtents
 */
bool megamol::core::misc::TestArrowsDataSource::onGetExtents(Call& caller) {
    using namespace megamol::core::moldyn;
    using vislib::sys::Log;

    auto c = dynamic_cast<MultiParticleDataCall *>(&caller);
    if (c == nullptr) {
        Log::DefaultLog.WriteError(_T("Call passed to ")
            _T("TestArrowsDataSource::onGetExtents is not a valid ")
            _T("MultiParticleDataCall."), nullptr);
        return false;
    }

    c->SetDataHash(1);
    c->SetExtent(1, this->extents[0], this->extents[1], this->extents[2],
        this->extents[3], this->extents[4], this->extents[5]);

    return true;
}


/*
 * megamol::core::misc::TestArrowsDataSource::release
 */
void megamol::core::misc::TestArrowsDataSource::release(void) { }
