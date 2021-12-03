/*
 * ParticleFortLoader.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "QuartzParticleDataCall.h"
#include "QuartzParticleFortLoader.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/utility/sys/MemmappedFile.h"
#include "vislib/String.h"
#include "vislib/math/Point.h"
#include "vislib/math/ShallowVector.h"
#include "vislib/memutils.h"
#include "vislib/utils.h"
#include <cmath>


namespace megamol {
namespace demos_gl {


/*
 * ParticleFortLoader::ParticleFortLoader
 */
ParticleFortLoader::ParticleFortLoader(void)
        : core::Module()
        , dataOutSlot("dataout", "The slot providing the loaded data")
        , positionFileNameSlot("positionFile", "The path to the position file")
        , attributeFileNameSlot("attributeFile", "The path to the attribute file (radius + orientation)")
        , datahash(0)
        , typeCnt(0)
        , partTypes(NULL)
        , partCnts(NULL)
        , partDatas(NULL)
        , bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f)
        , cbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f)
        , autoBBoxSlot("bbox::auto", "Flag whether or not to use the calculated bouning box")
        , bboxMinSlot("bbox::min", "The minimum values for the manual bounding box")
        , bboxMaxSlot("bbox::max", "The maximum values for the manual bounding box") {

    this->positionFileNameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->positionFileNameSlot);

    this->attributeFileNameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->attributeFileNameSlot);

    this->dataOutSlot.SetCallback(ParticleDataCall::ClassName(),
        ParticleDataCall::FunctionName(ParticleDataCall::CallForGetData), &ParticleFortLoader::getData);
    this->dataOutSlot.SetCallback(ParticleDataCall::ClassName(),
        ParticleDataCall::FunctionName(ParticleDataCall::CallForGetExtent), &ParticleFortLoader::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);

    this->autoBBoxSlot << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->autoBBoxSlot);

    this->bboxMinSlot << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(-1.0f, -1.0f, -1.0f));
    this->MakeSlotAvailable(&this->bboxMinSlot);

    this->bboxMaxSlot << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1.0f, 1.0f, 1.0f));
    this->MakeSlotAvailable(&this->bboxMaxSlot);
}


/*
 * ParticleFortLoader::~ParticleFortLoader
 */
ParticleFortLoader::~ParticleFortLoader(void) {
    this->Release();
}


/*
 * ParticleFortLoader::create
 */
bool ParticleFortLoader::create(void) {
    // intentionally empty
    return true;
}


/*
 * ParticleFortLoader::getData
 */
bool ParticleFortLoader::getData(core::Call& c) {
    ParticleDataCall* cdc = dynamic_cast<ParticleDataCall*>(&c);
    if (cdc == NULL)
        return false;
    this->assertData();
    cdc->SetDataHash(this->datahash);
    cdc->SetParticleData(this->typeCnt, this->partTypes, this->partCnts, this->partDatas);
    cdc->SetUnlocker(NULL);
    return true;
}


/*
 * ParticleFortLoader::getExtent
 */
bool ParticleFortLoader::getExtent(core::Call& c) {
    ParticleDataCall* cdc = dynamic_cast<ParticleDataCall*>(&c);
    if (cdc == NULL)
        return false;
    this->assertData();
    cdc->SetFrameCount(1);
    cdc->AccessBoundingBoxes().Clear();

    if ((this->autoBBoxSlot.IsDirty() || this->bboxMaxSlot.IsDirty() || this->bboxMinSlot.IsDirty()) &&
        (this->datahash != 0)) {
        this->datahash++;
        this->autoBBoxSlot.ResetDirty();
        this->bboxMaxSlot.ResetDirty();
        this->bboxMinSlot.ResetDirty();
    }

    if (this->autoBBoxSlot.Param<core::param::BoolParam>()->Value()) {
        cdc->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
    } else {
        const vislib::math::Vector<float, 3>& minval = this->bboxMinSlot.Param<core::param::Vector3fParam>()->Value();
        const vislib::math::Vector<float, 3>& maxval = this->bboxMaxSlot.Param<core::param::Vector3fParam>()->Value();
        vislib::math::Cuboid<float> mbbox(vislib::math::Min(minval.X(), maxval.X()),
            vislib::math::Min(minval.Y(), maxval.Y()), vislib::math::Min(minval.Z(), maxval.Z()),
            vislib::math::Max(minval.X(), maxval.X()), vislib::math::Max(minval.Y(), maxval.Y()),
            vislib::math::Max(minval.Z(), maxval.Z()));
        cdc->AccessBoundingBoxes().SetObjectSpaceBBox(mbbox);
    }
    cdc->AccessBoundingBoxes().SetObjectSpaceClipBox(this->cbox);
    cdc->SetDataHash(this->datahash);
    cdc->SetUnlocker(NULL);
    return true;
}


/*
 * ParticleFortLoader::release
 */
void ParticleFortLoader::release(void) {
    if (this->partDatas != NULL) {
        for (unsigned int i = 0; i < this->typeCnt; i++) {
            delete[] this->partDatas[i];
        }
        ARY_SAFE_DELETE(this->partDatas);
    }
    this->typeCnt = 0;
    ARY_SAFE_DELETE(this->partTypes);
    ARY_SAFE_DELETE(this->partCnts);
    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
}


/*
 * ParticleFortLoader::assertData
 */
void ParticleFortLoader::assertData(void) {
    using megamol::core::utility::log::Log;

    if (this->positionFileNameSlot.IsDirty() || this->attributeFileNameSlot.IsDirty()) {
        this->positionFileNameSlot.ResetDirty();
        this->attributeFileNameSlot.ResetDirty();

        const float tetGrowFac = 1.3f; // tetrahedron in-sphere to out-sphere radius
        const vislib::TString& posFileName =
            this->positionFileNameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str();
        const vislib::TString& attrFileName =
            this->attributeFileNameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str();

        this->release(); // because I know it only clears the memory arrays
        this->datahash++;

        try {
            const SIZE_T defaultIncrement = 1000;
            vislib::Array<unsigned int> types(0, 0, 1 * defaultIncrement);
            vislib::Array<float> positions(0, 0.0f, 3 * defaultIncrement);
            vislib::Array<float> attributes(0, 0.0f, 5 * defaultIncrement);
            vislib::sys::MemmappedFile file;
            unsigned int bound = 0;

            if (posFileName.IsEmpty()) {
                Log::DefaultLog.WriteError("Position data file missing");
                types.Clear();
                positions.Clear();

            } else if (file.Open(posFileName, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ,
                           vislib::sys::File::OPEN_ONLY)) {
                unsigned int t;
                float x, y, z;
                bool typeRangeWarning = false;

                while (!file.IsEOF()) {
                    if (file.Read(&bound, 4) != 4)
                        break; // unexpected end of file
                    if (bound != 0x10)
                        break; // wrong block size
                    if (file.Read(&t, 4) != 4)
                        break; // unexpected end of file
                    if (file.Read(&x, 4) != 4)
                        break; // unexpected end of file
                    if (file.Read(&y, 4) != 4)
                        break; // unexpected end of file
                    if (file.Read(&z, 4) != 4)
                        break; // unexpected end of file
                    if (file.Read(&bound, 4) != 4)
                        break; // unexpected end of file
                    if (bound != 0x10)
                        break; // wrong block size

                    // type fixing
                    t--; // required for 'new_orient' data
                    if ((t < 4000) || (t > 4999)) {
                        typeRangeWarning = true;
                    }
                    t %= 1000;

                    types.Add(t);     // type
                    positions.Add(x); // x
                    positions.Add(y); // y
                    positions.Add(z); // z
                }

                if (typeRangeWarning) {
                    Log::DefaultLog.WriteWarn(
                        "At least one particle type was not in quartz type range [4000..4999] and was altered!");
                }

                file.Close();
            } else {
                Log::DefaultLog.WriteError(
                    "Unable to open position file \"%s\"", vislib::StringA(posFileName).PeekBuffer());
            }

            if (attrFileName.IsEmpty()) {
                Log::DefaultLog.WriteWarn("Attribute data file missing");
                attributes.Clear();

            } else if (file.Open(attrFileName, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ,
                           vislib::sys::File::OPEN_ONLY)) {
                float r, q1, q2, q3, q4, d;
                bool first = true;
                float minRad, maxRad;

                while (!file.IsEOF()) {
                    if (file.Read(&bound, 4) != 4)
                        break; // unexpected end of file
                    if (bound != 0x20)
                        break; // wrong block size
                    if (file.Read(&r, 4) != 4)
                        break; // unexpected end of file
                    if (file.Read(&d, 4) != 4)
                        break; // unexpected end of file
                    if (file.Read(&d, 4) != 4)
                        break; // unexpected end of file
                    if (file.Read(&d, 4) != 4)
                        break; // unexpected end of file
                    if (file.Read(&q1, 4) != 4)
                        break; // unexpected end of file
                    if (file.Read(&q2, 4) != 4)
                        break; // unexpected end of file
                    if (file.Read(&q3, 4) != 4)
                        break; // unexpected end of file
                    if (file.Read(&q4, 4) != 4)
                        break; // unexpected end of file
                    if (file.Read(&bound, 4) != 4)
                        break; // unexpected end of file
                    if (bound != 0x20)
                        break;         // wrong block size
                    attributes.Add(r); // radius
                    if (first) {
                        minRad = maxRad = r;
                        first = false;
                    } else {
                        if (minRad > r)
                            minRad = r;
                        if (maxRad < r)
                            maxRad = r;
                    }
                    q4 = -q4; // required for 'new_orient/1cryt_+90xrot' data

                    vislib::math::Vector<float, 4> q(q1, q2, q3, q4);
                    float ol = q.Normalise();
                    if (!vislib::math::IsEqual(ol, 1.0f)) {
                        Log::DefaultLog.WriteWarn("Quaternion with length %f is not good\n", ol);
                    }
                    q1 = q[0];
                    q2 = q[1];
                    q3 = q[2];
                    q4 = q[3];

                    attributes.Add(q1); // qx (unclear if this is ok)
                    attributes.Add(q2); // qy
                    attributes.Add(q3); // qz
                    attributes.Add(q4); // qw (real term)
                }

                Log::DefaultLog.WriteInfo(100, "Particle radii: %f ... %f\n", minRad, maxRad);

                file.Close();
            } else {
                Log::DefaultLog.WriteError(
                    "Unable to open attribute file \"%s\"", vislib::StringA(attrFileName).PeekBuffer());
            }

            ASSERT(types.Count() * 3 == positions.Count());
            if ((types.Count() * 5 != attributes.Count())) {
                Log::DefaultLog.WriteError("Particle attribute list is of illegal length! Will be ignored");
                attributes.Clear();
            }

            vislib::Array<int> typeList;
            for (SIZE_T i = 0; i < types.Count(); i++) {
                if (!typeList.Contains(types[i]))
                    typeList.Add(types[i]);
            }
            typeList.Sort(&vislib::DiffComparator<int>);

            Log::DefaultLog.WriteMsg(350, "Loaded files %u particles and %u types\n", types.Count(), typeList.Count());

            this->partTypes = new unsigned int[typeList.Count()];
            this->partCnts = new unsigned int[typeList.Count()];
            for (SIZE_T i = 0; i < typeList.Count(); i++) {
                this->partTypes[i] = typeList[i];
                this->partCnts[i] = 0;
            }
            for (SIZE_T i = 0; i < types.Count(); i++) {
                INT_PTR idx = typeList.IndexOf(types[i]);
                ASSERT(idx != vislib::Array<int>::INVALID_POS);
                this->partCnts[idx]++; // :-)
            }
            this->partDatas = new float*[typeList.Count()];
            for (SIZE_T i = 0; i < typeList.Count(); i++) {
                this->partDatas[i] = new float[8 * this->partCnts[i]];
                this->partCnts[i] = 0;
            }
            for (SIZE_T i = 0; i < types.Count(); i++) {
                INT_PTR idx = typeList.IndexOf(types[i]);
                ASSERT(idx != vislib::Array<int>::INVALID_POS);
                float* v = this->partDatas[idx] + (8 * this->partCnts[idx]);
                this->partCnts[idx]++;
                v[0] = positions[i * 3 + 0];
                v[1] = positions[i * 3 + 1];
                v[2] = positions[i * 3 + 2];
                if (attributes.Count() > (i * 5 + 4)) {
                    v[3] = attributes[i * 5 + 0];
                    v[4] = attributes[i * 5 + 1];
                    v[5] = attributes[i * 5 + 2];
                    v[6] = attributes[i * 5 + 3];
                    v[7] = attributes[i * 5 + 4];

                } else {
                    v[3] = 0.1f;
                    v[4] = 0.0f;
                    v[5] = 0.0f;
                    v[6] = 0.0f;
                    v[7] = 1.0f;
                }
                vislib::math::Point<float, 3> p0(v[0], v[1], v[2]);
                vislib::math::Point<float, 3> p1(
                    v[0] - v[3] * tetGrowFac, v[1] - v[3] * tetGrowFac, v[2] - v[3] * tetGrowFac);
                vislib::math::Point<float, 3> p2(
                    v[0] + v[3] * tetGrowFac, v[1] + v[3] * tetGrowFac, v[2] + v[3] * tetGrowFac);
                if (i == 0) {
                    this->bbox.Set(p0.X(), p0.Y(), p0.Z(), p0.X(), p0.Y(), p0.Z());
                    this->cbox.Set(p1.X(), p1.Y(), p1.Z(), p2.X(), p2.Y(), p2.Z());
                } else {
                    this->bbox.GrowToPoint(p0);
                    this->cbox.GrowToPoint(p1);
                    this->cbox.GrowToPoint(p2);
                }
            }

            this->typeCnt = static_cast<unsigned int>(typeList.Count());

            if (this->bbox.IsEmpty()) {
                this->bbox = this->cbox;
            }

            Log::DefaultLog.WriteInfo("Calculated data set bounding box: (%f, %f, %f) - (%f, %f, %f)\n",
                this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back(), this->bbox.Right(), this->bbox.Top(),
                this->bbox.Front());

        } catch (vislib::Exception ex) {
            Log::DefaultLog.WriteError("Unexpected exception: %s\n", ex.GetMsgA());
        } catch (...) { Log::DefaultLog.WriteError("Unknown exception"); }
    }
}

} // namespace demos_gl
} /* end namespace megamol */
