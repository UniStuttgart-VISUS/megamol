/*
 * DirPartVolume.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define USE_MATH_DEFINES
#include "DirPartVolume.h"
#include "CallVolumeData.h"
#include "moldyn/DirectionalParticleDataCall.h"
#include "param/ButtonParam.h"
#include "param/IntParam.h"
#include "param/FloatParam.h"
#include "vislib/assert.h"
#include "vislib/Vector.h"
#include "vislib/ShallowVector.h"
#include "vislib/Log.h"
#include "vislib/ConsoleProgressBar.h"
#include "vislib/Matrix.h"
#include "vislib/pcautils.h"
#include <cmath>

using namespace megamol::core;

namespace megamol {
namespace core {
namespace moldyn {
namespace dirPartVolUtil {

    /**
     * Phat voxel collecting the direction vectors
     */
    class PhatVoxel {
    private:

        mutable vislib::math::Vector<float, 3> v;
        mutable vislib::math::Matrix<float, 3, vislib::math::COLUMN_MAJOR> mat;
        mutable float vv;
        mutable unsigned int vc;
        mutable bool fin;

    public:

        /**
         * Initializes this voxel
         */
        void Init(void) {
            v.Set(0.0f, 0.0f, 0.0f);
            mat.SetNull();
            vv = 0.0f;
            vc = 0;
            fin = false;
        }

        /**
         * Splats a direction vector 'dir' (3 floats) with weight 'w' onto this voxel
         *
         * @param dir The direction vector (3 floats)
         * @param w The weight
         */
        void Splat(const float* dir, float w) {
            ASSERT(!fin);
            vislib::math::Vector<float, 3> d(dir);
            d *= w;
            v += d;
            vv += d.Length();
            vc += 2;
            mat(0, 0) += (d[0] * d[0]);
            mat(0, 1) += (d[0] * d[1]);
            mat(0, 2) += (d[0] * d[2]);
            mat(1, 0) += (d[1] * d[0]);
            mat(1, 1) += (d[1] * d[1]);
            mat(1, 2) += (d[1] * d[2]);
            mat(2, 0) += (d[2] * d[0]);
            mat(2, 1) += (d[2] * d[1]);
            mat(2, 2) += (d[2] * d[2]);
            d *= -1.0f;
            mat(0, 0) += (d[0] * d[0]);
            mat(0, 1) += (d[0] * d[1]);
            mat(0, 2) += (d[0] * d[2]);
            mat(1, 0) += (d[1] * d[0]);
            mat(1, 1) += (d[1] * d[1]);
            mat(1, 2) += (d[1] * d[2]);
            mat(2, 0) += (d[2] * d[0]);
            mat(2, 1) += (d[2] * d[1]);
            mat(2, 2) += (d[2] * d[2]);
        }

        /**
         * Calculates the first value of this voxel
         *
         * @return the first value
         */
        float Value0(void) const {
            if (!fin) {
                fin = true;
                v /= vv;
                mat /= static_cast<float>(vc);
            }
            return v.Length();
        }

        /**
         * Calculates the second value of this voxel
         *
         * @return the second value
         */
        float Value1(void) const {
            if (!fin) {
                fin = true;
                v /= vv;
                mat /= static_cast<float>(vc);
            }

            float eiVal[3];
            unsigned int evc = mat.FindEigenvalues(eiVal, NULL, 3);
            if (evc == 0) return 0.0f;

            float maxEV, minEV;
            maxEV = minEV = eiVal[0];
            for (unsigned int i = 1; i < evc; i++) {
                if (maxEV < eiVal[i]) maxEV = eiVal[i];
                if (minEV > eiVal[i]) minEV = eiVal[i];
            }

            return (maxEV - minEV) / maxEV;
        }


        /**
         * Calculates the second value of this voxel
         * This is fractional anisotropy (http://en.wikipedia.org/wiki/Fractional_anisotropy)
         *
         * @return the second value
         */
        float Value2(void) const {
            if (!fin) {
                fin = true;
                v /= vv;
                mat /= static_cast<float>(vc);
            }

            float ev[3];
            unsigned int evc = mat.FindEigenvalues(ev, NULL, 3);
            if (evc == 0) return 0.0f;
            if (evc == 1) {
                ev[1] = ev[2] = ev[0];
            } else if (evc == 2) {
                ev[2] = ev[0]; // böh?
            }

            float fa = sqrt(0.5f) * sqrt(
                (ev[0] - ev[1])*(ev[0] - ev[1])
                + (ev[1] - ev[2])*(ev[1] - ev[2])
                + (ev[2] - ev[0])*(ev[2] - ev[0]))
                / sqrt(ev[0]*ev[0] + ev[1]*ev[1] + ev[2]*ev[2]);

            return fa;
        }

    };

}
}
}
}


/*
 * moldyn::DirPartVolume::DirPartVolume
 */
moldyn::DirPartVolume::DirPartVolume(void) : Module(),
        inDataSlot("inData", "Connects to the data source"),
        outDataSlot("outData", "Connects this data source"),
        xResSlot("resX", "Number of sample points in x direction"),
        yResSlot("resY", "Number of sample points in y direction"),
        zResSlot("resZ", "Number of sample points in z direction"),
        sampleRadiusSlot("radius", "Radius of the influence range of each particle in object space"),
        rebuildSlot("rebuild", "Force a rebuild of the volume"),
        dataHash(0), frameID(0), bbox(), data(NULL) {

    this->inDataSlot.SetCompatibleCall<moldyn::DirectionalParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);

    this->outDataSlot.SetCallback("CallVolumeData", "GetData", &DirPartVolume::outData);
    this->outDataSlot.SetCallback("CallVolumeData", "GetExtent", &DirPartVolume::outExtend);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->xResSlot << new param::IntParam(64, 2);
    this->MakeSlotAvailable(&this->xResSlot);

    this->yResSlot << new param::IntParam(64, 2);
    this->MakeSlotAvailable(&this->yResSlot);

    this->zResSlot << new param::IntParam(64, 2);
    this->MakeSlotAvailable(&this->zResSlot);

    this->sampleRadiusSlot << new param::FloatParam(2.0f, 0.0f);
    this->MakeSlotAvailable(&this->sampleRadiusSlot);

    this->rebuildSlot << new param::ButtonParam();
    this->MakeSlotAvailable(&this->rebuildSlot);

}


/*
 * moldyn::DirPartVolume::~DirPartVolume
 */
moldyn::DirPartVolume::~DirPartVolume(void) {
    this->Release();
}


/*
 * moldyn::DirPartVolume::create
 */
bool moldyn::DirPartVolume::create(void) {
    // intentionally empty
    return true;
}


/*
 * moldyn::DirPartVolume::release
 */
void moldyn::DirPartVolume::release(void) {
    ARY_SAFE_DELETE(this->data);
}


/*
 * moldyn::DirPartVolume::outExtend
 */
bool moldyn::DirPartVolume::outExtend(Call& caller) {
    CallVolumeData *cvd = dynamic_cast<CallVolumeData*>(&caller);
    if (cvd == NULL) return false;

    cvd->AccessBoundingBoxes().Clear();
    DirectionalParticleDataCall *dpd = this->inDataSlot.CallAs<DirectionalParticleDataCall>();
    if ((dpd == NULL) || (!(*dpd)(1))) {
        // no input data
        cvd->SetDataHash(0);
        cvd->SetFrameCount(1);

    } else {
        // input data in dpd
        cvd->SetDataHash(dpd->DataHash());
        cvd->SetFrameCount(dpd->FrameCount());
        this->bbox = dpd->AccessBoundingBoxes().ObjectSpaceBBox();

        float sx = 0.5f / static_cast<float>(this->xResSlot.Param<param::IntParam>()->Value());
        float sy = 0.5f / static_cast<float>(this->yResSlot.Param<param::IntParam>()->Value());
        float sz = 0.5f / static_cast<float>(this->zResSlot.Param<param::IntParam>()->Value());

        // voxel at cell center positions, I say ...
        this->bbox.Grow(
            -sx * this->bbox.Width(),
            -sy * this->bbox.Height(),
            -sz * this->bbox.Depth());

        cvd->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
    }
    cvd->SetUnlocker(NULL);

    return true;
}


/*
 * moldyn::DirPartVolume::outData
 */
bool moldyn::DirPartVolume::outData(Call& caller) {
    CallVolumeData *cvd = dynamic_cast<CallVolumeData*>(&caller);
    if (cvd == NULL) return false;

    DirectionalParticleDataCall *dpd = this->inDataSlot.CallAs<DirectionalParticleDataCall>();
    if (dpd == NULL) return false;
    dpd->SetFrameID(cvd->FrameID(), cvd->IsFrameForced());
    if (!(*dpd)(0)) return false;

    // We have data!

    bool rebuild = false;
    if (this->rebuildSlot.IsDirty()) {
        this->rebuildSlot.ResetDirty();
        rebuild = true;
    }
    if (this->xResSlot.IsDirty()) {
        this->xResSlot.ResetDirty();
        rebuild = true;
    }
    if (this->yResSlot.IsDirty()) {
        this->yResSlot.ResetDirty();
        rebuild = true;
    }
    if (this->zResSlot.IsDirty()) {
        this->zResSlot.ResetDirty();
        rebuild = true;
    }
    unsigned int sx = static_cast<unsigned int>(this->xResSlot.Param<param::IntParam>()->Value());
    unsigned int sy = static_cast<unsigned int>(this->yResSlot.Param<param::IntParam>()->Value());
    unsigned int sz = static_cast<unsigned int>(this->zResSlot.Param<param::IntParam>()->Value());

    if (this->dataHash != dpd->DataHash()) {
        rebuild = true;
        this->dataHash = dpd->DataHash();
    }
    if (this->frameID != dpd->FrameID()) {
        rebuild = true;
        this->frameID = dpd->FrameID();
    }
    if (this->sampleRadiusSlot.IsDirty()) {
        this->sampleRadiusSlot.ResetDirty();
        rebuild = true;
    }

    const unsigned int attrCnt = 3;
    //  0 : merged vector length
    //  1 : symmetric merged vector length
    //  2 : fractional anisotropie

    if (rebuild) {
        delete[] this->data;
        unsigned int voxSize = sx * sy * sz;
        this->data = new float[voxSize * attrCnt];

        float rad = this->sampleRadiusSlot.Param<param::FloatParam>()->Value();
        float vsx = this->bbox.Width() / static_cast<float>(sx);  // voxel size
        float vsy = this->bbox.Height() / static_cast<float>(sy); // voxel size
        float vsz = this->bbox.Depth() / static_cast<float>(sz);  // voxel size
        float vpx = this->bbox.Left();   // voxel origin
        float vpy = this->bbox.Bottom(); // voxel origin
        float vpz = this->bbox.Back();   // voxel origin
        int srx = static_cast<int>(ceil(rad / vsx)); // splat radius in voxels
        int sry = static_cast<int>(ceil(rad / vsy)); // splat radius in voxels
        int srz = static_cast<int>(ceil(rad / vsz)); // splat radius in voxels

        vislib::sys::Log::DefaultLog.WriteInfo("Creating fancy volume");

        vislib::sys::ConsoleProgressBar cpb;

        dirPartVolUtil::PhatVoxel *phats = new dirPartVolUtil::PhatVoxel[voxSize];
        for (unsigned int i = 0; i < voxSize; i++) {
            phats[i].Init();
        }

        unsigned int plCnt = dpd->GetParticleListCount();
        UINT64 allPartCnt = 0;

        for (unsigned int pli = 0; pli < plCnt; pli++) {
            DirectionalParticleDataCall::Particles& parts = dpd->AccessParticles(pli);
            UINT64 partCnt = parts.GetCount();
            if (partCnt == 0) continue;

            const unsigned char *vertPtr = static_cast<const unsigned char*>(parts.GetVertexData());
            SimpleSphericalParticles::VertexDataType vertDT = parts.GetVertexDataType();
            unsigned int vertStp = parts.GetVertexDataStride();

            const unsigned char *dirPtr = static_cast<const unsigned char*>(parts.GetDirData());
            DirectionalParticles::DirDataType dirDT = parts.GetDirDataType();
            unsigned int dirStp = parts.GetDirDataStride();
            if (dirStp < sizeof(float) * 3) dirStp = sizeof(float) * 3; // everything between 0 and this is stupid anyway

            if ((vertDT == SimpleSphericalParticles::VERTDATA_NONE)
                || (dirDT == DirectionalParticles::DIRDATA_NONE)) continue;

            if (vertDT == SimpleSphericalParticles::VERTDATA_SHORT_XYZ) continue; // I don't care for shorts

            allPartCnt += parts.GetCount();
        }

        cpb.Start("Progress", static_cast<vislib::sys::ConsoleProgressBar::Size>(allPartCnt));

        UINT64 allPartIdx = 0;
        for (unsigned int pli = 0; pli < plCnt; pli++) {
            DirectionalParticleDataCall::Particles& parts = dpd->AccessParticles(pli);
            UINT64 partCnt = parts.GetCount();
            if (partCnt == 0) continue;

            const unsigned char *vertPtr = static_cast<const unsigned char*>(parts.GetVertexData());
            SimpleSphericalParticles::VertexDataType vertDT = parts.GetVertexDataType();
            unsigned int vertStp = parts.GetVertexDataStride();

            const unsigned char *dirPtr = static_cast<const unsigned char*>(parts.GetDirData());
            DirectionalParticles::DirDataType dirDT = parts.GetDirDataType();
            unsigned int dirStp = parts.GetDirDataStride();
            if (dirStp < sizeof(float) * 3) dirStp = sizeof(float) * 3; // everything between 0 and this is stupid anyway

            if ((vertDT == SimpleSphericalParticles::VERTDATA_NONE)
                || (dirDT == DirectionalParticles::DIRDATA_NONE)) continue;

            if (vertDT == SimpleSphericalParticles::VERTDATA_SHORT_XYZ) continue; // I don't care for shorts

            if (vertStp < sizeof(float) * ((vertDT == SimpleSphericalParticles::VERTDATA_FLOAT_XYZ) ? 3 : 4)) {
                // everything between 0 and this is stupid anyway
                vertStp = sizeof(float) * ((vertDT == SimpleSphericalParticles::VERTDATA_FLOAT_XYZ) ? 3 : 4);
            }

            // splatting the particle into the volume
            for (UINT64 pi = 0; pi < partCnt; pi++, allPartIdx++, vertPtr += vertStp, dirPtr += dirStp) {
                const float *vert = reinterpret_cast<const float*>(vertPtr);
                const float *dir = reinterpret_cast<const float*>(dirPtr);

                // the particle voxel position
                int vppx = static_cast<int>(0.5f + (vert[0] - vpx) / vsx);
                int vppy = static_cast<int>(0.5f + (vert[1] - vpy) / vsy);
                int vppz = static_cast<int>(0.5f + (vert[2] - vpz) / vsz);

                // splat
                for (int xo = - srx; xo <= srx; xo++) {
                    int x = vppx + xo;
                    if ((x < 0) || (x >= static_cast<int>(sx))) continue;
                    float dx = (static_cast<float>(x) * vsx + vpx) - vert[0];
                    dx *= dx;

                    for (int yo = - sry; yo <= sry; yo++) {
                        int y = vppy + yo;
                        if ((y < 0) || (y >= static_cast<int>(sy))) continue;
                        float dy = (static_cast<float>(y) * vsy + vpy) - vert[1];
                        dy *= dy;

                        for (int zo = - srz; zo <= srz; zo++) {
                            int z = vppz + zo;
                            if ((z < 0) || (z >= static_cast<int>(sz))) continue;
                            float dz = (static_cast<float>(z) * vsz + vpz) - vert[2];
                            dz *= dz;

                            // particle voxel distance
                            float dist = sqrt(dx + dy + dz);
                            if (dist > rad) continue;

                            // dist to weight
                            dist /= rad;
                            dist = (2.0f * dist - 3.0f) * dist * dist + 1.0f;

                            phats[x + sx * (y + sy * z)].Splat(dir, dist);
                        }
                    }
                }
                if ((allPartIdx % 100) == 0) cpb.Set(static_cast<vislib::sys::ConsoleProgressBar::Size>(allPartIdx));
            }
        }
        cpb.Stop();

        vislib::sys::Log::DefaultLog.WriteInfo("Fancy volume splatted");

        for (unsigned int i = 0; i < voxSize; i++) {
            this->data[i + 0 * voxSize] = phats[i].Value0();
            this->data[i + 1 * voxSize] = phats[i].Value1();
            this->data[i + 2 * voxSize] = phats[i].Value2();
        }

        vislib::sys::Log::DefaultLog.WriteInfo("Fancy volume collected");

        delete[] phats;

    }

    cvd->SetAttributeCount(attrCnt);
    cvd->Attribute(0).SetName("o");
    cvd->Attribute(0).SetType(CallVolumeData::TYPE_FLOAT);
    cvd->Attribute(0).SetData(&this->data[sx * sy * sz * 0]);
    cvd->Attribute(1).SetName("s");
    cvd->Attribute(1).SetType(CallVolumeData::TYPE_FLOAT);
    cvd->Attribute(1).SetData(&this->data[sx * sy * sz * 1]);
    cvd->Attribute(2).SetName("a");
    cvd->Attribute(2).SetType(CallVolumeData::TYPE_FLOAT);
    cvd->Attribute(2).SetData(&this->data[sx * sy * sz * 2]);
    cvd->SetDataHash(this->dataHash);
    cvd->SetFrameID(this->frameID);
    cvd->SetSize(sx, sy, sz);
    cvd->SetUnlocker(NULL);

    return true;
}
