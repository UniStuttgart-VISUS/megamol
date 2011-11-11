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
#include "param/IntParam.h"
#include "param/FloatParam.h"
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

    /** number of sample buckets */
    static const unsigned int sampleDirCount = 194;

    /** directions of sample buckets */
    static const float sampleDir[] = {
        0.0f, 0.0f, 1.0f, 0.3826903f, 0.0f, 0.923876643f,
        0.0f, 0.3826903f, 0.923876643f, 0.707106769f, 0.0f, 0.707106769f,
        0.408214957f, 0.408214957f, 0.8165299f, 0.0f, 0.707106769f, 0.707106769f,
        0.923876643f, 0.0f, 0.3826903f, 0.759273231f, 0.426784962f, 0.491282672f,
        0.426784962f, 0.759273231f, 0.491282672f, 0.0f, 0.923876643f, 0.3826903f,
        1.0f, 0.0f, 0.0f, 0.923876643f, 0.3826903f, 0.0f,
        0.707106769f, 0.707106769f, 0.0f, 0.3826903f, 0.923876643f, 0.0f,
        0.0f, 1.0f, 0.0f, -0.3826903f, 0.0f, 0.923876643f,
        -0.408214957f, 0.408214957f, 0.8165299f, -0.707106769f, 0.0f, 0.707106769f,
        -0.426784962f, 0.759273231f, 0.491282672f, -0.759273231f, 0.426784962f, 0.491282672f,
        -0.923876643f, 0.0f, 0.3826903f, -0.3826903f, 0.923876643f, 0.0f,
        -0.707106769f, 0.707106769f, 0.0f, -0.923876643f, 0.3826903f, 0.0f,
        -1.0f, 0.0f, 0.0f, 0.0f, -0.3826903f, 0.923876643f,
        -0.408214957f, -0.408214957f, 0.8165299f, 0.0f, -0.707106769f, 0.707106769f,
        -0.759273231f, -0.426784962f, 0.491282672f, -0.426784962f, -0.759273231f, 0.491282672f,
        0.0f, -0.923876643f, 0.3826903f, -0.923876643f, -0.3826903f, 0.0f,
        -0.707106769f, -0.707106769f, 0.0f, -0.3826903f, -0.923876643f, 0.0f,
        0.0f, -1.0f, 0.0f, 0.408214957f, -0.408214957f, 0.8165299f,
        0.426784962f, -0.759273231f, 0.491282672f, 0.759273231f, -0.426784962f, 0.491282672f,
        0.3826903f, -0.923876643f, 0.0f, 0.707106769f, -0.707106769f, 0.0f,
        0.923876643f, -0.3826903f, 0.0f, 0.0f, 0.0f, -1.0f,
        0.0f, 0.3826903f, -0.923876643f, 0.3826903f, 0.0f, -0.923876643f,
        0.0f, 0.707106769f, -0.707106769f, 0.408214957f, 0.408214957f, -0.8165299f,
        0.707106769f, 0.0f, -0.707106769f, 0.0f, 0.923876643f, -0.3826903f,
        0.426784962f, 0.759273231f, -0.491282672f, 0.759273231f, 0.426784962f, -0.491282672f,
        0.923876643f, 0.0f, -0.3826903f, -0.3826903f, 0.0f, -0.923876643f,
        -0.707106769f, 0.0f, -0.707106769f, -0.408214957f, 0.408214957f, -0.8165299f,
        -0.923876643f, 0.0f, -0.3826903f, -0.759273231f, 0.426784962f, -0.491282672f,
        -0.426784962f, 0.759273231f, -0.491282672f, 0.0f, -0.3826903f, -0.923876643f,
        0.0f, -0.707106769f, -0.707106769f, -0.408214957f, -0.408214957f, -0.8165299f,
        0.0f, -0.923876643f, -0.3826903f, -0.426784962f, -0.759273231f, -0.491282672f,
        -0.759273231f, -0.426784962f, -0.491282672f, 0.408214957f, -0.408214957f, -0.8165299f,
        0.759273231f, -0.426784962f, -0.491282672f, 0.426784962f, -0.759273231f, -0.491282672f,
        0.13202025f, 0.13202025f, 0.982416034f, 0.5168325f, 0.140839145f, 0.844422042f,
        0.273712784f, 0.273712784f, 0.922042668f, 0.140839145f, 0.5168325f, 0.844422042f,
        0.82494843f, 0.147296146f, 0.5456775f, 0.6518121f, 0.290336341f, 0.700603962f,
        0.5527095f, 0.5527095f, 0.623718143f, 0.290336341f, 0.6518121f, 0.700603962f,
        0.147296146f, 0.82494843f, 0.5456775f, 0.982416034f, 0.13202025f, 0.13202025f,
        0.9095447f, 0.2824114f, 0.304913461f, 0.8319433f, 0.5278555f, 0.170993909f,
        0.663815737f, 0.663815737f, 0.344524831f, 0.5278555f, 0.8319433f, 0.170993909f,
        0.2824114f, 0.9095447f, 0.304913461f, 0.13202025f, 0.982416034f, 0.13202025f,
        -0.13202025f, 0.13202025f, 0.982416034f, -0.140839145f, 0.5168325f, 0.844422042f,
        -0.273712784f, 0.273712784f, 0.922042668f, -0.5168325f, 0.140839145f, 0.844422042f,
        -0.147296146f, 0.82494843f, 0.5456775f, -0.290336341f, 0.6518121f, 0.700603962f,
        -0.5527095f, 0.5527095f, 0.623718143f, -0.6518121f, 0.290336341f, 0.700603962f,
        -0.82494843f, 0.147296146f, 0.5456775f, -0.13202025f, 0.982416034f, 0.13202025f,
        -0.2824114f, 0.9095447f, 0.304913461f, -0.5278555f, 0.8319433f, 0.170993909f,
        -0.663815737f, 0.663815737f, 0.344524831f, -0.8319433f, 0.5278555f, 0.170993909f,
        -0.9095447f, 0.2824114f, 0.304913461f, -0.982416034f, 0.13202025f, 0.13202025f,
        -0.13202025f, -0.13202025f, 0.982416034f, -0.5168325f, -0.140839145f, 0.844422042f,
        -0.273712784f, -0.273712784f, 0.922042668f, -0.140839145f, -0.5168325f, 0.844422042f,
        -0.82494843f, -0.147296146f, 0.5456775f, -0.6518121f, -0.290336341f, 0.700603962f,
        -0.5527095f, -0.5527095f, 0.623718143f, -0.290336341f, -0.6518121f, 0.700603962f,
        -0.147296146f, -0.82494843f, 0.5456775f, -0.982416034f, -0.13202025f, 0.13202025f,
        -0.9095447f, -0.2824114f, 0.304913461f, -0.8319433f, -0.5278555f, 0.170993909f,
        -0.663815737f, -0.663815737f, 0.344524831f, -0.5278555f, -0.8319433f, 0.170993909f,
        -0.2824114f, -0.9095447f, 0.304913461f, -0.13202025f, -0.982416034f, 0.13202025f,
        0.13202025f, -0.13202025f, 0.982416034f, 0.140839145f, -0.5168325f, 0.844422042f,
        0.273712784f, -0.273712784f, 0.922042668f, 0.5168325f, -0.140839145f, 0.844422042f,
        0.147296146f, -0.82494843f, 0.5456775f, 0.290336341f, -0.6518121f, 0.700603962f,
        0.5527095f, -0.5527095f, 0.623718143f, 0.6518121f, -0.290336341f, 0.700603962f,
        0.82494843f, -0.147296146f, 0.5456775f, 0.13202025f, -0.982416034f, 0.13202025f,
        0.2824114f, -0.9095447f, 0.304913461f, 0.5278555f, -0.8319433f, 0.170993909f,
        0.663815737f, -0.663815737f, 0.344524831f, 0.8319433f, -0.5278555f, 0.170993909f,
        0.9095447f, -0.2824114f, 0.304913461f, 0.982416034f, -0.13202025f, 0.13202025f,
        0.13202025f, 0.13202025f, -0.982416034f, 0.140839145f, 0.5168325f, -0.844422042f,
        0.273712784f, 0.273712784f, -0.922042668f, 0.5168325f, 0.140839145f, -0.844422042f,
        0.147296146f, 0.82494843f, -0.5456775f, 0.290336341f, 0.6518121f, -0.700603962f,
        0.5527095f, 0.5527095f, -0.623718143f, 0.6518121f, 0.290336341f, -0.700603962f,
        0.82494843f, 0.147296146f, -0.5456775f, 0.13202025f, 0.982416034f, -0.13202025f,
        0.2824114f, 0.9095447f, -0.304913461f, 0.5278555f, 0.8319433f, -0.170993909f,
        0.663815737f, 0.663815737f, -0.344524831f, 0.8319433f, 0.5278555f, -0.170993909f,
        0.9095447f, 0.2824114f, -0.304913461f, 0.982416034f, 0.13202025f, -0.13202025f,
        -0.13202025f, 0.13202025f, -0.982416034f, -0.5168325f, 0.140839145f, -0.844422042f,
        -0.273712784f, 0.273712784f, -0.922042668f, -0.140839145f, 0.5168325f, -0.844422042f,
        -0.82494843f, 0.147296146f, -0.5456775f, -0.6518121f, 0.290336341f, -0.700603962f,
        -0.5527095f, 0.5527095f, -0.623718143f, -0.290336341f, 0.6518121f, -0.700603962f,
        -0.147296146f, 0.82494843f, -0.5456775f, -0.982416034f, 0.13202025f, -0.13202025f,
        -0.9095447f, 0.2824114f, -0.304913461f, -0.8319433f, 0.5278555f, -0.170993909f,
        -0.663815737f, 0.663815737f, -0.344524831f, -0.5278555f, 0.8319433f, -0.170993909f,
        -0.2824114f, 0.9095447f, -0.304913461f, -0.13202025f, 0.982416034f, -0.13202025f,
        -0.13202025f, -0.13202025f, -0.982416034f, -0.140839145f, -0.5168325f, -0.844422042f,
        -0.273712784f, -0.273712784f, -0.922042668f, -0.5168325f, -0.140839145f, -0.844422042f,
        -0.147296146f, -0.82494843f, -0.5456775f, -0.290336341f, -0.6518121f, -0.700603962f,
        -0.5527095f, -0.5527095f, -0.623718143f, -0.6518121f, -0.290336341f, -0.700603962f,
        -0.82494843f, -0.147296146f, -0.5456775f, -0.13202025f, -0.982416034f, -0.13202025f,
        -0.2824114f, -0.9095447f, -0.304913461f, -0.5278555f, -0.8319433f, -0.170993909f,
        -0.663815737f, -0.663815737f, -0.344524831f, -0.8319433f, -0.5278555f, -0.170993909f,
        -0.9095447f, -0.2824114f, -0.304913461f, -0.982416034f, -0.13202025f, -0.13202025f,
        0.13202025f, -0.13202025f, -0.982416034f, 0.5168325f, -0.140839145f, -0.844422042f,
        0.273712784f, -0.273712784f, -0.922042668f, 0.140839145f, -0.5168325f, -0.844422042f,
        0.82494843f, -0.147296146f, -0.5456775f, 0.6518121f, -0.290336341f, -0.700603962f,
        0.5527095f, -0.5527095f, -0.623718143f, 0.290336341f, -0.6518121f, -0.700603962f,
        0.147296146f, -0.82494843f, -0.5456775f, 0.982416034f, -0.13202025f, -0.13202025f,
        0.9095447f, -0.2824114f, -0.304913461f, 0.8319433f, -0.5278555f, -0.170993909f,
        0.663815737f, -0.663815737f, -0.344524831f, 0.5278555f, -0.8319433f, -0.170993909f,
        0.2824114f, -0.9095447f, -0.304913461f, 0.13202025f, -0.982416034f, -0.13202025f
    };

    /** Indices of opposing sample buckets */
    static const unsigned int opoDir[sampleDirCount] = {
        41, 51, 57, 52, 59, 58, 54, 62, 61, 60, 24, 31, 32, 33, 34, 43,
        63, 46, 65, 64, 50, 38, 39, 40, 10, 42, 45, 44, 49, 48, 47, 11,
        12, 13, 14, 53, 56, 55, 21, 22, 23, 0, 25, 15, 27, 26, 17, 30,
        29, 28, 20, 1, 3, 35, 6, 37, 36, 2, 5, 4, 9, 8, 7, 16,
        19, 18, 162, 165, 164, 163, 170, 169, 168, 167, 166, 177, 176, 175, 174, 173,
        172, 171, 178, 181, 180, 179, 186, 185, 184, 183, 182, 193, 192, 191, 190, 189,
        188, 187, 130, 133, 132, 131, 138, 137, 136, 135, 134, 145, 144, 143, 142, 141,
        140, 139, 146, 149, 148, 147, 154, 153, 152, 151, 150, 161, 160, 159, 158, 157,
        156, 155, 98, 101, 100, 99, 106, 105, 104, 103, 102, 113, 112, 111, 110, 109,
        108, 107, 114, 117, 116, 115, 122, 121, 120, 119, 118, 129, 128, 127, 126, 125,
        124, 123, 66, 69, 68, 67, 74, 73, 72, 71, 70, 81, 80, 79, 78, 77,
        76, 75, 82, 85, 84, 83, 90, 89, 88, 87, 86, 97, 96, 95, 94, 93,
        92, 91
    };

    /**
     * Phat voxel collecting the direction vectors
     */
    class PhatVoxel {
    private:

        /** The sample buckets */
        float samples[sampleDirCount];

    public:

        /**
         * Initializes this voxel
         */
        void Init(void) {
            for (unsigned int i = 0; i < sampleDirCount; i++) {
                this->samples[i] = 0.0f;
            }
        }

        /**
         * Splats a direction vector 'dir' (3 floats) with weight 'w' onto this voxel
         *
         * @param dir The direction vector (3 floats)
         * @param w The weight
         */
        void Splat(const float* dir, float w) {
            vislib::math::Vector<float, 3> d(dir);
            vislib::math::ShallowVector<float, 3> sd(d.PeekComponents());
            float len = d.Normalise();
            if (vislib::math::IsEqual(len, 0.0f)) return;

            float minAng = 100.0f;
            unsigned int minIdx = 300;
            for (unsigned int i = 0; i < sampleDirCount; i++) {
                const vislib::math::ShallowVector<float, 3> b(const_cast<float*>(&sampleDir[i * 3]));
                float ang = b.Angle(sd);
                if (ang < minAng) {
                    minAng = ang;
                    minIdx = i;
                }
            }
            if (minIdx == 300) {
                return;
            }

            this->samples[minIdx] += len * w;

        }

        /**
         * Calculates the first value of this voxel
         *
         * @return the first value
         */
        float Value0(void) const {
            vislib::math::Vector<float, 3> v;
            float vv = 0.0f;
            for (unsigned int i = 0; i < sampleDirCount; i++) {
                const vislib::math::ShallowVector<float, 3> b(const_cast<float*>(&sampleDir[i * 3]));
                v += b * this->samples[i];
                vv += this->samples[i];
            }

            return v.Length() / vv;
        }

        /**
         * Calculates the second value of this voxel
         *
         * @return the second value
         */
        float Value1(void) const {
            vislib::math::Matrix<float, 3, vislib::math::COLUMN_MAJOR> mat;
            vislib::math::Vector<float, 3> v[sampleDirCount];
            float vv = 0.0f;
            for (unsigned int i = 0; i < sampleDirCount; i++) {
                v[i].Set(sampleDir[i * 3], sampleDir[i * 3 + 1], sampleDir[i * 3 + 2]);
                v[i] *= (this->samples[i] + this->samples[opoDir[i]]);
                if (vv < this->samples[i]) {
                    vv = this->samples[i];
                }
            }
            vislib::math::CalcCovarianceMatrix(mat, &v[0], sampleDirCount);

            float eiVal[3];
            unsigned int evc = mat.FindEigenvalues(eiVal, NULL, 3);
            if (evc == 0) return 0.0f;

            float maxEV, minEV;
            maxEV = minEV = eiVal[0];
            for (unsigned int i = 1; i < evc; i++) {
                if (maxEV < eiVal[i]) maxEV = eiVal[i];
                if (minEV > eiVal[i]) minEV = eiVal[i];
            }

            return (maxEV - minEV) / vv;
        }


        /**
         * Calculates the second value of this voxel
         * This is fractional anisotropy (http://en.wikipedia.org/wiki/Fractional_anisotropy)
         *
         * @return the second value
         */
        float Value2(void) const {
            vislib::math::Matrix<float, 3, vislib::math::COLUMN_MAJOR> mat;
            vislib::math::Vector<float, 3> v[sampleDirCount];
            float vv = 0.0f;
            for (unsigned int i = 0; i < sampleDirCount; i++) {
                v[i].Set(sampleDir[i * 3], sampleDir[i * 3 + 1], sampleDir[i * 3 + 2]);
                v[i] *= (this->samples[i] + this->samples[opoDir[i]]);
                if (vv < this->samples[i]) {
                    vv = this->samples[i];
                }
            }
            vislib::math::CalcCovarianceMatrix(mat, &v[0], sampleDirCount);

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
