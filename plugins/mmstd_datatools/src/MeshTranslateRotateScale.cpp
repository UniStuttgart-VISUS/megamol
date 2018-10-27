/*
 * MeshTranslateRotateScale.cpp
 *
 * Copyright (C) 2018 MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "MeshTranslateRotateScale.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "vislib/math/Matrix4.h"

using namespace megamol;
using namespace megamol::stdplugin;


/*
 * datatools::MeshTranslateRotateScale::MeshTranslateRotateScale
 */
datatools::MeshTranslateRotateScale::MeshTranslateRotateScale(void)
    : AbstractMeshManipulator("outData", "indata")
    , translateSlot("translation", "Translates the particles in x, y, z direction")
    , quaternionSlot("quaternion", "Rotates the particles around x, y, z axes")
    , scaleSlot("scale", "Scales the particle data")
    , finalData(NULL), mesh(NULL) {
    this->translateSlot.SetParameter(new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0, 0, 0)));
    this->MakeSlotAvailable(&this->translateSlot);

    this->quaternionSlot.SetParameter(new core::param::Vector4fParam(vislib::math::Vector<float, 4>(0, 0, 0, 1)));
    this->MakeSlotAvailable(&this->quaternionSlot);

    this->scaleSlot.SetParameter(new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1, 1, 1)));
    this->MakeSlotAvailable(&this->scaleSlot);
}


/*
 * datatools::MeshTranslateRotateScale::~MeshTranslateRotateScale
 */
datatools::MeshTranslateRotateScale::~MeshTranslateRotateScale(void) {
    this->Release();
    delete[] finalData;
    delete[] mesh;
}


/*
 * datatools::MeshTranslateRotateScale::InterfaceIsDirty
 */
bool datatools::MeshTranslateRotateScale::InterfaceIsDirty() const {
    return (translateSlot.IsDirty() || quaternionSlot.IsDirty() || scaleSlot.IsDirty());
}


/*
 * datatools::MeshTranslateRotateScale::InterfaceResetDirty
 */
void datatools::MeshTranslateRotateScale::InterfaceResetDirty() {
    translateSlot.ResetDirty();
    quaternionSlot.ResetDirty();
    scaleSlot.ResetDirty();
}


/*
 * datatools::MeshTranslateRotateScale::manipulateData
 */
bool datatools::MeshTranslateRotateScale::manipulateData(
    geocalls::CallTriMeshData& outData, geocalls::CallTriMeshData& inData) {


    outData = inData; // also transfers the unlocker to 'outData'

    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData

    const float transX = translateSlot.Param<core::param::Vector3fParam>()->Value().GetX();
    const float transY = translateSlot.Param<core::param::Vector3fParam>()->Value().GetY();
    const float transZ = translateSlot.Param<core::param::Vector3fParam>()->Value().GetZ();

    const float scaleX = scaleSlot.Param<core::param::Vector3fParam>()->Value().GetX();
    const float scaleY = scaleSlot.Param<core::param::Vector3fParam>()->Value().GetY();
    const float scaleZ = scaleSlot.Param<core::param::Vector3fParam>()->Value().GetZ();

    // Calculate matrices
    vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> transMX(
        1, 0, 0, transX, 0, 1, 0, transY, 0, 0, 1, transZ, 0, 0, 0, 1);

    vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> scaleMX(
        scaleX, 0, 0, 0, 0, scaleY, 0, 0, 0, 0, scaleZ, 0, 0, 0, 0, 1);

    /*
    const float rotX = 2 * M_PI * rotateSlot.Param<core::param::Vector3fParam>()->Value().GetX() / 360.0f;
    const float rotY = 2 * M_PI * rotateSlot.Param<core::param::Vector3fParam>()->Value().GetY() / 360.0f;
    const float rotZ = 2 * M_PI * rotateSlot.Param<core::param::Vector3fParam>()->Value().GetZ() / 360.0f;


    vislib::math::Quaternion<float> qx(rotX, vislib::math::Vector<float, 3>(1.0f, 0.0f, 0.0f));
    vislib::math::Quaternion<float> qy(rotY, vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f));
    vislib::math::Quaternion<float> qz(rotZ, vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f));

    vislib::math::Matrix<float, 4, vislib::math::MatrixLayout::COLUMN_MAJOR> totRotMX;
    totRotMX = qx * qy * qz;
    */

    vislib::math::Quaternion<float> q(quaternionSlot.Param<core::param::Vector4fParam>()->Value().GetX(),
        quaternionSlot.Param<core::param::Vector4fParam>()->Value().GetY(),
        quaternionSlot.Param<core::param::Vector4fParam>()->Value().GetZ(),
        quaternionSlot.Param<core::param::Vector4fParam>()->Value().GetW());

    auto qr = q.GetK();
    auto qi = q.GetI();
    auto qj = q.GetJ();
    auto qk = q.GetR();
    auto s = 1.0f / (qr * qr + qi * qi + qj * qj + qk * qk);

    vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> rotMX(1 - 2 * s * (qj * qj + qk * qk),
        2 * s * (qi * qj - qk * qr), 2 * s * (qi * qk - qj * qr), 0, 2 * s * (qi * qj + qk * qr),
        1 - 2 * s * (qi * qi + qk * qk), 2 * s * (qj * qk - qi * qr), 0, 2 * s * (qi * qk - qj * qr),
        2 * s * (qj * qk + qi * qr), 1 - 2 * s * (qi * qi + qj * qj), 0, 0, 0, 0, 1);

    auto bboxCenterX = inData.GetBoundingBoxes().ObjectSpaceBBox().CalcCenter().GetX();
    auto bboxCenterY = inData.GetBoundingBoxes().ObjectSpaceBBox().CalcCenter().GetY();
    auto bboxCenterZ = inData.GetBoundingBoxes().ObjectSpaceBBox().CalcCenter().GetZ();

    vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> origTransMX(
        1, 0, 0, -bboxCenterX, 0, 1, 0, -bboxCenterY, 0, 0, 1, -bboxCenterZ, 0, 0, 0, 1);

    vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> invOrigTransMX(
        1, 0, 0, bboxCenterX, 0, 1, 0, bboxCenterY, 0, 0, 1, bboxCenterZ, 0, 0, 0, 1);

    auto const totMX = transMX * invOrigTransMX * scaleMX * rotMX * origTransMX;
    //auto const totMX = transMX * invOrigTransMX * scaleMX * origTransMX;

    unsigned int mlc = outData.Count();
    vislib::math::Vector<float, 4> pos;

    finalData = new float*[mlc];
    this->mesh = new geocalls::CallTriMeshData::Mesh[mlc];
    for (unsigned int i = 0; i < mlc; i++) {
        auto& m = outData.Objects()[i];

        uint64_t vcnt = m.GetVertexCount();
        uint64_t fcnt = m.GetTriCount();

        auto vdt = m.GetVertexDataType();
        const float* vd;
        if (vdt == geocalls::CallTriMeshData::Mesh::DT_FLOAT) {
            vd = m.GetVertexPointerFloat();
        } else {
            vd = reinterpret_cast<const float*>(m.GetVertexPointerDouble());
        }

        unsigned char* cd;
        unsigned int* indexd;

        if (m.HasColourPointer()) {
            if (m.GetColourDataType() == geocalls::CallTriMeshData::Mesh::DT_BYTE) {
                cd = const_cast<unsigned char*>(m.GetColourPointerByte());
            }
        }
        if (m.HasTriIndexPointer()) {
            if (m.GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_UINT32) {
                indexd = const_cast<unsigned int*>(m.GetTriIndexPointerUInt32());
            }
        }

        if (m.HasNormalPointer()) {
        }
        if (m.HasTextureCoordinatePointer()) {
        }
        if (m.HasVertexAttribPointer()) {
        }


        finalData[i] = new float[vcnt * 3];
        for (size_t loop = 0; loop < vcnt; loop++) {

            pos.SetX(vd[3 * loop]);
            pos.SetY(vd[3 * loop + 1]);
            pos.SetZ(vd[3 * loop + 2]);
            pos.SetW(1.0f);

            pos = totMX * pos;

            finalData[i][3 * loop + 0] = pos.GetX();
            finalData[i][3 * loop + 1] = pos.GetY();
            finalData[i][3 * loop + 2] = pos.GetZ();


            // Does not work ??
            // finalData.emplace_back(pos.GetX(), pos.GetY(), pos.GetZ(), part.col.GetRf(), part.col.GetGf(),
            //    part.col.GetBf(), part.col.GetAf());
        }

        this->mesh[i].SetVertexData(vcnt, finalData[i], NULL, cd, NULL, false);
        this->mesh[i].SetTriangleData(fcnt, indexd, false);
    }

    outData.SetObjects(mlc, this->mesh);

    return true;
}


#if 0
/*
 * datatools::MeshTranslateRotateScale::manipulateExtent
 */
bool datatools::MeshTranslateRotateScale::manipulateExtent(
    geocalls::CallTriMeshData& outData, geocalls::CallTriMeshData& inData) {

    auto lbb = static_cast<vislib::math::Vector<float, 3>>(inData.GetBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack());
    auto rtf = static_cast<vislib::math::Vector<float, 3>>(inData.GetBoundingBoxes().ObjectSpaceBBox().GetRightTopFront());
    lbb = totMX * lbb;rotMX
    rtf = totMX * rtf;
    vislib::math::Cuboid<float> newBox;
    newBox.Set(lbb.GetX(), lbb.GetY(), lbb.GetZ(), rtf.GetX(), rtf.GetY(), rtf.GetZ());

    auto scale = 10.0f / newBox.LongestEdge();

    outData.AccessBoundingBoxes().SetObjectSpaceBBox(newBox);
    outData.AccessBoundingBoxes().MakeScaledWorld(scale);


    }

#endif
