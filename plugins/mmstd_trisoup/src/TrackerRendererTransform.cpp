/*
 * TrackerRendererTransform.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "TrackerRendererTransform.h"
#include "mmcore/param/FloatParam.h"
#ifdef WITH_VRPN
#include "mmcore/param/StringParam.h"
#endif /* WITH_VRPN */
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/assert.h"
#include "vislib/sys/Log.h"
#include "vislib/math/Matrix.h"
#include "vislib/memutils.h"
#include "vislib/math/Quaternion.h"
#include "vislib/math/ShallowQuaternion.h"
#include "vislib/math/ShallowVector.h"
#include "vislib/String.h"
#ifdef WITH_VRPN
#include "vislib/StringConverter.h"
#include "vislib/sys/SystemInformation.h"
#endif /* WITH_VRPN */
#include "vislib/math/Vector.h"
#include <GL/gl.h>

using namespace megamol;
using namespace megamol::trisoup;


/*
 * TrackerRendererTransform::TrackerRendererTransform
 */
TrackerRendererTransform::TrackerRendererTransform(void) : Renderer3DModule(),
        outRenderSlot("outrender", "The slot to call the real renderer"),
        translateSlot("translate", "The translation applied"),
        rotateSlot("rotate", "The rotation applied"),
        scaleSlot("scale", "The scale applied"),
        bboxMinSlot("bbox::min", "The minimum vector of the bounding box"),
        bboxMaxSlot("bbox::max", "The maximum vector of the bounding box")
#ifdef WITH_VRPN
        , vrpnConn(NULL),
        vrpnTracker(NULL),
        vrpnIsConnected(false),
        vrpnIsHealthy(true),
        vrpnClientSlot("vrpn::client", "Only connect to vrpn if the client name matches the local computer name"),
        vrpnServerSlot("vrpn::server", "The address of the vrpn server to connect to"),
        vrpnTrackerSlot("vrpn::tracker", "The name of the tracker object to track")
#endif /* WITH_VRPN */
        {
    this->outRenderSlot.SetCompatibleCall<core::view::CallRender3DDescription>();
    this->MakeSlotAvailable(&this->outRenderSlot);

    this->translateSlot << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
    this->MakeSlotAvailable(&this->translateSlot);

    this->rotateSlot << new core::param::Vector4fParam(
        vislib::math::ShallowVector<float, 4>(
            vislib::math::Quaternion<float>().PeekComponents()));
    this->MakeSlotAvailable(&this->rotateSlot);

    this->scaleSlot << new core::param::FloatParam(0.5f);
    this->MakeSlotAvailable(&this->scaleSlot);

    this->bboxMinSlot << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(-3.0f, -3.0f, -3.0f));
    this->MakeSlotAvailable(&this->bboxMinSlot);

    this->bboxMaxSlot << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(3.0f, 3.0f, 3.0f));
    this->MakeSlotAvailable(&this->bboxMaxSlot);

#ifdef WITH_VRPN
    this->vrpnClientSlot << new core::param::StringParam("");
    this->MakeSlotAvailable(&this->vrpnClientSlot);

    this->vrpnServerSlot << new core::param::StringParam("localhost:3883");
    this->MakeSlotAvailable(&this->vrpnServerSlot);
    this->vrpnServerSlot.ForceSetDirty();

    this->vrpnTrackerSlot << new core::param::StringParam("Tracker");
    this->MakeSlotAvailable(&this->vrpnTrackerSlot);
    this->vrpnTrackerSlot.ForceSetDirty();
#endif /* WITH_VRPN */

}


/*
 * TrackerRendererTransform::~TrackerRendererTransform
 */
TrackerRendererTransform::~TrackerRendererTransform(void) {
    this->Release();
#ifdef WITH_VRPN
    ASSERT(this->vrpnConn == NULL);
#endif /* WITH_VRPN */
}


/*
 * TrackerRendererTransform::create
 */
bool TrackerRendererTransform::create(void) {
    // intentionally empty
    return true;
}


/*
 * TrackerRendererTransform::GetExtents
 */
bool TrackerRendererTransform::GetExtents(core::Call& call) {
    core::view::CallRender3D *inCr3d = dynamic_cast<core::view::CallRender3D*>(&call);
    if (inCr3d == NULL) return false;

    inCr3d->AccessBoundingBoxes().Clear();
    inCr3d->SetTimeFramesCount(1);

    core::view::CallRender3D *outCr3d = this->outRenderSlot.CallAs<core::view::CallRender3D>();
    if ((outCr3d != NULL) && ((*outCr3d)(1))) {

        // TODO: calculate real clip box ... ok for now

        inCr3d->SetTimeFramesCount(outCr3d->TimeFramesCount());
    }

    const vislib::math::Vector<float, 3>& minV = this->bboxMinSlot.Param<core::param::Vector3fParam>()->Value();
    const vislib::math::Vector<float, 3>& maxV = this->bboxMaxSlot.Param<core::param::Vector3fParam>()->Value();
    inCr3d->AccessBoundingBoxes().SetWorldSpaceBBox(minV.X(), minV.Y(), minV.Z(), maxV.X(), maxV.Y(), maxV.Z());

    return true;
}


/*
 * TrackerRendererTransform::release
 */
void TrackerRendererTransform::release(void) {
#ifdef WITH_VRPN
    SAFE_DELETE(this->vrpnTracker);
    if (this->vrpnConn != NULL) {
        this->vrpnConn->removeReference();
        this->vrpnConn = NULL;
    }
#endif /* WITH_VRPN */
}


/*
 * TrackerRendererTransform::Render
 */
bool TrackerRendererTransform::Render(core::Call& call) {
    core::view::CallRender3D *inCr3d = dynamic_cast<core::view::CallRender3D*>(&call);
    if (inCr3d == NULL) return false;
    core::view::CallRender3D *outCr3d = this->outRenderSlot.CallAs<core::view::CallRender3D>();
    if (outCr3d == NULL) return false;

#ifdef WITH_VRPN

    static vislib::TString computerName;
    if (computerName.IsEmpty()) {
        vislib::sys::SystemInformation::ComputerName(computerName);
    }
    if (this->vrpnClientSlot.Param<core::param::StringParam>()->Value().Equals(computerName, false)) {
        if (this->vrpnServerSlot.IsDirty()) {
            this->vrpnServerSlot.ResetDirty();
            SAFE_DELETE(this->vrpnTracker);
            if (this->vrpnConn != NULL) {
                this->vrpnConn->removeReference();
                this->vrpnConn = NULL;
                this->vrpnIsConnected = false;
                this->vrpnIsHealthy = false;
            }
            this->vrpnConn = vrpn_get_connection_by_name(
                T2A(this->vrpnServerSlot.Param<core::param::StringParam>()->Value()));
            if (this->vrpnConn == NULL) {
                vislib::sys::Log::DefaultLog.WriteError(
                    _T("Unable to open VRPN connection to \"%s\""),
                    this->vrpnServerSlot.Param<core::param::StringParam>()->Value().PeekBuffer());
            }
        }
        if ((this->vrpnConn != NULL) && this->vrpnTrackerSlot.IsDirty()) {
            this->vrpnTrackerSlot.ResetDirty();
            if (this->vrpnTracker != NULL) {
                SAFE_DELETE(this->vrpnTracker);
            }
            this->vrpnTracker = new vrpn_Tracker_Remote(
                T2A(this->vrpnTrackerSlot.Param<core::param::StringParam>()->Value()),
                this->vrpnConn);
            if (this->vrpnTracker != NULL) {
                this->vrpnTracker->shutup = true;
                this->vrpnTracker->register_change_handler(
                    static_cast<void *>(this), &TrackerRendererTransform::vrpnTrackerCallback);
            } else {
                vislib::sys::Log::DefaultLog.WriteError(
                    _T("Unable to connect to tracker service of \"%s\""),
                    this->vrpnTrackerSlot.Param<core::param::StringParam>()->Value().PeekBuffer());
            }
        }
    }

    if (this->vrpnConn != NULL) {
        this->vrpnConn->mainloop();
        bool isConn = (this->vrpnConn->connected() != 0);
        bool isOk = (this->vrpnConn->doing_okay() != 0);
        if (this->vrpnIsConnected != isConn) {
            this->vrpnIsConnected = isConn;
            if (isConn) {
                vislib::sys::Log::DefaultLog.WriteInfo("VRPN connected");
            } else {
                vislib::sys::Log::DefaultLog.WriteInfo("VRPN disconnected");
            }
        }
        if (this->vrpnIsHealthy != isOk) {
            this->vrpnIsHealthy = isOk;
            if (isOk) {
                vislib::sys::Log::DefaultLog.WriteInfo("VRPN connection healthy");
            } else {
                vislib::sys::Log::DefaultLog.WriteInfo("VRPN connection broken");
            }
        }
    }
    if (this->vrpnTracker != NULL) {
        this->vrpnTracker->mainloop();
    }

#endif /* WITH_VRPN */

    ::glMatrixMode(GL_MODELVIEW);
    ::glPushMatrix();

    const vislib::math::Vector<float, 3>& trans = this->translateSlot.Param<core::param::Vector3fParam>()->Value();
    const vislib::math::Vector<float, 4>& rot = this->rotateSlot.Param<core::param::Vector4fParam>()->Value();
    const float& scale = this->scaleSlot.Param<core::param::FloatParam>()->Value();
    vislib::math::Matrix<float, 3, vislib::math::COLUMN_MAJOR> rotMat;
    rotMat = vislib::math::ShallowQuaternion<const float>(rot.PeekComponents());
    float rotMatBig[16];
    rotMatBig[0] = rotMat(0, 0);
    rotMatBig[1] = rotMat(1, 0);
    rotMatBig[2] = rotMat(2, 0);
    rotMatBig[3] = 0.0f;
    rotMatBig[4] = rotMat(0, 1);
    rotMatBig[5] = rotMat(1, 1);
    rotMatBig[6] = rotMat(2, 1);
    rotMatBig[7] = 0.0f;
    rotMatBig[8] = rotMat(0, 2);
    rotMatBig[9] = rotMat(1, 2);
    rotMatBig[10] = rotMat(2, 2);
    rotMatBig[11] = 0.0f;
    rotMatBig[12] = 0.0f;
    rotMatBig[13] = 0.0f;
    rotMatBig[14] = 0.0f;
    rotMatBig[15] = 1.0f;

    ::glTranslatef(trans.X(), trans.Y(), trans.Z());
    ::glMultMatrixf(rotMatBig);
    ::glScalef(scale, scale, scale);

    if ((*outCr3d)(1)) {
        vislib::math::Point<float, 3> oc = outCr3d->AccessBoundingBoxes().WorldSpaceBBox().CalcCenter();
        ::glTranslatef(-oc.X(), -oc.Y(), -oc.Z());
    }

    *outCr3d = *inCr3d;
    bool retVal = (*outCr3d)(core::view::AbstractCallRender::FnRender);

    ::glMatrixMode(GL_MODELVIEW);
    ::glPopMatrix();

    return retVal;
}


#ifdef WITH_VRPN
/*
 * TrackerRendererTransform::vrpnTrackerCallback
 */
void VRPN_CALLBACK TrackerRendererTransform::vrpnTrackerCallback(void *ctxt, const vrpn_TRACKERCB track) {
    TrackerRendererTransform *that = static_cast<TrackerRendererTransform *>(ctxt);
    that->translateSlot.Param<core::param::Vector3fParam>()->SetValue(
        vislib::math::ShallowVector<const vrpn_float64, 3>(track.pos), true);
    that->rotateSlot.Param<core::param::Vector4fParam>()->SetValue(
        vislib::math::ShallowVector<const vrpn_float64, 4>(track.quat), true);
}
#endif /* WITH_VRPN */
