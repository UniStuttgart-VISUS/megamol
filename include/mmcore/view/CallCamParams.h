//
// CallCamParams.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//

#ifndef MEGAMOLCORE_CALLCAMPARAMS_H_INCLUDED
#define MEGAMOLCORE_CALLCAMPARAMS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "vislib/graphics/CameraParameters.h"
#include "mmcore/CallAutoDescription.h"
#include "vislib/SmartPtr.h"

namespace megamol {
namespace core {
namespace view {

class MEGAMOLCORE_API CallCamParams : public core::Call {

public:

    /// Index of the 'GetCamparams' function
    static const unsigned int CallForGetCamParams;

    /// Index of the 'SetCamparams' function
    static const unsigned int CallForSetCamParams;

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char *ClassName(void) {
        return "CallCamParams";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char *Description(void) {
        return "Call to transmit camera parameters";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
        return 2;
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char * FunctionName(unsigned int idx) {
        switch( idx) {
        case 0:
            return "getCamParams";
        case 1:
            return "setCamParams";
        }
        return "";
    }

    /** Ctor. */
    CallCamParams(void);

    /** Dtor. */
    virtual ~CallCamParams(void);

    /**
     * Copies all values from 'src' into this object.
     *
     * @param src The source object to copy from.
     */
    inline void CopyFrom(const vislib::SmartPtr<vislib::graphics::CameraParameters> src) {
        this->CopyFrom(src.operator->());
    }

    /**
     * Copies all values from 'src' into this object.
     *
     * @param src The source object to copy from.
     */
    inline void CopyFrom(const vislib::graphics::CameraParameters *src) {
//        this->camParams->SetApertureAngle(src->ApertureAngle());
//        this->camParams->SetAutoFocusOffset(src->AutoFocusOffset());
//        this->camParams->SetClip(src->NearClip(), src->FarClip());
//        this->camParams->SetCoordSystemType(src->CoordSystemType());
//        this->camParams->SetProjection(src->Projection());
//        this->camParams->SetStereoParameters(src->StereoDisparity(), src->Eye(), src->FocalDistance());
        this->camParams->SetView(src->Position(), src->LookAt(), src->Up());
//        this->camParams->SetVirtualViewSize(src->VirtualViewSize());
//        this->camParams->SetTileRect(src->TileRect()); // set after virtual view size
//        this->camParams->SetLimits(src->Limits()); // set as last
    }

    /**
     * Gets the camera parameters pointer.
     *
     * @return The camera parameters pointer.
     */
    inline const vislib::SmartPtr<vislib::graphics::CameraParameters>& GetCamParams(void) const {
        return this->camParams;
    }

    /**
     * Sets the camera parameters pointer.
     *
     * @param camParams The new value for the camera parameters pointer.
     */
    inline void SetCameraParameters(const vislib::SmartPtr<
            vislib::graphics::CameraParameters>& camParams) {
        this->camParams = camParams;
    }


protected:


private:

#ifdef _WIN32
#pragma warning(disable:4251)
#endif /* _WIN32 */
    /// The camera parameters
    vislib::SmartPtr<vislib::graphics::CameraParameters> camParams;
#ifdef _WIN32
#pragma warning(default:4251)
#endif /* _WIN32 */

};

/// Description class typedef
typedef core::CallAutoDescription<CallCamParams> CallCamParamsDescription;


} // end namespace view
} // end namespace core
} // end namespace megamol

#endif // MEGAMOLCORE_CALLCAMPARAMS_H_INCLUDED
