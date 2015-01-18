/*
 * CamParamObserver.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_CAMPARAMOBSERVER_H_INCLUDED
#define VISLIBTEST_CAMPARAMOBSERVER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include <iostream>

#include "vislib/CameraParameterObserver.h"


class CamParamObserver : public vislib::graphics::CameraParameterObserver {

public:

    CamParamObserver(void);

    virtual ~CamParamObserver(void);

    virtual void OnApertureAngleChanged(
        const vislib::math::AngleDeg newValue);

    virtual void OnEyeChanged(
        const vislib::graphics::CameraParameters::StereoEye newValue);

    virtual void OnFarClipChanged(
        const vislib::graphics::SceneSpaceType newValue);

    virtual void OnFocalDistanceChanged(
        const vislib::graphics::SceneSpaceType newValue);

    virtual void OnLookAtChanged(
        const vislib::graphics::SceneSpacePoint3D& newValue);

    virtual void OnNearClipChanged(
        const vislib::graphics::SceneSpaceType newValue);

    virtual void OnPositionChanged(
        const vislib::graphics::SceneSpacePoint3D& newValue);

    virtual void OnProjectionChanged(
        const vislib::graphics::CameraParameters::ProjectionType newValue);

    virtual void OnStereoDisparityChanged(
        const vislib::graphics::SceneSpaceType newValue);

    virtual void OnTileRectChanged(
        const vislib::graphics::ImageSpaceRectangle& newValue);

    virtual void OnUpChanged(
        const vislib::graphics::SceneSpaceVector3D& newValue);

    virtual void OnVirtualViewSizeChanged(
        const vislib::graphics::ImageSpaceDimension& newValue);

protected:

    std::ostream& dump(std::ostream& out, 
        const vislib::graphics::SceneSpaceVector3D& obj);

    std::ostream& dump(std::ostream& out, 
        const vislib::graphics::SceneSpacePoint3D& obj);

    std::ostream& dump(std::ostream& out, 
        const vislib::graphics::ImageSpaceRectangle& obj);

    std::ostream& dump(std::ostream& out, 
        const vislib::graphics::ImageSpaceDimension& obj);

    std::ostream& dump(std::ostream& out, 
        const vislib::graphics::CameraParameters::StereoEye& obj);

    std::ostream& dump(std::ostream& out, 
        const vislib::graphics::CameraParameters::ProjectionType& obj);
};

#endif /* VISLIBTEST_CAMPARAMOBSERVER_H_INCLUDED */
