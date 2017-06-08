/*
 * CamParamObserver.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "CamParamObserver.h"


/*
 * CamParamObserver::CamParamObserver
 */
CamParamObserver::CamParamObserver(void) {
}


/*
 * CamParamObserver::~CamParamObserver
 */
CamParamObserver::~CamParamObserver(void){
}


/*
 * CamParamObserver::OnApertureAngleChanged
 */
void CamParamObserver::OnApertureAngleChanged(
        const vislib::math::AngleDeg newValue) {
    std::cout << "Aperture Angle: " << newValue << std::endl;
}


/*
 * CamParamObserver::OnEyeChanged
 */
void CamParamObserver::OnEyeChanged(
        const vislib::graphics::CameraParameters::StereoEye newValue) {
    std::cout << "Stereo Eye: ";
    this->dump(std::cout, newValue) << std::endl;
}


/*
 * CamParamObserver::OnFarClipChanged
 */
void CamParamObserver::OnFarClipChanged(
        const vislib::graphics::SceneSpaceType newValue) {
    std::cout << "Far Clipping Plane: " << newValue << std::endl;
}


/*
 * CamParamObserver::OnFocalDistanceChanged
 */
void CamParamObserver::OnFocalDistanceChanged(
        const vislib::graphics::SceneSpaceType newValue) {
    std::cout << "Focal Distance: " << newValue << std::endl;
}


/*
 * CamParamObserver::OnLookAtChanged
 */
void CamParamObserver::OnLookAtChanged(
        const vislib::graphics::SceneSpacePoint3D& newValue) {
    std::cout << "Look-At Point: ";
    this->dump(std::cout, newValue) << std::endl;
}


/*
 * CamParamObserver::OnNearClipChanged
 */
void CamParamObserver::OnNearClipChanged(
        const vislib::graphics::SceneSpaceType newValue) {
    std::cout << "Near Clipping Plane: " << newValue << std::endl;
}


/*
 * CamParamObserver::OnPositionChanged
 */
void CamParamObserver::OnPositionChanged(
        const vislib::graphics::SceneSpacePoint3D& newValue) {
    std::cout << "Camera Position: ";
    this->dump(std::cout, newValue) << std::endl;
}


/*
 * CamParamObserver::OnProjectionChanged
 */
void CamParamObserver::OnProjectionChanged(
        const vislib::graphics::CameraParameters::ProjectionType newValue) {
    std::cout << "Projection Type: ";
    this->dump(std::cout, newValue) << std::endl;
}


/*
 * CamParamObserver::OnStereoDisparityChanged
 */
void CamParamObserver::OnStereoDisparityChanged(
        const vislib::graphics::SceneSpaceType newValue) {
    std::cout << "Stereo Disparity: " << newValue << std::endl;
}


/*
 * CamParamObserver::OnTileRectChanged
 */
void CamParamObserver::OnTileRectChanged(
        const vislib::graphics::ImageSpaceRectangle& newValue) {
    std::cout << "Tile: ";
    this->dump(std::cout, newValue) << std::endl;
}


/*
 * CamParamObserver::OnUpChanged
 */
void CamParamObserver::OnUpChanged(
        const vislib::graphics::SceneSpaceVector3D& newValue) {
    std::cout << "Camera Up-Vector: ";
    this->dump(std::cout, newValue) << std::endl;
}


/*
 * CamParamObserver::OnVirtualViewSizeChanged
 */
void CamParamObserver::OnVirtualViewSizeChanged(
        const vislib::graphics::ImageSpaceDimension& newValue) {
    std::cout << "Virtual Screen: ";
    this->dump(std::cout, newValue) << std::endl;
}


/*
 * CamParamObserver::dump
 */
std::ostream& CamParamObserver::dump(std::ostream& out, 
        const vislib::graphics::SceneSpaceVector3D& obj) {
    out << "(" << obj.X() << ", " << obj.Y() << ", " << obj.Z() << ")";
    return out;
}


/*
 * CamParamObserver::dump
 */
std::ostream& CamParamObserver::dump(std::ostream& out, 
        const vislib::graphics::SceneSpacePoint3D& obj) {
    out << "(" << obj.X() << ", " << obj.Y() << ", " << obj.Z() << ")";
    return out;
}


/*
 * CamParamObserver::dump
 */
std::ostream& CamParamObserver::dump(std::ostream& out, 
        const vislib::graphics::ImageSpaceRectangle& obj) {
    out << "((" << obj.Left() << ", " << obj.Bottom() << "), (" 
        << obj.Right() << ", " << obj.Top() << "))";
    return out;
}


/*
 * CamParamObserver::dump
 */
std::ostream& CamParamObserver::dump(std::ostream& out, 
        const vislib::graphics::ImageSpaceDimension& obj) {
    out << "[" << obj.Width() << ", " << obj.Height() << "]";
    return out;
}


/*
 * CamParamObserver::dump
 */
std::ostream& CamParamObserver::dump(std::ostream& out, 
        const vislib::graphics::CameraParameters::StereoEye& obj) {
    using namespace vislib::graphics;
    out << ((obj == CameraParameters::LEFT_EYE) ? "LEFT_EYE" : "RIGHT_EYE");
    return out;
}


/*
 * CamParamObserver::dump
 */
std::ostream& CamParamObserver::dump(std::ostream& out, 
        const vislib::graphics::CameraParameters::ProjectionType& obj) {
    using namespace vislib::graphics;

    switch (obj) {
        case CameraParameters::MONO_PERSPECTIVE: 
            out << "MONO_PERSPECTIVE"; 
            break;

        case CameraParameters::MONO_ORTHOGRAPHIC: 
            out << "MONO_ORTHOGRAPHIC"; 
            break;

        case CameraParameters::STEREO_PARALLEL:
            out << "STEREO_PARALLEL"; 
            break;

        case CameraParameters::STEREO_OFF_AXIS: 
            out << "STEREO_OFF_AXIS"; 
            break;

        case CameraParameters::STEREO_TOE_IN: 
            out << "STEREO_TOE_IN"; 
            break;

        default:
            out << "<???"">"; 
            break;
    }

    return out;
}
