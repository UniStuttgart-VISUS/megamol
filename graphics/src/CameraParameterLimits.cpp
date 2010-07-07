/*
 * CameraParameterLimits.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */


#include "vislib/CameraParameterLimits.h"
#include "vislib/Float16.h"
#include "vislib/mathfunctions.h"
#include "vislib/vislibsymbolimportexport.inl"


/*
 * __vl_CameraParameterLimits_defaultlimits
 */
VISLIB_STATICSYMBOL vislib::SmartPtr<vislib::graphics::CameraParameterLimits>
    __vl_CameraParameterLimits_defaultlimits
#ifndef VISLIB_SYMBOL_IMPORT
        (new vislib::graphics::CameraParameterLimits())
#endif /* !VISLIB_SYMBOL_IMPORT */
        ;


/* Default values */
#define DEFAULT_MAX_HALF_APERTURE_ANGLE     vislib::math::AngleDeg2Rad(85.0)
#define DEFAULT_MIN_CLIP_PLANE_DIST         float(vislib::math::Float16::MIN)
#define DEFAULT_MIN_FOCAL_DIST              float(vislib::math::Float16::MIN)
#define DEFAULT_MIN_HALF_APERTURE_ANGLE     vislib::math::AngleDeg2Rad(5.0)
#define DEFAULT_MIN_LOOK_AT_DIST            float(vislib::math::Float16::MIN)
#define DEFAULT_MIN_NEAR_CLIP_DIST          float(vislib::math::Float16::MIN)


/*
 * vislib::graphics::CameraParameterLimits::DefaultLimits
 */
vislib::SmartPtr<vislib::graphics::CameraParameterLimits>& 
vislib::graphics::CameraParameterLimits::DefaultLimits(void) {
    return __vl_CameraParameterLimits_defaultlimits;
}


/*
 * vislib::graphics::CameraParameterLimits::CameraParameterLimits
 */
vislib::graphics::CameraParameterLimits::CameraParameterLimits(void)
        : maxHalfApertureAngle(DEFAULT_MAX_HALF_APERTURE_ANGLE), 
        minClipPlaneDist(DEFAULT_MIN_CLIP_PLANE_DIST), 
        minFocalDist(DEFAULT_MIN_FOCAL_DIST), 
        minHalfApertureAngle(DEFAULT_MIN_HALF_APERTURE_ANGLE), 
        minLookAtDist(DEFAULT_MIN_LOOK_AT_DIST), 
        minNearClipDist(DEFAULT_MIN_NEAR_CLIP_DIST) {
}


/*
 * vislib::graphics::CameraParameterLimits::CameraParameterLimits
 */
vislib::graphics::CameraParameterLimits::CameraParameterLimits(
        const vislib::graphics::CameraParameterLimits& rhs) {
    *this = rhs;
}


/*
 * vislib::graphics::CameraParameterLimits::~CameraParameterLimits
 */
vislib::graphics::CameraParameterLimits::~CameraParameterLimits(void) {
}


/*
 * vislib::graphics::CameraParameterLimits::Deserialise
 */
void vislib::graphics::CameraParameterLimits::Deserialise(
        Serialiser& serialiser) {
    float f;

    serialiser.Deserialise(f, "maxHalfApertureAngle");
    this->maxHalfApertureAngle = f;

    serialiser.Deserialise(f, "minClipPlaneDist");
    this->minClipPlaneDist = f;

    serialiser.Deserialise(f, "minFocalDist");
    this->minFocalDist = f;

    serialiser.Deserialise(f, "minHalfApertureAngle");
    this->minHalfApertureAngle = f;

    serialiser.Deserialise(f, "minLookAtDist");
    this->minLookAtDist = f;

    serialiser.Deserialise(f, "minNearClipDist");
    this->minNearClipDist = f;
}


/*
 * vislib::graphics::CameraParameterLimits::LimitApertureAngle
 */
bool vislib::graphics::CameraParameterLimits::LimitApertureAngle(
        vislib::math::AngleRad minValue, vislib::math::AngleRad maxValue) {
    if ((minValue <= 0.0f) || (minValue >= float(vislib::math::PI_DOUBLE)) 
        || (maxValue <= 0.0f) || (maxValue >= float(vislib::math::PI_DOUBLE)) 
        || (minValue > maxValue)) {
        return false;
    }

    this->maxHalfApertureAngle = maxValue * 0.5f;
    this->minHalfApertureAngle = minValue * 0.5f;

    return true;
}


/*
 * vislib::graphics::CameraParameterLimits::LimitClippingDistances
 */
bool vislib::graphics::CameraParameterLimits::LimitClippingDistances(
        vislib::graphics::SceneSpaceType minNearDist, 
        vislib::graphics::SceneSpaceType minClipDist) {
    if (minClipDist <= 0.0f) {
        return false;
    }

    this->minNearClipDist = minNearDist;
    this->minClipPlaneDist = minClipDist;

    return true;
}


/*
 * vislib::graphics::CameraParameterLimits::LimitFocalDistance
 */
bool vislib::graphics::CameraParameterLimits::LimitFocalDistance(
        vislib::graphics::SceneSpaceType minFocalDist) {

    this->minFocalDist = minFocalDist;

    return true;
}


/*
 * vislib::graphics::CameraParameterLimits::LimitLootAtDistance
 */
bool vislib::graphics::CameraParameterLimits::LimitLootAtDistance(
        vislib::graphics::SceneSpaceType minLookAtDist) {
    if (minLookAtDist <= 0.0f) {
        return false;
    }

    this->minLookAtDist = minLookAtDist;

    return true;
}


/*
 * vislib::graphics::CameraParameterLimits::Reset
 */
void vislib::graphics::CameraParameterLimits::Reset(void) {
    this->maxHalfApertureAngle = DEFAULT_MAX_HALF_APERTURE_ANGLE;
    this->minClipPlaneDist = DEFAULT_MIN_CLIP_PLANE_DIST;
    this->minFocalDist = DEFAULT_MIN_FOCAL_DIST;
    this->minHalfApertureAngle = DEFAULT_MIN_HALF_APERTURE_ANGLE;
    this->minLookAtDist = DEFAULT_MIN_LOOK_AT_DIST;
    this->minNearClipDist = DEFAULT_MIN_NEAR_CLIP_DIST;
}


/*
 * vislib::graphics::CameraParameterLimits::Serialise
 */
void vislib::graphics::CameraParameterLimits::Serialise(
        Serialiser& serialiser) const {

    serialiser.Serialise(static_cast<float>(this->maxHalfApertureAngle),
        "maxHalfApertureAngle");
    serialiser.Serialise(static_cast<float>(this->minClipPlaneDist),
        "minClipPlaneDist");
    serialiser.Serialise(static_cast<float>(this->minFocalDist),
        "minFocalDist");
    serialiser.Serialise(static_cast<float>(this->minHalfApertureAngle),
        "minHalfApertureAngle");
    serialiser.Serialise(static_cast<float>(this->minLookAtDist),
        "minLookAtDist");
    serialiser.Serialise(static_cast<float>(this->minNearClipDist),
        "minNearClipDist");
}


/*
 * vislib::graphics::CameraParameterLimits::operator=
 */
vislib::graphics::CameraParameterLimits& 
vislib::graphics::CameraParameterLimits::operator=(
        const vislib::graphics::CameraParameterLimits& rhs) {
    this->maxHalfApertureAngle = rhs.maxHalfApertureAngle;
    this->minClipPlaneDist = rhs.minClipPlaneDist;
    this->minFocalDist = rhs.minFocalDist;
    this->minHalfApertureAngle = rhs.minHalfApertureAngle;
    this->minLookAtDist = rhs.minLookAtDist;
    this->minNearClipDist = rhs.minNearClipDist;
    return *this;
}


/*
 * vislib::graphics::CameraParameterLimits::operator==
 */
bool vislib::graphics::CameraParameterLimits::operator==(
        const vislib::graphics::CameraParameterLimits& rhs) const {
    return ((this->maxHalfApertureAngle == rhs.maxHalfApertureAngle)
        && (this->minClipPlaneDist == rhs.minClipPlaneDist)
        && (this->minFocalDist == rhs.minFocalDist)
        && (this->minHalfApertureAngle == rhs.minHalfApertureAngle)
        && (this->minLookAtDist == rhs.minLookAtDist)
        && (this->minNearClipDist == rhs.minNearClipDist));
}
