/*
 * CameraParamsOverride.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */

#include "stdafx.h"
#include "CameraParamsOverride.h"



/*
 *  megamol::cinematiccamera::CameraParamsOverride::~CameraParamsOverride
 */
megamol::cinematiccamera::CameraParamsOverride::~CameraParamsOverride(void) { }

const megamol::cinematiccamera::CameraParamsOverride::Point& megamol::cinematiccamera::CameraParamsOverride::LookAt(void) const {
	return this->lookAt;
}

const megamol::cinematiccamera::CameraParamsOverride::Point& megamol::cinematiccamera::CameraParamsOverride::Position(void) const {
	return this->position;
}


void megamol::cinematiccamera::CameraParamsOverride::SetApertureAngle(vislib::math::AngleDeg apertureAngle){
	this->halfApertureAngle = vislib::math::AngleDeg2Rad(apertureAngle);
	this->halfApertureAngle /= 2;
}


vislib::math::AngleRad megamol::cinematiccamera::CameraParamsOverride::HalfApertureAngle(void) const{
	//return this->paramsBase()->HalfApertureAngle();
	//auto x = this->paramsBase()->HalfApertureAngle();
	return this->halfApertureAngle;
}

void megamol::cinematiccamera::CameraParamsOverride::SetView(
		const Point& position, const Point& lookAt, const Vector& up) {
	this->lookAt = lookAt;
	this->position = position;
	this->up = up;
	this->indicateValueChange();
}


void megamol::cinematiccamera::CameraParamsOverride::SetPosition(const Point& position) {
	this->position = position;
	this->indicateValueChange();
}

void megamol::cinematiccamera::CameraParamsOverride::SetLookAt(const Point& lookAt) {
	this->lookAt = lookAt;
	this->indicateValueChange();
}

const megamol::cinematiccamera::CameraParamsOverride::Vector& megamol::cinematiccamera::CameraParamsOverride::Up(void) const {
	return this->up;
}


megamol::cinematiccamera::CameraParamsOverride&
megamol::cinematiccamera::CameraParamsOverride::operator=(
        const CameraParamsOverride& rhs) {
    Base::operator=(rhs);
	this->SetView(rhs.position, rhs.lookAt, rhs.up);
	this->halfApertureAngle = rhs.halfApertureAngle;
    return *this;
}


/*
 *  megamol::cinematiccamera::CameraParamsOverride::operator==
 */
bool megamol::cinematiccamera::CameraParamsOverride::operator==(
        const CameraParamsOverride& rhs) const {
    return (Base::operator==(rhs)
		&& (this->lookAt == rhs.lookAt)
		&& (this->position == rhs.position)
		&& (this->up == rhs.up)
		&& (this->halfApertureAngle == rhs.halfApertureAngle));
}


void megamol::cinematiccamera::CameraParamsOverride::preBaseSet(const vislib::SmartPtr<CameraParameters>& params) {}


/*
 *  megamol::cinematiccamera::CameraParamsOverride::resetOverride
 */
void megamol::cinematiccamera::CameraParamsOverride::resetOverride(void) {
    this->lookAt = this->paramsBase()->LookAt();
	this->position = this->paramsBase()->Position();
	this->up = this->paramsBase()->Up();
	this->halfApertureAngle = this->paramsBase()->HalfApertureAngle();
	
    this->indicateValueChange();
}
