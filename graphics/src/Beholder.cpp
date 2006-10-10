/*
 * Beholder.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/Beholder.h"


/*
 * vislib::graphics::Beholder::Beholder
 */
vislib::graphics::Beholder::Beholder() : position(), 
        lookAt(0, 0, -1), 
        front(), right(), // front and right are calculated based on position, lookAt and up
        up(0, 1, 0),
        updateCounter(0) {
    this->CalcOrthoNormalVectors();
};


/*
 * vislib::graphics::Beholder::Beholder
 */
vislib::graphics::Beholder::Beholder(const Beholder &rhs) {
    this->operator=(rhs);
}


/*
 * vislib::graphics::Beholder::operator=
 */
vislib::graphics::Beholder& vislib::graphics::Beholder::operator=(const Beholder &rhs) {
    this->position = rhs.position;
    this->lookAt = rhs.lookAt;
    this->up = rhs.up;
    this->front = rhs.front;
    this->right = rhs.right;
    this->updateCounter = 0;
    return *this;
}


/*
 * vislib::graphics::Beholder::CalcOrthoNormalVectors
 */
void vislib::graphics::Beholder::CalcOrthoNormalVectors(void) {
    this->front = this->lookAt - this->position;
    this->front.Normalise();
    this->right = this->front.Cross(this->up);
    this->right.Normalise();
    this->up = this->right.Cross(this->front);
    this->up.Normalise();
}
