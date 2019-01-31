/*
 * Camera_2.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/nextgen/Camera_2.h"

using namespace megamol::core;
using namespace megamol::core::nextgen;

/*
 * Camera_2::Camera_2
 */
Camera_2::Camera_2(void) : cam_type() {}

/*
 * Camera_2::Camera_2
 */
Camera_2::Camera_2(const cam_type::minimal_state_type& rhs) : cam_type() {
    *this = rhs;
}

/*
 * Camera_2::~Camera_2
 */
Camera_2::~Camera_2(void) {
    // intentionally empty
}

/*
 * Camera_2::operator=
 */
Camera_2& Camera_2::operator=(const cam_type::minimal_state_type& rhs) {
    camera::operator=(rhs); // TODO is this correct?
    return *this;
}

/*
 * Camera_2::CalcClipping
 */
void Camera_2::CalcClipping(const vislib::math::Cuboid<float>& bbox, float border) {
    glm::vec4 front4 = this->view_vector();
    glm::vec3 front = glm::vec3(front4.x, front4.y, front4.z);
    glm::vec4 pos4 = this->position();
    glm::vec3 pos = glm::vec3(pos4.x, pos4.y, pos4.z);

    float dist, minDist, maxDist;

    dist = glm::dot(front, glm::make_vec3(bbox.GetLeftBottomBack().PeekCoordinates()) - pos);
    minDist = maxDist = dist;

    dist = glm::dot(front, glm::make_vec3(bbox.GetLeftBottomFront().PeekCoordinates()) - pos);
    if (dist < minDist) minDist = dist;
    if (dist > maxDist) maxDist = dist;

    dist = glm::dot(front, glm::make_vec3(bbox.GetLeftTopBack().PeekCoordinates()) - pos);
    if (dist < minDist) minDist = dist;
    if (dist > maxDist) maxDist = dist;

    dist = glm::dot(front, glm::make_vec3(bbox.GetLeftTopFront().PeekCoordinates()) - pos);
    if (dist < minDist) minDist = dist;
    if (dist > maxDist) maxDist = dist;

    dist = glm::dot(front, glm::make_vec3(bbox.GetRightBottomBack().PeekCoordinates()) - pos);
    if (dist < minDist) minDist = dist;
    if (dist > maxDist) maxDist = dist;

    dist = glm::dot(front, glm::make_vec3(bbox.GetRightBottomFront().PeekCoordinates()) - pos);
    if (dist < minDist) minDist = dist;
    if (dist > maxDist) maxDist = dist;

    dist = glm::dot(front, glm::make_vec3(bbox.GetRightTopBack().PeekCoordinates()) - pos);
    if (dist < minDist) minDist = dist;
    if (dist > maxDist) maxDist = dist;

    dist = glm::dot(front, glm::make_vec3(bbox.GetRightTopFront().PeekCoordinates()) - pos);
    if (dist < minDist) minDist = dist;
    if (dist > maxDist) maxDist = dist;

    minDist -= border;
    maxDist += border;

    // since the minDist is broken, we fix it here
    minDist = maxDist * 0.001f;

    if (!(std::abs(this->near_clipping_plane() - minDist) < 0.00001f) ||
        !(std::abs(this->far_clipping_plane() - maxDist) < 0.00001f)) {
        this->near_clipping_plane(minDist);
        this->far_clipping_plane(maxDist);
    }
}
