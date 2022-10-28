/**
 * MegaMol
 * Copyright (c) 2018, MegaMol Dev Team
 * All rights reserved.
 */

#include "Renderer2D.h"

#include "mmcore/utility/log/Log.h"

using namespace megamol;
using namespace megamol::infovis_gl;

void Renderer2D::computeDispatchSizes(
    uint64_t numItems, GLint const localSizes[3], GLint const maxCounts[3], GLuint dispatchCounts[3]) {
    if (numItems > 0) {
        const auto localSize = localSizes[0] * localSizes[1] * localSizes[2];
        const uint64_t needed_groups = (numItems + localSize - 1) / localSize; // round up int div
        dispatchCounts[0] = std::clamp<GLint>(needed_groups, 1, maxCounts[0]);
        dispatchCounts[1] =
            std::clamp<GLint>((needed_groups + dispatchCounts[0] - 1) / dispatchCounts[0], 1, maxCounts[1]);
        const auto tmp = dispatchCounts[0] * dispatchCounts[1];
        dispatchCounts[2] = std::clamp<GLint>((needed_groups + tmp - 1) / tmp, 1, maxCounts[2]);
        const uint64_t totalCounts = dispatchCounts[0] * dispatchCounts[1] * dispatchCounts[2];
        ASSERT(totalCounts * localSize >= numItems);
        ASSERT(totalCounts * localSize - numItems < localSize);
    } else {
        dispatchCounts[0] = 0;
        dispatchCounts[1] = 0;
        dispatchCounts[2] = 0;
    }
}

std::tuple<double, double> Renderer2D::mouseCoordsToWorld(
    double mouse_x, double mouse_y, const core::view::Camera& cam, int width, int height) {

    auto cam_pose = cam.get<core::view::Camera::Pose>();
    auto cam_intrinsics = cam.get<core::view::Camera::OrthographicParameters>();
    double world_x = ((mouse_x * 2.0 / static_cast<double>(width)) - 1.0);
    double world_y = 1.0 - (mouse_y * 2.0 / static_cast<double>(height));
    world_x = world_x * 0.5 * cam_intrinsics.frustrum_height * cam_intrinsics.aspect + cam_pose.position.x;
    world_y = world_y * 0.5 * cam_intrinsics.frustrum_height + cam_pose.position.y;

    return std::make_tuple(world_x, world_y);
}

void Renderer2D::makeDebugLabel(GLenum identifier, GLuint name, const char* label) {
#ifdef _DEBUG
    glObjectLabel(identifier, name, -1, label);
#endif
}
void Renderer2D::debugNotify(GLuint name, const char* message) {
#ifdef _DEBUG
    glDebugMessageInsert(
        GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER, name, GL_DEBUG_SEVERITY_NOTIFICATION, -1, message);
#endif
}
void Renderer2D::debugPush(GLuint name, const char* groupLabel) {
#ifdef _DEBUG
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, name, -1, groupLabel);
#endif
}
void Renderer2D::debugPop() {
#ifdef _DEBUG
    glPopDebugGroup();
#endif
}
