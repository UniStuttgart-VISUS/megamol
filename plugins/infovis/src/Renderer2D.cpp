#include "stdafx.h"
#include "Renderer2D.h"

#include "mmcore/CoreInstance.h"

#include "mmcore/utility/log/Log.h"
#include "vislib/graphics/gl/ShaderSource.h"

using namespace megamol;
using namespace megamol::infovis;


void Renderer2D::computeDispatchSizes(
    uint64_t numItems, GLint const localSizes[3], GLint const maxCounts[3], GLuint dispatchCounts[3]) const {
    const auto localSize = localSizes[0] * localSizes[1] * localSizes[2];
    const uint64_t needed_groups = (numItems + localSize - 1) / localSize; // round up int div
    dispatchCounts[0] = std::clamp<GLint>(needed_groups, 1, maxCounts[0]);
    dispatchCounts[1] = std::clamp<GLint>((needed_groups + dispatchCounts[0] - 1) / dispatchCounts[0], 1, maxCounts[1]);
    const auto tmp = dispatchCounts[0] * dispatchCounts[1];
    dispatchCounts[2] = std::clamp<GLint>((needed_groups + tmp - 1) / tmp, 1, maxCounts[2]);
    const uint64_t totalCounts = dispatchCounts[0] * dispatchCounts[1] * dispatchCounts[2];
    ASSERT(totalCounts * localSize >= numItems);
    ASSERT(totalCounts * localSize - numItems < localSize);
}

void Renderer2D::makeDebugLabel(GLenum identifier, GLuint name, const char* label) const {
#ifdef _DEBUG
    glObjectLabel(identifier, name, -1, label);
#endif
}
void Renderer2D::debugNotify(GLuint name, const char* message) const {
#ifdef _DEBUG
    glDebugMessageInsert(
        GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER, name, GL_DEBUG_SEVERITY_NOTIFICATION, -1, message);
#endif
}
void Renderer2D::debugPush(GLuint name, const char* groupLabel) const {
#ifdef _DEBUG
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, name, -1, groupLabel);
#endif
}
void Renderer2D::debugPop() const {
#ifdef _DEBUG
    glPopDebugGroup();
#endif
}
