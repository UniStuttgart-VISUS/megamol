#include "GLQuery.h"

#ifdef MEGAMOL_USE_OPENGL
#include "glad/gl.h"
#endif

namespace megamol::frontend_resources::performance {
GLQuery::GLQuery() {
#ifdef MEGAMOL_USE_OPENGL
    glGenQueries(1, &handle_);
#endif
}

GLQuery::~GLQuery() {
#ifdef MEGAMOL_USE_OPENGL
    glDeleteQueries(1, &handle_);
#endif
}

void GLQuery::Counter() const {
#ifdef MEGAMOL_USE_OPENGL
    glQueryCounter(handle_, GL_TIMESTAMP);
#endif
}

uint64_t GLQuery::GetNW() {
#ifdef MEGAMOL_USE_OPENGL
    if (!value_) {
        glGetQueryObjectui64v(handle_, GL_QUERY_RESULT_NO_WAIT, &value_);
    }
#endif
    return value_;
}
} // namespace megamol::frontend_resources::performance
