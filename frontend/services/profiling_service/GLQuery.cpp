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

void GLQuery::Counter() {
#ifdef MEGAMOL_USE_OPENGL
    glQueryCounter(handle_, GL_TIMESTAMP);
#endif
}

time_point GLQuery::GetNW() {
#ifdef MEGAMOL_USE_OPENGL
    if (value_ == zero_time) {
        uint64_t val;
        glGetQueryObjectui64v(handle_, GL_QUERY_RESULT_NO_WAIT, &val);
        if (val != 0) {
            value_ = time_point{std::chrono::nanoseconds(val)};
        }
    }
#endif
    return value_;
}
} // namespace megamol::frontend_resources::performance
