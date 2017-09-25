/*
 * glfwInst.h
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCON_GL_GLFWINST_H_INCLUDED
#define MEGAMOLCON_GL_GLFWINST_H_INCLUDED
#pragma once

#include <memory>

namespace megamol {
namespace console {
namespace gl {

    /**
     * GLFW instance control for automatic termination
     */
    class glfwInst {
    public:
        static std::shared_ptr<glfwInst> Instance();
        ~glfwInst();
        inline bool OK() const { return !error; }
    private:
        static std::weak_ptr<glfwInst> inst;
        glfwInst();
        bool error;
    };

} /* end namespace gl */
} /* end namespace console */
} /* end namespace megamol */

#endif /* MEGAMOLCON_GL_GLFWINST_H_INCLUDED */
