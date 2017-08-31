/*
 * atbInst.h
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCON_GL_ATBINST_H_INCLUDED
#define MEGAMOLCON_GL_ATBINST_H_INCLUDED
#pragma once

#ifdef HAS_ANTTWEAKBAR
#include <memory>

namespace megamol {
namespace console {
namespace gl {

    /**
     * GLFW instance control for automatic termination
     */
    class atbInst {
    public:
        static std::shared_ptr<atbInst> Instance();
        ~atbInst();
        inline bool OK() const { return !error; }
    private:
        static std::weak_ptr<atbInst> inst;
        atbInst();
        bool error;
    };

} /* end namespace gl */
} /* end namespace console */
} /* end namespace megamol */

#endif /* HAS_ANTTWEAKBAR */
#endif /* MEGAMOLCON_GL_ATBINST_H_INCLUDED */
