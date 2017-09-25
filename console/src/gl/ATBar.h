/*
 * gl/ATBar.h
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCON_GL_ATBAR_H_INCLUDED
#define MEGAMOLCON_GL_ATBAR_H_INCLUDED
#pragma once

#ifdef HAS_ANTTWEAKBAR
#include "gl/atbInst.h"
#include "AntTweakBar.h"
#include <string>

namespace megamol {
namespace console {
namespace gl {

    /** Base class for AntTweakBar bars */
    class ATBar {
    public:
        ATBar(const char* name);
        virtual ~ATBar();

        inline const char* Name() const {
            return barName.c_str();
        }
        inline TwBar *Handle() const {
            return bar;
        }

    private:
        std::shared_ptr<atbInst> atb;
        std::string barName;
        TwBar *bar;
    };

} /* end namespace gl */
} /* end namespace console */
} /* end namespace megamol */

#endif /* HAS_ANTTWEAKBAR */
#endif /* MEGAMOLCON_GL_ATBAR_H_INCLUDED */
