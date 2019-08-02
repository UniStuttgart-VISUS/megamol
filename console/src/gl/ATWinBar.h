/*
 * gl/ATWinBar.h
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCON_GL_ATWINBAR_H_INCLUDED
#define MEGAMOLCON_GL_ATWINBAR_H_INCLUDED
#pragma once

#ifdef HAS_ANTTWEAKBAR
#include "gl/ATBar.h"
#include <array>
#include <tuple>

namespace megamol {
namespace console {
namespace gl {

    /* forward declarations */
    class Window;
    class ATBUILayer;

    /** Base class for AntTweakBar bars */
    class ATWinBar : public ATBar {
    public:
        ATWinBar(ATBUILayer& layer, const char* wndName);
        virtual ~ATWinBar();

    private:
        static void getWndValues(ATWinBar *inst);
        static void setWndValues(ATWinBar *inst);
        static void setWinSizePreset(std::tuple<ATWinBar*, unsigned int, unsigned int> *preset);
        static void toggleGUI(ATWinBar *inst);
        static void setShowFPSinWindowTitle(const void *value, Window *wnd);
        static void getShowFPSinWindowTitle(void *value, Window *wnd);
        static void setShowSamplesinWindowTitle(const void *value, Window *wnd);
        static void getShowSamplesinWindowTitle(void *value, Window *wnd);
        static void setShowPrimsinWindowTitle(const void *value, Window *wnd);
        static void getShowPrimsinWindowTitle(void *value, Window *wnd);
        static void copyFPS(Window *wnd);
        static void copyFPSList(Window *wnd);
        static void setParamFilePath(const void *value, void *ctxt);
        static void getParamFilePath(void *value, void *ctxt);
        static void loadParamFile(void *ctxt);
        static void saveParamFile(void *ctxt);

        //Window& wnd;
        ATBUILayer& layer;

		float fps = -0.0f; // TODO: broken but needed to kick gl::Window reference in class
        int winX, winY;
        unsigned int winW, winH;
        std::array<std::tuple<ATWinBar*, unsigned int, unsigned int>, 5> winSizePresets;

    };

} /* end namespace gl */
} /* end namespace console */
} /* end namespace megamol */

#endif /* HAS_ANTTWEAKBAR */
#endif /* MEGAMOLCON_GL_ATWINBAR_H_INCLUDED */
