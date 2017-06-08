/*
 * gl/ATBUILayer.h
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCON_GL_ATBUILAYER_H_INCLUDED
#define MEGAMOLCON_GL_ATBUILAYER_H_INCLUDED
#pragma once

#ifdef HAS_ANTTWEAKBAR
#include "AbstractUILayer.h"
#include "gl/atbInst.h"
#include "gl/ATBar.h"
#include "AntTweakBar.h"
#include <chrono>

namespace megamol {
namespace console {
namespace gl {

    /**
     * This UI layer implements a graphical user interface using anttweakbar
     * http://anttweakbar.sourceforge.net/doc/
     */
    class ATBUILayer : public AbstractUILayer {
    public:

        ATBUILayer(Window& wnd, const char* wndName, void* hView, void *hCore);
        virtual ~ATBUILayer();

        virtual bool Enabled();
        void ToggleEnable();

        virtual void onResize(int w, int h);
        virtual void onDraw();
        virtual bool onKey(Key key, int scancode, KeyAction action, Modifiers mods);
        virtual bool onChar(unsigned int charcode);
        virtual bool onMouseMove(double x, double y);
        virtual bool onMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods);
        virtual bool onMouseWheel(double x, double y);

    private:
        static int nextWinID;
        std::shared_ptr<atbInst> atb;
        int atbWinID;
        int atbKeyMod;
        void* hView;
        std::shared_ptr<ATBar> winBar;
        std::shared_ptr<ATBar> paramBar;
        bool enabled;
        std::chrono::system_clock::time_point lastParamUpdateTime;
    };

} /* end namespace gl */
} /* end namespace console */
} /* end namespace megamol */

#endif /* HAS_ANTTWEAKBAR */
#endif /* MEGAMOLCON_GL_ATBUILAYER_H_INCLUDED */
