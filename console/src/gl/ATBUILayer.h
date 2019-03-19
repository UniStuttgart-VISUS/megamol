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
#include <string>

namespace megamol {
namespace console {
namespace gl {

    /**
     * This UI layer implements a graphical user interface using anttweakbar
     * http://anttweakbar.sourceforge.net/doc/
     */
    class ATBUILayer : public AbstractUILayer {
    public:

        ATBUILayer(const char* wndName, void* hView, void *hCore);
        virtual ~ATBUILayer();

        virtual bool Enabled();
        void ToggleEnable();

        virtual void OnResize(int w, int h);
        virtual void OnDraw();
        virtual bool OnKey(core::view::Key key,   core::view::KeyAction action,core::view:: Modifiers mods);
        virtual bool OnChar(unsigned int charcode);
        virtual bool OnMouseMove(double x, double y);
        virtual bool OnMouseButton(core::view::MouseButton button,core::view:: MouseButtonAction action, core::view::Modifiers mods);
        virtual bool OnMouseScroll(double x, double y);

    private:
        static int nextWinID;
        std::shared_ptr<atbInst> atb;
        int atbWinID;
        int atbKeyMod;
        void* hView;
        std::shared_ptr<ATBar> winBar;
        std::shared_ptr<ATBar> paramBar;
        bool enabled;
        bool isCoreHotFixed;
        std::chrono::system_clock::time_point lastParamUpdateTime;
        std::string wndName;

        bool fwd = false, back = false, left = false, right = false;
        bool wasdHotfixed = false;
    };

} /* end namespace gl */
} /* end namespace console */
} /* end namespace megamol */

#endif /* HAS_ANTTWEAKBAR */
#endif /* MEGAMOLCON_GL_ATBUILAYER_H_INCLUDED */
