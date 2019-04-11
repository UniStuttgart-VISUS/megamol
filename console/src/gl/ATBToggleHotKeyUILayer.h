/*
 * gl/ATBToggleHotKeyUILayer.h
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCON_GL_ATBTOGGLEHOTKEYUILAYER_H_INCLUDED
#define MEGAMOLCON_GL_ATBTOGGLEHOTKEYUILAYER_H_INCLUDED
#pragma once

#ifdef HAS_ANTTWEAKBAR
#include "gl/ATBUILayer.h"

namespace megamol {
namespace console {
namespace gl {

    /**
     * This UI layer complements the ATBUILayer
     * The only function is to react on F12 to toggle the deactivated gui
     */
    class ATBToggleHotKeyUILayer : public AbstractUILayer {
    public:

        ATBToggleHotKeyUILayer(ATBUILayer& atbLayer);
        virtual ~ATBToggleHotKeyUILayer();

        virtual bool Enabled();

        virtual bool OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods);

    private:
        ATBUILayer& atbLayer;
    };

} /* end namespace gl */
} /* end namespace console */
} /* end namespace megamol */

#endif /* HAS_ANTTWEAKBAR */
#endif /* MEGAMOLCON_GL_ATBTOGGLEHOTKEYUILAYER_H_INCLUDED */
