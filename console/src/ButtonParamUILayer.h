/*
 * ButtonParamUILayer.h
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten. All rights reserved.
 */
#pragma once

#include <chrono>
#include <map>
#include "vislib/String.h"
#include "mmcore/view/Input.h"
#include <AbstractUILayer.h>

namespace megamol {
namespace console {

	using megamol::input_events::Key;
	using megamol::input_events::KeyAction;
	using megamol::input_events::Modifiers;

    /**
     * This UI layer implements hot key for button parameter.
     */
    class ButtonParamUILayer : public megamol::input_events::AbstractUILayer {
    public:
        ButtonParamUILayer(void * coreHandle, void * viewHandle);
        virtual ~ButtonParamUILayer();

        inline void SetMaskingLayer(AbstractUILayer *layer) {
            maskingLayer = layer;
        }
        virtual bool Enabled();

        virtual bool OnKey(Key key, KeyAction action, Modifiers mods);
    private:
        void updateHotkeyList();

        void *hCore; // handle memory is owned by application
        void *hView; // handle memory is owned by Window

        size_t last_param_hash;
        std::map<std::string, vislib::TString> hotkeys;
        AbstractUILayer *maskingLayer;
    };

}
}
