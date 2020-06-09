/*
 * utility/HotFixFileName.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once
#include <AbstractUILayer.h>

namespace megamol {
namespace console {
namespace utility {

	using namespace megamol::input_events;

    /** Utility class closing a window when ESC is pressed */
    class HotFixFileName : public AbstractUILayer {
    public:
        HotFixFileName(void* hCore);
        virtual ~HotFixFileName();

        virtual bool OnKey(Key key, KeyAction action, Modifiers mods);
    private:
        void* hCore;
    };

}
}
}

