/*
 * CallCursorInput.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLCURSORINPUT_H_INCLUDED
#define MEGAMOLCORE_CALLCURSORINPUT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Call.h"
#include "CallAutoDescription.h"
#include "api/MegaMolCore.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * Base class of rendering graph calls
     */
    class CallCursorInput : public Call {
    public:

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "CallCursorInput";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call of mouse cursor input";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return 4;
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            switch (idx) {
                case 0: return "SetCursor2DButtonState";
                case 1: return "SetCursor2DPosition";
                case 2: return "SetInputModifier";
                case 4: return "ResetView";
                default: return NULL;
            }
        }

        /** Ctor. */
        CallCursorInput(void);

        /** Dtor. */
        virtual ~CallCursorInput(void);

        /**
         * Gets or sets the button.
         *
         * @return The button
         */
        inline unsigned int& Btn(void) {
            return this->btn;
        }

        /**
         * Gets or sets the 'down' flag.
         *
         * @return The 'down' flag
         */
        inline bool& Down(void) {
            return this->down;
        }

        /**
         * Gets or sets the x coordinate.
         *
         * @return The x coordinate
         */
        inline float& X(void) {
            return this->x;
        }

        /**
         * Gets or sets the y coordinate.
         *
         * @return The y coordinate
         */
        inline float& Y(void) {
            return this->y;
        }

        /**
         * Gets or sets the input modifier
         *
         * @return The input modifier
         */
        inline mmcInputModifier& Mod(void) {
            return this->mod;
        }

    private:

        /** The button */
        unsigned int btn;

        /**
         * Flag whether the button is pressed, or not, or the new input
         * modifier state
         */
        bool down;

        /** The x coordinate */
        float x;

        /** The y coordinate */
        float y;

        /** The input modifier to be set */
        mmcInputModifier mod;

    };


    /** Description class typedef */
    typedef CallAutoDescription<CallCursorInput> CallCursorInputDescription;


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLCURSORINPUT_H_INCLUDED */
