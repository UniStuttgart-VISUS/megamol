/*
 * AnaglyphStereoDisplay.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ANAGLYPHSTEREODISPLAY_H_INCLUDED
#define MEGAMOLCORE_ANAGLYPHSTEREODISPLAY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "param/ParamSlot.h"
#include "special/AbstractStereoDisplay.h"


namespace megamol {
namespace core {
namespace special {

    /**
     * Special view module used for column based stereo displays
     */
    class AnaglyphStereoDisplay : public AbstractStereoDisplay {
    public:

        /**
         * Gets the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "AnaglyphStereoDisplay";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Special view module used for anaglyph stereo rendering";
        }

        /**
         * Gets whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor. */
        AnaglyphStereoDisplay(void);

        /** Dtor. */
        virtual ~AnaglyphStereoDisplay(void);

        /**
         * Renders this AbstractView3D in the currently active OpenGL context.
         */
        virtual void Render(void);

    private:

        /**
         * Implementation of 'Module::Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'Module::Release'.
         */
        virtual void release(void);

        /** The color separation mode */
        param::ParamSlot colorModeSlot;

    };


} /* end namespace special */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ANAGLYPHSTEREODISPLAY_H_INCLUDED */
