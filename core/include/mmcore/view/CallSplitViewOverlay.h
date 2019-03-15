/*
 * CallSplitViewOverlay.h
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLSPLITVIEWOVERLAY_H_INCLUDED
#define MEGAMOLCORE_CALLSPLITVIEWOVERLAY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/view/InputCall.h"
#include "mmcore/factories/CallAutoDescription.h"

#include "vislib/math/Rectangle.h"


namespace megamol {
namespace core {
namespace view {

    /**
     * Call connecting SplitView with special overlay renderer (e.g. gui::GUIRenderer)
     * 
     * (Provides only currently available viewport and instance time.)
     *
     */
    class MEGAMOLCORE_API CallSplitViewOverlay : public InputCall {
    public:

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "CallSplitViewOverlay";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call the special overlay renderer";
        }

		/** Function index of 'render' */
        static const unsigned int FnOverlay = 5;

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            ASSERT(FnOverlay == InputCall::FunctionCount() && "Enum has bad magic number");
            return InputCall::FunctionCount() + 1;
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
		static const char* FunctionName(unsigned int idx) {
            #define CaseFunction(id) case Fn##id: return #id
            switch (idx) {
                CaseFunction(Overlay);
            default: return InputCall::FunctionName(idx);
            }
            #undef CaseFunction
		}

        /**
         * Ctor.
         */
        CallSplitViewOverlay(void);

        /**
         * ~Dtor.
         */
        virtual ~CallSplitViewOverlay(void);


        /**
         * Answer the viewport to be used.
         *
         * @return The viewport to be used
         */
        const vislib::math::Rectangle<int>& GetViewport(void) const {
            return this->viewport;
        }

        /**
         * Gets the instance time code
         *
         * @return The instance time code
         */
        inline double InstanceTime(void) const {
            return this->instTime;
        }

        /**
         * Resize the viewport (meaning equals SetViewport)
         *
         * @param width  The width of the viewport to set
         * @param height The height of the viewport to set
         */
        const void Resize(int width, int height) {
            this->viewport.SetWidth(width);
            this->viewport.SetHeight(height);
        }

        /**
         * Sets the instance time code
         *
         * @param time The time code of the frame to render
         */
        inline void SetInstanceTime(double time) {
            this->instTime = time;
        }

    private:

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */

        /** The instance time code */
        double instTime;
      
        /** The viewport on the buffer */
        vislib::math::Rectangle<int> viewport;

#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */
    };

    /** Description class typedef */
    typedef megamol::core::factories::CallAutoDescription<CallSplitViewOverlay>
        CallSplitViewOverlayDescription;

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLSPLITVIEWOVERLAY_H_INCLUDED */
