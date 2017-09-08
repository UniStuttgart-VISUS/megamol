/*
 * ExtBezierMeshRenderer.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef BEZTUBE_EXTBEZIERMESHRENDERER_H_INCLUDED
#define BEZTUBE_EXTBEZIERMESHRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "beztube/beztube.h"
#include "ext/ExtBezierDataCall.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/forceinline.h"
#include "vislib/math/Point.h"
#include "vislib/math/Vector.h"


namespace megamol {
namespace beztube {
namespace ext {

    /**
     * Mesh-based renderer for extended bézier curve tubes
     */
    class ExtBezierMeshRenderer : public core::view::Renderer3DModule {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "ExtBezierMeshRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Renderer for extended bézier curve";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor. */
        ExtBezierMeshRenderer(void);

        /** Dtor. */
        virtual ~ExtBezierMeshRenderer(void);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * The get capabilities callback. The module should set the members
         * of 'call' to tell the caller its capabilities.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetCapabilities(core::Call& call);

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetExtents(core::Call& call);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Render(core::Call& call);

    private:

        /**
         * Calculates the local coordinate system base on the curve at value t
         *
         * @param curve The curve
         * @param t The interpolation value t
         * @param x The x axis
         * @param y The y axis
         * @param z The z axis
         */
        VISLIB_FORCEINLINE void calcBase(const vislib::math::BezierCurve<
            ExtBezierDataCall::Point, 3>& curve, float t,
            vislib::math::Vector<float, 3>& x,
            vislib::math::Vector<float, 3>& y,
            vislib::math::Vector<float, 3>& z);

        /**
         * draws curves with given profile
         *
         * @param curves The array of curves
         * @param cnt The number of curves
         * @param profile The profile points
         * @param profCnt The number of profile counts
         * @param lengthSections Number of sections used for each curve
         * @param profSmooth Flag whether or not to smooth the profile
         */
        void drawCurves(const vislib::math::BezierCurve<
            ExtBezierDataCall::Point, 3> *curves, SIZE_T cnt,
            const vislib::math::Point<float, 2> *profile, SIZE_T profCnt,
            unsigned int lengthSections, bool profSmooth);

        /**
         * draws curves with elliptic profile
         *
         * @param curves The array of curves
         * @param cnt The number of curves
         * @param profileSections Number of sections used for the profile
         * @param lengthSections Number of sections used for each curve
         */
        void drawEllipCurves(const vislib::math::BezierCurve<
            ExtBezierDataCall::Point, 3> *curves, SIZE_T cnt,
            unsigned int profileSections, unsigned int lengthSections);

        /**
         * draws curves with rectangular profile
         *
         * @param curves The array of curves
         * @param cnt The number of curves
         * @param lengthSections Number of sections used for each curve
         */
        void drawRectCurves(const vislib::math::BezierCurve<
            ExtBezierDataCall::Point, 3> *curves, SIZE_T cnt,
            unsigned int lengthSections);

        /** The call for data */
        core::CallerSlot getDataSlot;

        /** The number of linear sections along the curve */
        core::param::ParamSlot curveSectionsSlot;

        /** The number of section along the profile */
        core::param::ParamSlot profileSectionsSlot;

        /** The display list storing the objects */
        unsigned int objs;

        /** The data hash of the objects stored in the list */
        SIZE_T objsHash;

    };

} /* end namespace ext */
} /* end namespace beztube */
} /* end namespace megamol */

#endif /* BEZTUBE_EXTBEZIERMESHRENDERER_H_INCLUDED */
