/*
 * MuxRenderer3D.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MUXRENDERER3D_H_INCLUDED
#define MEGAMOLCORE_MUXRENDERER3D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/BoundingBoxes.h"
#include "mmcore/CallerSlot.h"
#include "CallRender3D.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ParamSlot.h"
#include "Renderer3DModule.h"
#include "vislib/graphics/CameraParameters.h"
#include "vislib/graphics/CameraParamsStore.h"
#include "vislib/math/Cuboid.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/memutils.h"
#include "vislib/String.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * A multi-renderer-3d multiplexer
     * T is the number of outgoing renderers
     */
    template<unsigned int T>
    class MuxRenderer3D : public Renderer3DModule {
    public:

        /**
         * Gets the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            static vislib::StringA classname;
            classname.Format("Mux%dRenderer3D", (int)(T));
            return classname.PeekBuffer();
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            static vislib::StringA description;
            description.Format("Multiplex renderer 3d to merge %d Renderer3D modules into a single view", (int)(T));
            return description.PeekBuffer();
        }

        /**
         * Gets whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /**
         * Disallow usage in quickstarts
         *
         * @return false
         */
        static bool SupportQuickstart(void) {
            return false;
        }

        /** Ctor. */
        MuxRenderer3D(void);

        /** Dtor. */
        virtual ~MuxRenderer3D(void);

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

    protected:

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetExtents(Call& call);

        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Render(Call& call);

    private:

        /** The renderer caller slots */
        CallerSlot *rendererSlot[T];

        /** Activation flags for the rendering slots */
        param::ParamSlot *rendererActiveSlot[T];

        /** The frame count */
        unsigned int frameCnt;

        /** The bounding boxes */
        BoundingBoxes bboxs;

        /** A scaling factor */
        float scale;

    };


    /*
     * MuxRenderer3D<T>::MuxRenderer3D
     */
    template<unsigned int T>
    MuxRenderer3D<T>::MuxRenderer3D(void) : Renderer3DModule(), frameCnt(0), bboxs(), scale(1.0f) {
        vislib::StringA name, desc;
        for (unsigned int i = 0; i < T; i++) {

            name.Format("renderer%u", i + 1);
            desc.Format("Outgoing renderer #%u", i + 1);
            this->rendererSlot[i] = new CallerSlot(name, desc);
            this->rendererSlot[i]->template SetCompatibleCall<CallRender3DDescription>();
            this->MakeSlotAvailable(this->rendererSlot[i]);

            name += "active";
            desc.Format("De-/Activates outgoing renderer #%u", i + 1);
            this->rendererActiveSlot[i] = new param::ParamSlot(name, desc);
            this->rendererActiveSlot[i]->SetParameter(new param::BoolParam(true));
            this->MakeSlotAvailable(this->rendererActiveSlot[i]);
        }
    }


    /*
     * MuxRenderer3D<T>::~MuxRenderer3D
     */
    template<unsigned int T>
    MuxRenderer3D<T>::~MuxRenderer3D(void) {
        this->Release();
        for (unsigned int i = 0; i < T; i++) {
            SAFE_DELETE(this->rendererSlot[i]);
        }
    }


    /*
     * MuxRenderer3D<T>::create
     */
    template<unsigned int T>
    bool MuxRenderer3D<T>::create(void) {
        // intentionally empty
        return true;
    }


    /*
     * MuxRenderer3D<T>::release
     */
    template<unsigned int T>
    void MuxRenderer3D<T>::release(void) {
        // intentionally empty
    }


    /*
     * MuxRenderer3D<T>::GetExtents
     */
    template<unsigned int T>
    bool MuxRenderer3D<T>::GetExtents(Call& call) {
        CallRender3D *cr3d = dynamic_cast<CallRender3D*>(&call);
        if (cr3d == NULL) return false;

        this->bboxs.Clear();
        this->frameCnt = 0;
        for (unsigned int i = 0; i < T; i++) {
            if (!this->rendererActiveSlot[i]->template Param<param::BoolParam>()->Value()) continue;
            CallRender3D *oc = this->rendererSlot[i]->template CallAs<CallRender3D>();
            if ((oc == NULL) || (!(*oc)(AbstractCallRender::FnGetExtents))) continue;
            if (this->frameCnt == 0) {
                if (oc->AccessBoundingBoxes().IsObjectSpaceBBoxValid()) {
                    this->bboxs.SetObjectSpaceBBox(oc->AccessBoundingBoxes().ObjectSpaceBBox());
                } else if (oc->AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
                    this->bboxs.SetObjectSpaceBBox(oc->AccessBoundingBoxes().ObjectSpaceClipBox());
                } else {
                    this->bboxs.SetObjectSpaceBBox(vislib::math::Cuboid<float>(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f));
                }
                if (oc->AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
                    this->bboxs.SetObjectSpaceClipBox(oc->AccessBoundingBoxes().ObjectSpaceClipBox());
                } else if (oc->AccessBoundingBoxes().IsObjectSpaceBBoxValid()) {
                    this->bboxs.SetObjectSpaceClipBox(oc->AccessBoundingBoxes().ObjectSpaceBBox());
                } else {
                    this->bboxs.SetObjectSpaceClipBox(vislib::math::Cuboid<float>(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f));
                }

            } else {
                if (oc->AccessBoundingBoxes().IsObjectSpaceBBoxValid()) {
                    vislib::math::Cuboid<float> box(this->bboxs.ObjectSpaceBBox());
                    box.Union(oc->AccessBoundingBoxes().ObjectSpaceBBox());
                    this->bboxs.SetObjectSpaceBBox(box);
                } else if (oc->AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
                    vislib::math::Cuboid<float> box(this->bboxs.ObjectSpaceBBox());
                    box.Union(oc->AccessBoundingBoxes().ObjectSpaceClipBox());
                    this->bboxs.SetObjectSpaceBBox(box);
                }
                if (oc->AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
                    vislib::math::Cuboid<float> box(this->bboxs.ObjectSpaceClipBox());
                    box.Union(oc->AccessBoundingBoxes().ObjectSpaceClipBox());
                    this->bboxs.SetObjectSpaceClipBox(box);
                } else if (oc->AccessBoundingBoxes().IsObjectSpaceBBoxValid()) {
                    vislib::math::Cuboid<float> box(this->bboxs.ObjectSpaceClipBox());
                    box.Union(oc->AccessBoundingBoxes().ObjectSpaceBBox());
                    this->bboxs.SetObjectSpaceClipBox(box);
                }
            }
            this->frameCnt = vislib::math::Max(this->frameCnt, oc->TimeFramesCount());

        }
        if (this->frameCnt == 0) {
            this->frameCnt = 1;
            this->scale = 1.0f;
            this->bboxs.Clear();
        } else {
            this->scale = 1.0f / this->bboxs.ObjectSpaceBBox().LongestEdge();
            this->bboxs.MakeScaledWorld(scale);
        }

        cr3d->SetTimeFramesCount(this->frameCnt);
        cr3d->AccessBoundingBoxes() = this->bboxs;

        return true;
    }


    /*
     * MuxRenderer3D<T>::Render
     */
    template<unsigned int T>
    bool MuxRenderer3D<T>::Render(Call& call) {
        CallRender3D *cr3d = dynamic_cast<CallRender3D*>(&call);
        vislib::SmartPtr<vislib::graphics::CameraParameters> camParams
            = new vislib::graphics::CameraParamsStore();
        if (cr3d == NULL) return false;

        if (this->frameCnt == 0) {
            this->GetExtents(call);
        }

        for (unsigned int i = 0; i < T; i++) {
            if (!this->rendererActiveSlot[i]->template Param<param::BoolParam>()->Value()) continue;
            CallRender3D *oc = this->rendererSlot[i]->template CallAs<CallRender3D>();
            if (oc == NULL) continue;
            *oc = *cr3d;
            if (!(*oc)(view::AbstractCallRender::FnGetExtents)) continue;

            // Back translation ocWS -> ocOS
            float sx, sy, sz, tx, ty, tz;
            const vislib::math::Cuboid<float>& ocWS = oc->AccessBoundingBoxes().WorldSpaceBBox();
            const vislib::math::Cuboid<float>& ocOS = oc->AccessBoundingBoxes().ObjectSpaceBBox();

            sx = ocOS.Width() / ocWS.Width();
            sy = ocOS.Height() / ocWS.Height();
            sz = ocOS.Depth() / ocWS.Depth();

            tx = ocWS.Left() * sx - ocOS.Left();
            ty = ocWS.Bottom() * sy - ocOS.Bottom();
            tz = ocWS.Back() * sz - ocOS.Back();

            // We clamp the time to the range of the individual renderers
            int octfc = oc->TimeFramesCount();
            *oc = *cr3d;
            oc->SetTime(vislib::math::Min<float>(cr3d->Time(), static_cast<float>(octfc - 1)));

            camParams->CopyFrom(cr3d->GetCameraParameters());
            oc->SetCameraParameters(camParams);
            vislib::math::Point<float, 3> p = camParams->Position();
            vislib::math::Point<float, 3> l = camParams->LookAt();
            p.Set((p.X() / this->scale - tx) / sx,
                  (p.Y() / this->scale - ty) / sy,
                  (p.Z() / this->scale - tz) / sz);
            l.Set((l.X() / this->scale - tx) / sx,
                  (l.Y() / this->scale - ty) / sy,
                  (l.Z() / this->scale - tz) / sz);

            camParams->SetView(p, l, camParams->Up());

            ::glMatrixMode(GL_MODELVIEW);
            ::glPushMatrix();

            ::glScalef(sx, sy, sz);
            ::glTranslatef(tx, ty, tz); // TODO: Not sure about the ordering here!
            ::glScalef(this->scale, this->scale, this->scale);

            (*oc)(view::AbstractCallRender::FnRender);

            ::glMatrixMode(GL_MODELVIEW);
            ::glPopMatrix();
        }

        return true;
    }

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MUXRENDERER3D_H_INCLUDED */
