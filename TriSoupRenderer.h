/*
 * TriSoupRenderer.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_TRISOUPRENDERER_H_INCLUDED
#define MEGAMOLCORE_TRISOUPRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "view/Renderer3DModule.h"
#include "Call.h"
#include "param/ParamSlot.h"
#include "vislib/Cuboid.h"
#include "vislib/memutils.h"


namespace megamol {
namespace core {
namespace misc {


    /**
     * Renderer for rendering the vis logo into the unit cube.
     */
    class TriSoupRenderer : public view::Renderer3DModule {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "TriSoupRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Renderer for triangle soups.";
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
        TriSoupRenderer(void);

        /** Dtor. */
        virtual ~TriSoupRenderer(void);

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
        virtual bool GetCapabilities(Call& call);

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
        virtual bool Render(Call& call);

    private:

        /**
         * Nested class storing the triangle mesh of a cluster.
         */
        class Cluster {
        public:

            /** Ctor */
            Cluster(void) : id(0), tc(0), t(NULL), vc(0), v(NULL), n(NULL),
                    c(NULL) {
                // intentionally empty
            }

            /** Dtor. */
            ~Cluster(void) {
                this->tc = 0;
                ARY_SAFE_DELETE(this->t);
                this->vc = 0;
                ARY_SAFE_DELETE(this->v);
                ARY_SAFE_DELETE(this->n);
                ARY_SAFE_DELETE(this->c);
            }

            /** Cluster ID */
            unsigned int id;

            /** Triangle count */
            unsigned int tc;

            /** Triangle vertex indices (3 times tc) */
            unsigned int *t;

            /** Vertex count */
            unsigned int vc;

            /** Vertices (3 times vc) */
            float * v;

            /** Normals (3 times vc) */
            float * n;

            /** Colors (3 times vc) */
            unsigned char * c;

        };

        /**
         * Tries to load the file 'filename' into memory
         */
        void tryLoadFile(void);

        /**
         * Just for debugging purposes.
         *
         * @param slot The calling slot.
         *
         * @return Always 'true' to reset the dirty flag.
         */
        bool bullchit(param::ParamSlot& slot);

        /** The file name */
        param::ParamSlot filename;

        /** number of clusters stored */
        unsigned int cnt;

        /** array of clusters stored */
        Cluster *clusters;

        /** Flag whether or not to show vertices */
        param::ParamSlot showVertices;

        /** Flag whether or not use lighting for the surface */
        param::ParamSlot lighting;

        /** The rendering style for the surface */
        param::ParamSlot surStyle;

        /** A test button parameter slot */
        param::ParamSlot testButton;

        /** The fancy bounding box */
        vislib::math::Cuboid<float> bbox;

    };


} /* end namespace misc */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_TRISOUPRENDERER_H_INCLUDED */
