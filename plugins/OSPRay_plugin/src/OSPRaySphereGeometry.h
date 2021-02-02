/*
 * OSPRaySphereGeometry.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "OSPRay_plugin/AbstractOSPRayStructure.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace ospray {

    class OSPRaySphereGeometry : public AbstractOSPRayStructure {

    public:
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char* ClassName(void) {
            return "OSPRaySphereGeometry";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char* Description(void) {
            return "Creator for OSPRay sphere geometries.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** Dtor. */
        virtual ~OSPRaySphereGeometry(void);

        /** Ctor. */
        OSPRaySphereGeometry(void);

    protected:
        virtual bool create();
        virtual void release();

        virtual bool readData(core::Call& call);
        virtual bool getExtends(core::Call& call);

        bool InterfaceIsDirty();
        void getClipData(float* clipDat, float* clipCol);

        core::param::ParamSlot particleList;

        /** The call for data */
        core::CallerSlot getDataSlot;

        /** The call for clipping plane */
        core::CallerSlot getClipPlaneSlot;

    private:
        // data objects
        std::vector<float> cd_rgba;
        std::vector<float> vd;
    };

} // namespace ospray
} // namespace megamol
