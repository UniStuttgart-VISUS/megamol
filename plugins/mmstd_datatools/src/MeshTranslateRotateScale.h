/*
 * MeshTranslateRotateScale.h
 *
 * Copyright (C) 2018 MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "mmstd_datatools/AbstractMeshManipulator.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol {
namespace stdplugin {
namespace datatools {

    /**
     * Module thinning the number of particles
     *
     * Migrated from SGrottel particle's tool box
     */
    class MeshTranslateRotateScale : public AbstractMeshManipulator {
    public:

        /** Return module class name */
        static const char *ClassName(void) {
            return "MeshTranslateRotateScale";
        }

        /** Return module class description */
        static const char *Description(void) {
            return "Rotates, translates and scales the data";
        }

        /** Module is always available */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor */
        MeshTranslateRotateScale(void);

        /** Dtor */
        virtual ~MeshTranslateRotateScale(void);
        bool InterfaceIsDirty() const;
        void InterfaceResetDirty();

    protected:

        /**
         * Manipulates the particle data
         *
         * @remarks the default implementation does not changed the data
         *
         * @param outData The call receiving the manipulated data
         * @param inData The call holding the original data
         *
         * @return True on success
         */
        virtual bool manipulateData(
            megamol::geocalls::CallTriMeshData& outData,
            megamol::geocalls::CallTriMeshData& inData);

    private:

        core::param::ParamSlot translateSlot;
        core::param::ParamSlot quaternionSlot;
        core::param::ParamSlot scaleSlot;

        float** finalData;
        geocalls::CallTriMeshData::Mesh* mesh;
    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */
