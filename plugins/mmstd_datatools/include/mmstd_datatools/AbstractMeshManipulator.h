/*
 * AbstractMeshManipulator.h
 *
 * Copyright (C) 2018
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmstd_datatools/mmstd_datatools.h"
#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "geometry_calls/CallTriMeshData.h"


namespace megamol {
namespace stdplugin {
namespace datatools {


    /**
     * Abstract class of particle data manipulators
     *
     * Migrated from SGrottel particle's tool box
     */
    class MMSTD_DATATOOLS_API AbstractMeshManipulator : public megamol::core::Module {
    public:

        /**
         * Ctor
         *
         * @param outSlotName The name for the slot providing the manipulated data
         * @param inSlotName The name for the slot accessing the original data
         */
        AbstractMeshManipulator(const char *outSlotName,
            const char *inSlotName);

        /** Dtor */
        virtual ~AbstractMeshManipulator(void);

    protected:

        /** Lazy initialization of the module */
        virtual bool create(void);

        /** Resource release */
        virtual void release(void);

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
            geocalls::CallTriMeshData& outData, geocalls::CallTriMeshData& inData);

        /**
         * Manipulates the particle data extend information
         *
         * @remarks the default implementation does not changed the data
         *
         * @param outData The call receiving the manipulated information
         * @param inData The call holding the original data
         *
         * @return True on success
         */
        virtual bool manipulateExtent(
            geocalls::CallTriMeshData& outData, geocalls::CallTriMeshData& inData);

    private:

        /**
         * Called when the data is requested by this module
         *
         * @param c The incoming call
         *
         * @return True on success
         */
        bool getDataCallback(megamol::core::Call& c);

        /**
         * Called when the extend information is requested by this module
         *
         * @param c The incoming call
         *
         * @return True on success
         */
        bool getExtentCallback(megamol::core::Call& c);

        /** The slot providing access to the manipulated data */
        megamol::core::CalleeSlot outDataSlot;

        /** The slot accessing the original data */
        megamol::core::CallerSlot inDataSlot;

    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

