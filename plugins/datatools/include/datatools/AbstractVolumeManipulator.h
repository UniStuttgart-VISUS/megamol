/*
 * AbstractVolumeManipulator.h
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "geometry_calls/VolumetricDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"


namespace megamol {
namespace datatools {


/**
 * Abstract class of particle data manipulators
 *
 * Migrated from SGrottel particle's tool box
 */
class AbstractVolumeManipulator : public megamol::core::Module {
public:
    /**
     * Ctor
     *
     * @param outSlotName The name for the slot providing the manipulated data
     * @param inSlotName The name for the slot accessing the original data
     */
    AbstractVolumeManipulator(const char* outSlotName, const char* inSlotName);

    /** Dtor */
    ~AbstractVolumeManipulator() override;

protected:
    /** Lazy initialization of the module */
    bool create() override;

    /** Resource release */
    void release() override;

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
    virtual bool manipulateData(geocalls::VolumetricDataCall& outData, geocalls::VolumetricDataCall& inData);

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
    virtual bool manipulateExtent(geocalls::VolumetricDataCall& outData, geocalls::VolumetricDataCall& inData);

    virtual bool manipulateMetaData(geocalls::VolumetricDataCall& outData, geocalls::VolumetricDataCall& inData);

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

    bool getMetaDataCallback(megamol::core::Call& c);

    bool startAsyncCallback(megamol::core::Call& c);

    bool stopAsyncCallback(megamol::core::Call& c);

    bool tryGetDataCallback(megamol::core::Call& c);

    /** The slot providing access to the manipulated data */
    megamol::core::CalleeSlot outDataSlot;

    /** The slot accessing the original data */
    megamol::core::CallerSlot inDataSlot;
};

} /* end namespace datatools */
} /* end namespace megamol */
