/*
 * AbstractManipulator.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Copyright (C) 2019 by MegaMol Dev Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd_datatools/mmstd_datatools.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

/**
 * Abstract class data manipulators for calls with getData/getExtent interface
 */
template <class C> class AbstractManipulator : public megamol::core::Module {
public:
    /**
     * Ctor
     *
     * @param outSlotName The name for the slot providing the manipulated data
     * @param inSlotName The name for the slot accessing the original data
     */
    AbstractManipulator(const char* outSlotName, const char* inSlotName);

    /** Dtor */
    virtual ~AbstractManipulator(void);

protected:
    /** Lazy initialization of the module */
    bool create(void) override;

    /** Resource release */
    void release(void) override;

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
    virtual bool manipulateData(C& outData, C& inData);

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
    virtual bool manipulateExtent(C& outData, C& inData);

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


template <class C>
AbstractManipulator<C>::AbstractManipulator(const char* outSlotName, const char* inSlotName)
    : megamol::core::Module()
    , outDataSlot(outSlotName, "providing access to the manipulated data")
    , inDataSlot(inSlotName, "accessing the original data") {

    this->outDataSlot.SetCallback(C::ClassName(), "GetData", &AbstractManipulator::getDataCallback);
    this->outDataSlot.SetCallback(C::ClassName(), "GetExtent", &AbstractManipulator::getExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->inDataSlot.template SetCompatibleCall<core::factories::CallAutoDescription<C>>();
    this->MakeSlotAvailable(&this->inDataSlot);
}


template <class C> AbstractManipulator<C>::~AbstractManipulator() { this->Release(); }


template <class C> bool AbstractManipulator<C>::create() { return true; }


template <class C> void AbstractManipulator<C>::release() {}


template <class C> bool AbstractManipulator<C>::manipulateData(C& outData, C& inData) {
    outData = inData;
    inData.SetUnlocker(nullptr, false);
    return true;
}


template <class C> bool AbstractManipulator<C>::manipulateExtent(C& outData, C& inData) {
    outData = inData;
    inData.SetUnlocker(nullptr, false);
    return true;
}


template <class C> bool AbstractManipulator<C>::getDataCallback(megamol::core::Call& c) {
    auto outMpdc = dynamic_cast<C*>(&c);
    if (outMpdc == NULL) return false;

    auto inMpdc = this->inDataSlot.template CallAs<C>();
    if (inMpdc == NULL) return false;

    *inMpdc = *outMpdc; // to get the correct request time
    if (!(*inMpdc)(0)) return false;

    if (!this->manipulateData(*outMpdc, *inMpdc)) {
        inMpdc->Unlock();
        return false;
    }

    inMpdc->Unlock();

    return true;
}


template <class C> bool AbstractManipulator<C>::getExtentCallback(megamol::core::Call& c) {
    auto outMpdc = dynamic_cast<C*>(&c);
    if (outMpdc == NULL) return false;

    auto inMpdc = this->inDataSlot.template CallAs<C>();
    if (inMpdc == NULL) return false;

    *inMpdc = *outMpdc; // to get the correct request time
    if (!(*inMpdc)(1)) return false;

    if (!this->manipulateExtent(*outMpdc, *inMpdc)) {
        inMpdc->Unlock();
        return false;
    }

    inMpdc->Unlock();

    return true;
}


} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */
