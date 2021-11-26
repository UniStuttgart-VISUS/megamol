/**
 * MegaMol
 * Copyright (c) 2016, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef MEGAMOL_MEGAMOL101_CALLSPHERES_H
#define MEGAMOL_MEGAMOL101_CALLSPHERES_H

#include "mmcore/AbstractGetData3DCall.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol::megamol101_gl {

/**
 * Call for sphere data.
 */
class CallSpheres : public core::AbstractGetData3DCall {
public:
    // TUTORIAL: These static const values should be set for each call that offers callback functionality.
    // This provides a guidance which callback function gets called.

    /** Index of the 'GetData' function */
    static const unsigned int CallForGetData;

    /** Index of the 'GetExtent' function */
    static const unsigned int CallForGetExtent;

    /**
     * Answer the name of the objects of this description.
     *
     * TUTORIAL: Mandatory method for every module or call that states the name of the class.
     * This name should be unique.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName() {
        return "CallSpheres";
    }

    /**
     * Gets a human readable description of the module.
     *
     * TUTORIAL: Mandatory method for every module or call that returns a description.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return "Call to get sphere data";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * TUTORIAL: Mandatory method for every call stating the number of usable callback functions.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount() {
        return core::AbstractGetData3DCall::FunctionCount();
    }

    /**
     * Answer the name of the function used for this call.
     *
     * TUTORIAL: This function should be overloaded if you want to use other callback functions
     * than the default two.
     *
     * @param idx The index of the function to return it's name.
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        return core::AbstractGetData3DCall::FunctionName(idx);
    }

    /** Constructor */
    CallSpheres();

    /** Destructor */
    ~CallSpheres() override;

    /**
     * Answer the number of contained spheres
     *
     * @return The size of contained spheres
     */
    std::size_t Count() const;

    /**
     * Answer the color list.
     * It contains four values (RGBA) per sphere.
     * The order is RGBARGBARGBA...
     *
     * @return The color list. May be NULL if
     */
    float* GetColors() const;

    /**
     * Answer the sphere list.
     * It contains four values per sphere.
     * While the first three values (XYZ) represent the position, the last value (R) stands for the radius.
     * The order is XYZRXYZRXYZR...
     *
     * @return The sphere list.
     */
    const float* GetSpheres() const;

    /**
     * Answers whether this call has colors per sphere available.
     *
     * @return 'true' if colors are available, 'false' otherwise.
     */
    bool HasColors() const;

    /**
     * Resets the colors contained in the call to none.
     */
    void ResetColors();

    /**
     * Sets the color data. The object will not take ownership of the memory 'colors' points to.
     * The caller is responsible for keeping the data valid as long as it is used.
     *
     * The number of colors in the color list has to be the same as the number of stored spheres in this call.
     * The order of the data has to be RGBARGBARGBA...
     *
     * @param colors Pointer to a float array containing the color data.
     */
    void SetColors(float* colors);

    /**
     * Sets the data. The object will not take ownership of the memory 'spheres'
     * and 'colors' point to. The caller is responsible for keeping the data valid
     * as long as it is used.
     *
     * If no colors are provided in this call, 'HasColors' will return 'false', unless
     * 'SetColors' is used afterwards.
     * The order of the sphere data has to be XYZRXYZRXYZR...
     * The order of the color has to be RGBARGBARGBA...
     *
     * @param count The number of spheres stored in 'spheres'
     * @param spheres Pointer to a float array containing the sphere data.
     * @param colors Pointer to a float array containing the color data. Default = nullptr.
     */
    void SetData(std::size_t count, const float* spheres, float* colors = nullptr);

    /**
     * Assignment operator
     * Makes a deep copy of all members.
     *
     * TUTORIAL: The assignment operator should always be overloaded for calls.
     * This makes the creation of data modifying modules easier.
     *
     * @param rhs The right hand side operand.
     * @return A referenc to this.
     */
    CallSpheres& operator=(const CallSpheres& rhs);

private:
    /** The number of spheres */
    std::size_t count;

    /** Flag whether colors are available in this call or not */
    bool colorsAvailable;

    /** The sphere list */
    const float* spheres;

    /** The sphere color list */
    float* colors;
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<CallSpheres> CallSpheresDescription;

} // namespace megamol::megamol101_gl

#endif // MEGAMOL_MEGAMOL101_CALLSPHERES_H
