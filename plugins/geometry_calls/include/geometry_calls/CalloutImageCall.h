/*
 * CalloutImageCall.h
 *
 * Copyright (C) 2014 by Florian Frieß
 * Copyright (C) 2008-2014 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/Array.h"
#include "vislib/math/Vector.h"
#include <vector>

namespace megamol::geocalls {

/**
 * The CalloutImageCall contains 5 parameters:
 *
 * The id of the QR-code. Th id is -1 if the QR-code is not
 * registered yet. After the first call the id is a unique
 * number > 0.
 *
 * The data for the image, which is stored in an std::vector
 * of type float. Therefor only greyscale images can be send
 * with this call.
 *
 * The width of the image.
 *
 * The height of the image.
 *
 * The point in world coordinates the CalloutBox is supposed to
 * point at.
 */

class CalloutImageCall : public core::Call {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) {
        return "CalloutImageCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "Call to get QR-Code Image";
    }

    /** Index of the 'GetData' function */
    static const unsigned int CallForGetID;

    static const unsigned int CallForSetID;

    static const unsigned int CallForGetImage;

    static const unsigned int CallForSetImage;

    static const unsigned int CallForGetPointAt;

    static const unsigned int CallForSetPointAt;

    static const unsigned int CallForGetWidth;

    static const unsigned int CallForSetWidth;

    static const unsigned int CallForGetHeight;

    static const unsigned int CallForSetHeight;

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
        return 12;
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case 0:
            return "getID";
        case 1:
            return "setID";
        case 2:
            return "getImage";
        case 3:
            return "setImage";
        case 4:
            return "getPointAt";
        case 5:
            return "setPointAt";
        case 6:
            return "getWidth";
        case 7:
            return "setWidth";
        case 8:
            return "getHeight";
        case 9:
            return "setHeight";
        case 10:
            return "RegisterQR";
        case 11:
            return "DelteQR";
        default:
            return NULL;
        }
        return "";
    }

    inline int* GetID(void) const {
        return this->qr_id;
    }

    inline void SetID(int* p_qr_id) {
        this->qr_id = p_qr_id;
    }

    inline std::vector<float>* GetImage(void) const {
        return this->qr_image;
    }

    inline void SetImage(std::vector<float>* p_qr_image) {
        this->qr_image = p_qr_image;
    }

    inline vislib::math::Vector<float, 3>* GetPointAt(void) const {
        return this->qr_pointer;
    }

    inline void SetPointAt(vislib::math::Vector<float, 3>* p_qr_pointer) {
        this->qr_pointer = p_qr_pointer;
    }

    inline int* GetWidth(void) const {
        return this->width;
    }

    inline void SetWidth(int* p_width) {
        this->width = p_width;
    }

    inline int* GetHeight(void) const {
        return this->height;
    }

    inline void SetHeight(int* p_height) {
        this->height = p_height;
    }

    CalloutImageCall(void);
    ~CalloutImageCall(void) override;

private:
    int* qr_id;
    std::vector<float>* qr_image;
    vislib::math::Vector<float, 3>* qr_pointer;
    int* width;
    int* height;
};

/** Description class typedef */
typedef core::factories::CallAutoDescription<CalloutImageCall> CalloutImageCallDescription;

} // namespace megamol::geocalls
