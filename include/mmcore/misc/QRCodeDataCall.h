/*
 * QRCodeDataCall.h
 *
 * Copyright (C) 2014 by Florian Frieß
 * Copyright (C) 2008-2014 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_QRCODEDATACALL_H_INCLUDED
#define MEGAMOLCORE_QRCODEDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Call.h"
#include "CallAutoDescription.h"
#include "BoundingBoxes.h"
#include "vislib/Array.h"
#include "vislib/math/Vector.h"
#include <vector>


namespace megamol {
namespace core {
namespace misc {

    class MEGAMOLCORE_API QRCodeDataCall : public megamol::core::Call {
    public:
        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "QRCodeDataCall";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call to create a QR code Image";
        }

        /** Index of the 'GetData' function */
		static const unsigned int CallForGetText;

		static const unsigned int CallForSetText;

		static const unsigned int CallForGetPointAt;

		static const unsigned int CallForSetPointAt;

		static const unsigned int CallForGetBoundingBox;

		static const unsigned int CallForSetBoundingBox;

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return 8;
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
                return "getText";
            case 1:
                return "setText";
			case 2:
				return "getPointAt";
			case 3:
				return "setPointAt";
			case 4:
				return "getBoundingBox";
			case 5:
				return "setBoundingBox";
			case 6:
				return "QRDataDelivered";
			case 7:
				return "QRDeleteData";
			default:
				return NULL;
            }
            return "";
        }

		inline std::string* GetTextPointer(void) const
		{
			return this->qr_text;
		}

		inline void SetTextPointer(std::string* p_qr_text)
		{
			this->qr_text = p_qr_text;
		}

		inline vislib::math::Vector<float, 3>* GetPointAtPointer(void) const
		{
			return this->qr_pointer;
		}

		inline void SetPointAtPointer(vislib::math::Vector<float, 3>* p_qr_pointer)
		{
			this->qr_pointer = p_qr_pointer;
		}

		inline BoundingBoxes* GetBoundingBoxPointer(void) const {
            return this->bbox;
        }

		inline void SetBoundingBoxPointer(BoundingBoxes* p_bbox)
		{
			this->bbox = p_bbox;
		}

        QRCodeDataCall(void);
        virtual ~QRCodeDataCall(void);

    private:
		std::string* qr_text;
		vislib::math::Vector<float, 3>* qr_pointer;
		BoundingBoxes* bbox;
    };

    /** Description class typedef */
    typedef CallAutoDescription<QRCodeDataCall> QRCodeDataCallDescription;

} /* end namespace misc */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_QRCODEDATACALL_H_INCLUDED */
