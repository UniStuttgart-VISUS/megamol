/*
 * APIValueUtil.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_APIVALUEUTIL_H_INCLUDED
#define MEGAMOLCORE_APIVALUEUTIL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "api/MegaMolCore.h"
#include "vislib/IllegalParamException.h"
#include "vislib/String.h"
#include "vislib/types.h"


namespace megamol {
namespace core {
namespace utility {

    /**
     * Utility class for managing api values.
     */
    class APIValueUtil {
    public:

        /**
         * Converts the generic type to a 32 bit integer.
         *
         * @param type The input type of 'value'.
         * @param value Pointer to the value.
         *
         * @return The converted value.
         *
         * @throw vislib::IllegalParamException if 'type' specifies a type not
         *        compatible with the requested type.
         */
        static int AsInt32(mmcValueType type, const void* value);

        /**
         * Converts the generic type to an unsigned 32 bit integer.
         *
         * @param type The input type of 'value'.
         * @param value Pointer to the value.
         *
         * @return The converted value.
         *
         * @throw vislib::IllegalParamException if 'type' specifies a type not
         *        compatible with the requested type.
         */
        static unsigned int AsUint32(mmcValueType type, const void* value);

        /**
         * Converts the generic type to a 64 bit integer.
         *
         * @param type The input type of 'value'.
         * @param value Pointer to the value.
         *
         * @return The converted value.
         *
         * @throw vislib::IllegalParamException if 'type' specifies a type not
         *        compatible with the requested type.
         */
        static INT64 AsInt64(mmcValueType type, const void* value);

        /**
         * Converts the generic type to an unsigned 64 bit integer.
         *
         * @param type The input type of 'value'.
         * @param value Pointer to the value.
         *
         * @return The converted value.
         *
         * @throw vislib::IllegalParamException if 'type' specifies a type not
         *        compatible with the requested type.
         */
        static UINT64 AsUint64(mmcValueType type, const void* value);

        /**
         * Converts the generic type to a 32 bit float.
         *
         * @param type The input type of 'value'.
         * @param value Pointer to the value.
         *
         * @return The converted value.
         *
         * @throw vislib::IllegalParamException if 'type' specifies a type not
         *        compatible with the requested type.
         */
        static float AsFloat(mmcValueType type, const void* value);

        /**
         * Converts the generic type to a 8 bit integer.
         *
         * @param type The input type of 'value'.
         * @param value Pointer to the value.
         *
         * @return The converted value.
         *
         * @throw vislib::IllegalParamException if 'type' specifies a type not
         *        compatible with the requested type.
         */
        static BYTE AsByte(mmcValueType type, const void* value);

        /**
         * Converts the generic type to a boolean.
         *
         * @param type The input type of 'value'.
         * @param value Pointer to the value.
         *
         * @return The converted value.
         *
         * @throw vislib::IllegalParamException if 'type' specifies a type not
         *        compatible with the requested type.
         */
        static bool AsBool(mmcValueType type, const void* value);

        /**
         * Converts the generic type to an ANSI string.
         *
         * @param type The input type of 'value'.
         * @param value Pointer to the value.
         *
         * @return The converted value.
         *
         * @throw vislib::IllegalParamException if 'type' specifies a type not
         *        compatible with the requested type.
         */
        static vislib::StringA AsStringA(mmcValueType type, const void* value);

        /**
         * Converts the generic type to an Unicode string.
         *
         * @param type The input type of 'value'.
         * @param value Pointer to the value.
         *
         * @return The converted value.
         *
         * @throw vislib::IllegalParamException if 'type' specifies a type not
         *        compatible with the requested type.
         */
        static vislib::StringW AsStringW(mmcValueType type, const void* value);

        /**
         * Answers whether the given type is a integer type.
         *
         * @param type The type to test.
         *
         * @return 'true' if the type is a integer type, 'false' otherwise.
         */
        static bool IsIntType(mmcValueType type);

        /**
         * Answers whether the given type is a string type.
         *
         * @param type The type to test.
         *
         * @return 'true' if the type is a string type, 'false' otherwise.
         */
        static bool IsStringType(mmcValueType type);

        /**
         * Answers whether the given type is a float type.
         *
         * @param type The type to test.
         *
         * @return 'true' if the type is a float type, 'false' otherwise.
         */
        static inline bool IsFloatType(mmcValueType type) {
            return type == MMC_TYPE_FLOAT;
        }

        /**
         * Answers whether the given type is a boolean type.
         *
         * @param type The type to test.
         *
         * @return 'true' if the type is a boolean type, 'false' otherwise.
         */
        static inline bool IsBoolType(mmcValueType type) {
            return type == MMC_TYPE_BOOL;
        }

    };


} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_APIVALUEUTIL_H_INCLUDED */
