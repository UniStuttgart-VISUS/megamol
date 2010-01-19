/*
 * APIValueUtil.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "utility/APIValueUtil.h"
#include "vislib/CharTraits.h"
#include "vislib/mathfunctions.h"
#include "vislib/StringConverter.h"


/*
 * megamol::core::utility::APIValueUtil::IsIntType
 */
int megamol::core::utility::APIValueUtil::AsInt32(mmcValueType type, const void* value) {
    switch (type) {
        case MMC_TYPE_INT32:
            return *static_cast<const int*>(value);
        case MMC_TYPE_UINT32:
            return static_cast<int>(*static_cast<const unsigned int*>(value));
        case MMC_TYPE_INT64:
            return static_cast<int>(*static_cast<const INT64*>(value));
        case MMC_TYPE_UINT64:
            return static_cast<int>(*static_cast<const UINT64*>(value));
        case MMC_TYPE_BYTE:
            return static_cast<int>(*static_cast<const unsigned char*>(value));
        case MMC_TYPE_BOOL:
            return static_cast<int>(*static_cast<const bool*>(value));
        case MMC_TYPE_FLOAT:
            return static_cast<int>(*static_cast<const float*>(value));
        case MMC_TYPE_CSTR:
            return vislib::CharTraitsA::ParseInt(static_cast<const char*>(value));
        case MMC_TYPE_WSTR:
            return vislib::CharTraitsW::ParseInt(static_cast<const wchar_t*>(value));
        case MMC_TYPE_VOIDP:
            break;
        default:
            break;
    }
    throw vislib::IllegalParamException("Cannot convert value", __FILE__, __LINE__);
}


/*
 * megamol::core::utility::APIValueUtil::IsIntType
 */
unsigned int megamol::core::utility::APIValueUtil::AsUint32(mmcValueType type, const void* value) {
    switch (type) {
        case MMC_TYPE_INT32:
            return static_cast<unsigned int>(*static_cast<const int*>(value));
        case MMC_TYPE_UINT32:
            return *static_cast<const unsigned int*>(value);
        case MMC_TYPE_INT64:
            return static_cast<unsigned int>(*static_cast<const INT64*>(value));
        case MMC_TYPE_UINT64:
            return static_cast<unsigned int>(*static_cast<const UINT64*>(value));
        case MMC_TYPE_BYTE:
            return static_cast<unsigned int>(*static_cast<const unsigned char*>(value));
        case MMC_TYPE_BOOL:
            return static_cast<unsigned int>(*static_cast<const bool*>(value));
        case MMC_TYPE_FLOAT:
            return static_cast<unsigned int>(*static_cast<const float*>(value));
        case MMC_TYPE_CSTR:
            return static_cast<unsigned int>(vislib::CharTraitsA::ParseInt(static_cast<const char*>(value)));
        case MMC_TYPE_WSTR:
            return static_cast<unsigned int>(vislib::CharTraitsW::ParseInt(static_cast<const wchar_t*>(value)));
        case MMC_TYPE_VOIDP:
            break;
        default:
            break;
    }
    throw vislib::IllegalParamException("Cannot convert value", __FILE__, __LINE__);
}


/*
 * megamol::core::utility::APIValueUtil::IsIntType
 */
INT64 megamol::core::utility::APIValueUtil::AsInt64(mmcValueType type, const void* value) {
    switch (type) {
        case MMC_TYPE_INT32:
            return static_cast<INT64>(*static_cast<const int*>(value));
        case MMC_TYPE_UINT32:
            return static_cast<INT64>(*static_cast<const unsigned int*>(value));
        case MMC_TYPE_INT64:
            return *static_cast<const INT64*>(value);
        case MMC_TYPE_UINT64:
            return static_cast<INT64>(*static_cast<const UINT64*>(value));
        case MMC_TYPE_BYTE:
            return static_cast<INT64>(*static_cast<const unsigned char*>(value));
        case MMC_TYPE_BOOL:
            return static_cast<INT64>(*static_cast<const bool*>(value));
        case MMC_TYPE_FLOAT:
            return static_cast<INT64>(*static_cast<const float*>(value));
        case MMC_TYPE_CSTR:
            return static_cast<INT64>(vislib::CharTraitsA::ParseInt(static_cast<const char*>(value)));
        case MMC_TYPE_WSTR:
            return static_cast<INT64>(vislib::CharTraitsW::ParseInt(static_cast<const wchar_t*>(value)));
        case MMC_TYPE_VOIDP:
            break;
        default:
            break;
    }
    throw vislib::IllegalParamException("Cannot convert value", __FILE__, __LINE__);
}


/*
 * megamol::core::utility::APIValueUtil::IsIntType
 */
UINT64 megamol::core::utility::APIValueUtil::AsUint64(mmcValueType type, const void* value) {
    switch (type) {
        case MMC_TYPE_INT32:
            return static_cast<UINT64>(*static_cast<const int*>(value));
        case MMC_TYPE_UINT32:
            return static_cast<UINT64>(*static_cast<const unsigned int*>(value));
        case MMC_TYPE_INT64:
            return static_cast<UINT64>(*static_cast<const INT64*>(value));
        case MMC_TYPE_UINT64:
            return *static_cast<const UINT64*>(value);
        case MMC_TYPE_BYTE:
            return static_cast<UINT64>(*static_cast<const unsigned char*>(value));
        case MMC_TYPE_BOOL:
            return static_cast<UINT64>(*static_cast<const bool*>(value));
        case MMC_TYPE_FLOAT:
            return static_cast<UINT64>(*static_cast<const float*>(value));
        case MMC_TYPE_CSTR:
            return static_cast<UINT64>(vislib::CharTraitsA::ParseInt(static_cast<const char*>(value)));
        case MMC_TYPE_WSTR:
            return static_cast<UINT64>(vislib::CharTraitsW::ParseInt(static_cast<const wchar_t*>(value)));
        case MMC_TYPE_VOIDP:
            break;
        default:
            break;
    }
    throw vislib::IllegalParamException("Cannot convert value", __FILE__, __LINE__);
}


/*
 * megamol::core::utility::APIValueUtil::IsIntType
 */
float megamol::core::utility::APIValueUtil::AsFloat(mmcValueType type, const void* value) {
    switch (type) {
        case MMC_TYPE_INT32:
            return static_cast<float>(*static_cast<const int*>(value));
        case MMC_TYPE_UINT32:
            return static_cast<float>(*static_cast<const unsigned int*>(value));
        case MMC_TYPE_INT64:
            return static_cast<float>(*static_cast<const INT64*>(value));
        case MMC_TYPE_UINT64:
            return static_cast<float>(*static_cast<const UINT64*>(value));
        case MMC_TYPE_BYTE:
            return static_cast<float>(*static_cast<const unsigned char*>(value));
        case MMC_TYPE_BOOL:
            return static_cast<float>(*static_cast<const bool*>(value));
        case MMC_TYPE_FLOAT:
            return *static_cast<const float*>(value);
        case MMC_TYPE_CSTR:
            return static_cast<float>(vislib::CharTraitsA::ParseDouble(static_cast<const char*>(value)));
        case MMC_TYPE_WSTR:
            return static_cast<float>(vislib::CharTraitsW::ParseDouble(static_cast<const wchar_t*>(value)));
        case MMC_TYPE_VOIDP:
            break;
        default:
            break;
    }
    throw vislib::IllegalParamException("Cannot convert value", __FILE__, __LINE__);
}


/*
 * megamol::core::utility::APIValueUtil::IsIntType
 */
BYTE megamol::core::utility::APIValueUtil::AsByte(mmcValueType type, const void* value) {
    switch (type) {
        case MMC_TYPE_INT32:
            return static_cast<BYTE>(*static_cast<const int*>(value));
        case MMC_TYPE_UINT32:
            return static_cast<BYTE>(*static_cast<const unsigned int*>(value));
        case MMC_TYPE_INT64:
            return static_cast<BYTE>(*static_cast<const INT64*>(value));
        case MMC_TYPE_UINT64:
            return static_cast<BYTE>(*static_cast<const UINT64*>(value));
        case MMC_TYPE_BYTE:
            return *static_cast<const unsigned char*>(value);
        case MMC_TYPE_BOOL:
            return static_cast<BYTE>(*static_cast<const bool*>(value));
        case MMC_TYPE_FLOAT:
            return static_cast<BYTE>(*static_cast<const float*>(value));
        case MMC_TYPE_CSTR:
            return static_cast<BYTE>(vislib::CharTraitsA::ParseInt(static_cast<const char*>(value)));
        case MMC_TYPE_WSTR:
            return static_cast<BYTE>(vislib::CharTraitsW::ParseInt(static_cast<const wchar_t*>(value)));
        case MMC_TYPE_VOIDP:
            break;
        default:
            break;
    }
    throw vislib::IllegalParamException("Cannot convert value", __FILE__, __LINE__);
}


/*
 * megamol::core::utility::APIValueUtil::IsIntType
 */
bool megamol::core::utility::APIValueUtil::AsBool(mmcValueType type, const void* value) {
    switch (type) {
        case MMC_TYPE_INT32:
            return (*static_cast<const int*>(value) != 0);
        case MMC_TYPE_UINT32:
            return (*static_cast<const unsigned int*>(value) != 0);
        case MMC_TYPE_INT64:
            return (*static_cast<const INT64*>(value) != 0);
        case MMC_TYPE_UINT64:
            return (*static_cast<const UINT64*>(value) != 0);
        case MMC_TYPE_BYTE:
            return (*static_cast<const unsigned char*>(value) != 0);
        case MMC_TYPE_BOOL:
            return *static_cast<const bool*>(value);
        case MMC_TYPE_FLOAT:
            return vislib::math::IsEqual(*static_cast<const float*>(value), 0.0f);
        case MMC_TYPE_CSTR:
            return vislib::CharTraitsA::ParseBool(static_cast<const char*>(value));
        case MMC_TYPE_WSTR:
            return vislib::CharTraitsW::ParseBool(static_cast<const wchar_t*>(value));
        case MMC_TYPE_VOIDP:
            break;
        default:
            break;
    }
    throw vislib::IllegalParamException("Cannot convert value", __FILE__, __LINE__);
}


/*
 * megamol::core::utility::APIValueUtil::IsIntType
 */
vislib::StringA megamol::core::utility::APIValueUtil::AsStringA(mmcValueType type, const void* value) {
    vislib::StringA str;
    switch (type) {
            str.Format("%d", *static_cast<const int*>(value));
            return str;
        case MMC_TYPE_UINT32:
            str.Format("%u", *static_cast<const unsigned int*>(value));
            return str;
        case MMC_TYPE_INT64:
            str.Format("%d", static_cast<int>(*static_cast<const INT64*>(value)));
            return str;
        case MMC_TYPE_UINT64:
            str.Format("%u", static_cast<unsigned int>(*static_cast<const UINT64*>(value)));
            return str;
        case MMC_TYPE_BYTE:
            str.Format("%u", static_cast<unsigned int>(*static_cast<const unsigned char*>(value)));
            return str;
        case MMC_TYPE_BOOL:
            return (*static_cast<const bool*>(value)) ? "true" : "false";
        case MMC_TYPE_FLOAT:
            str.Format("%f", *static_cast<const float*>(value));
            return str;
        case MMC_TYPE_CSTR:
            return static_cast<const char*>(value);
        case MMC_TYPE_WSTR:
            return W2A(static_cast<const wchar_t*>(value));
        case MMC_TYPE_VOIDP:
            break;
        default:
            break;
    }
    throw vislib::IllegalParamException("Cannot convert value", __FILE__, __LINE__);
}


/*
 * megamol::core::utility::APIValueUtil::IsIntType
 */
vislib::StringW megamol::core::utility::APIValueUtil::AsStringW(mmcValueType type, const void* value) {
    vislib::StringW str;
    switch (type) {
        case MMC_TYPE_INT32:
            str.Format(L"%d", *static_cast<const int*>(value));
            return str;
        case MMC_TYPE_UINT32:
            str.Format(L"%u", *static_cast<const unsigned int*>(value));
            return str;
        case MMC_TYPE_INT64:
            str.Format(L"%d", static_cast<int>(*static_cast<const INT64*>(value)));
            return str;
        case MMC_TYPE_UINT64:
            str.Format(L"%u", static_cast<unsigned int>(*static_cast<const UINT64*>(value)));
            return str;
        case MMC_TYPE_BYTE:
            str.Format(L"%u", static_cast<unsigned int>(*static_cast<const unsigned char*>(value)));
            return str;
        case MMC_TYPE_BOOL:
            return (*static_cast<const bool*>(value)) ? L"true" : L"false";
        case MMC_TYPE_FLOAT:
            str.Format(L"%f", *static_cast<const float*>(value));
            return str;
        case MMC_TYPE_CSTR:
            return A2W(static_cast<const char*>(value));
        case MMC_TYPE_WSTR:
            return static_cast<const wchar_t*>(value);
        case MMC_TYPE_VOIDP:
            break;
        default:
            break;
    }
    throw vislib::IllegalParamException("Cannot convert value", __FILE__, __LINE__);
}


/*
 * megamol::core::utility::APIValueUtil::IsIntType
 */
bool megamol::core::utility::APIValueUtil::IsIntType(mmcValueType type) {
    return (type == MMC_TYPE_INT32)
        || (type == MMC_TYPE_UINT32)
        || (type == MMC_TYPE_INT64)
        || (type == MMC_TYPE_UINT64)
        || (type == MMC_TYPE_BYTE);
}


/*
 * megamol::core::utility::APIValueUtil::IsStringType
 */
bool megamol::core::utility::APIValueUtil::IsStringType(mmcValueType type) {
    return (type == MMC_TYPE_CSTR)
        || (type == MMC_TYPE_WSTR);
}

