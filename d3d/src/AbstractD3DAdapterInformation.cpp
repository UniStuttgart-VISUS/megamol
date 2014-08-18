/*
 * AbstractD3DAdapterInformation.cpp
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/AbstractD3DAdapterInformation.h"

#include "vislib/OutOfRangeException.h"


using namespace vislib::graphics::d3d;


/*
 * ...::d3d::AbstractD3DAdapterInformation::~AbstractD3DAdapterInformation
 */
AbstractD3DAdapterInformation::~AbstractD3DAdapterInformation(void) {
    VLSTACKTRACE("AbstractD3DAdapterInformation::"
        "~AbstractD3DAdapterInformation", __FILE__, __LINE__);
}


/*
 * ...::graphics::d3d::AbstractD3DAdapterInformation::INVALID_OUTPUT_IDX
 */
const INT_PTR AbstractD3DAdapterInformation::INVALID_OUTPUT_IDX = -1;


/*
 * ...::d3d::AbstractD3DAdapterInformation::FindOutputIdxForDeviceName
 */
INT_PTR AbstractD3DAdapterInformation::FindOutputIdxForDeviceName(
        const vislib::StringW& deviceName) const {
    VLSTACKTRACE("AbstractD3DAdapterInformation::FindOutputIdxForDeviceName",
        __FILE__, __LINE__);
    SIZE_T cntOutputs = this->GetOutputCount();

    for (SIZE_T i = 0; i < cntOutputs; i++) {
        if (deviceName.Equals(this->GetDeviceName(i))) {
            return static_cast<INT_PTR>(i);
        }
    }
    /* Not found. */

    return INVALID_OUTPUT_IDX;
}


/*
 * ...::d3d::AbstractD3DAdapterInformation::GetDesktopCoordinates
 */
vislib::math::Rectangle<LONG>& 
AbstractD3DAdapterInformation::GetDesktopCoordinates(
        vislib::math::Rectangle<LONG>& outDesktopCoordinates,
        const SIZE_T outputIdx) const {
    VLSTACKTRACE("AbstractD3DAdapterInformation::GetDesktopCoordinates",
        __FILE__, __LINE__);

    const MONITORINFOEXW& mi = this->getMonitorInfo(outputIdx);
    ASSERT(mi.cbSize >= sizeof(MONITORINFOEXW));

    outDesktopCoordinates.Set(mi.rcMonitor.left, mi.rcMonitor.bottom,
        mi.rcMonitor.right, mi.rcMonitor.top);
    return outDesktopCoordinates;
}


/*
 * ...::d3d::AbstractD3DAdapterInformation::GetDeviceName
 */
vislib::StringW AbstractD3DAdapterInformation::GetDeviceName(
        const SIZE_T outputIdx) const {
    VLSTACKTRACE("AbstractD3DAdapterInformation::GetDeviceName",
        __FILE__, __LINE__);

    const MONITORINFOEXW& mi = this->getMonitorInfo(outputIdx);
    ASSERT(mi.cbSize >= sizeof(MONITORINFOEXW));
    return StringW(mi.szDevice);
}


/*
 * ...::d3d::AbstractD3DAdapterInformation::IsPrimaryDisplay
 */
bool AbstractD3DAdapterInformation::IsPrimaryDisplay(
        const SIZE_T outputIdx) const {
    VLSTACKTRACE("AbstractD3DAdapterInformation::IsPrimaryDisplay",
        __FILE__, __LINE__);

    const MONITORINFOEXW& mi = this->getMonitorInfo(outputIdx);
    ASSERT(mi.cbSize >= sizeof(MONITORINFOEXW));
    return ((mi.dwFlags & MONITORINFOF_PRIMARY) != 0);
}


/*
 * ...::d3d::AbstractD3DAdapterInformation::AbstractD3DAdapterInformation
 */
AbstractD3DAdapterInformation::AbstractD3DAdapterInformation(void) {
    VLSTACKTRACE("AbstractD3DAdapterInformation::"
        "AbstractD3DAdapterInformation", __FILE__, __LINE__);
}
