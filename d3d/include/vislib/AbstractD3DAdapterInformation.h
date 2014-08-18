/*
 * AbstractD3DAdapterInformation.h
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTD3DADAPTERINFORMATION_H_INCLUDED
#define VISLIB_ABSTRACTD3DADAPTERINFORMATION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Rectangle.h"
#include "vislib/StackTrace.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"


namespace vislib {
namespace graphics {
namespace d3d {


    /**
     * This class is the superclass for Direct3D 9 or DXGI specific containers
     * that wrap all information about the display subsystem.
     */
    class AbstractD3DAdapterInformation {

    public:

        /** This constant marks an invalid output index. */
        static const INT_PTR INVALID_OUTPUT_IDX;

        /** Dtor. */
        virtual ~AbstractD3DAdapterInformation(void);

        /**
         * Answer the index of the output that has the display device with the
         * specified name attached. If no such device is attached to any of the
         * adapter's outputs, INVALID_OUTPUT_IDX is returned.
         *
         * @param deviceName The name of the output device to be searched.
         *
         * @return The index of the output having the specified device attached
         *         or INVALID_OUTPUT_IDX in case such a device was not found.
         */
        virtual INT_PTR FindOutputIdxForDeviceName(
            const StringW& deviceName) const;

        /**
         * Answer the index of the output that has the display device with the
         * specified name attached. If no such device is attached to any of the
         * adapter's outputs, INVALID_OUTPUT_IDX is returned.
         *
         * @param deviceName The name of the output device to be searched.
         *
         * @return The index of the output having the specified device attached
         *         or INVALID_OUTPUT_IDX in case such a device was not found.
         */
        inline INT_PTR FindOutputIdxForDeviceName(
                const StringA& deviceName) const {
            VLSTACKTRACE("AbstractD3DAdapterInformation::"
                "FindOutputIdxForDeviceName", __FILE__, __LINE__);
            return this->FindOutputIdxForDeviceName(
                A2W(deviceName.PeekBuffer()));
        }

        /**
         * Answer the desktop coordinates of the 'outputIdx'th output of this
         * adapter.
         *
         * @param outDesktopCoordinates Receives the coordinates of the monitor
         *                              rectangle of the 'outputIdx'th output of
         *                              this adapter.
         * @param outputIdx             The number of the output to be retrieved.
         *                              Must be within [0, GetOutputCount()[.
         *
         * @return A reference to 'outDesktopCoordinates'.
         * 
         * @throws OutOfRangeException If 'outputIdx' does not designate a valid
         *                             output attached to the adapter.
         */
        virtual math::Rectangle<LONG>& GetDesktopCoordinates(
            math::Rectangle<LONG>& outDesktopCoordinates,
            const SIZE_T outputIdx) const;

        /**
         * Answer the name of the device that is attached to the 'outputIdx'th 
         * output of this adapter.
         *
         * @param outputIdx The number of the output to get the device name for.
         *                  Must be within [0, GetOutputCount()[.
         *
         * @return
         *
         * @throws OutOfRangeException If 'outputIdx' does not designate a valid
         *                             output attached to the adapter.
         */
        virtual StringW GetDeviceName(const SIZE_T outputIdx) const;

        /**
         * Answer the number of outputs this adapter has.
         *
         * @return The number of outputs this adapter has.
         */
        virtual SIZE_T GetOutputCount(void) const = 0;

        //virtual math::Rectangle<LONG>& GetDisplayWorkArea(
        //    math::Rectangle<LONG> outWorkArea,
        //    const SIZE_T outputIdx) const = 0;

        /**
         * Answer whether the device attached to the 'outputIdx'th output of 
         * this adapter is the primary display of the desktop.
         *
         * @return
         *
         * @throws OutOfRangeException If 'outputIdx' does not designate a valid
         *                             output attached to the adapter.
         */
        virtual bool IsPrimaryDisplay(const SIZE_T outputIdx) const;

    protected:

        /**
         * Answer MONITORINFOEXW for the display attached to the 'outputIdx'th
         * output.
         *
         * @param outputIdx Index of the output to retrieve the monitor 
         *                  description for.
         * 
         * @return Reference to the monitor description. The value designated 
         *         must live as long as this object lives.
         *
         * @throws OutOfRangeException If 'outputIdx' does not designate a valid
         *                             output attached to the adapter.
         */
        virtual const MONITORINFOEXW& getMonitorInfo(
            const SIZE_T outputIdx) const = 0;

        /** Ctor. */
        AbstractD3DAdapterInformation(void);

    };

} /* end namespace d3d */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTD3DADAPTERINFORMATION_H_INCLUDED */
