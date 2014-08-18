/*
 * D3D9AdapterInformation.h
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_D3D9ADAPTERINFORMATION_H_INCLUDED
#define VISLIB_D3D9ADAPTERINFORMATION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <d3d9.h>

#include "vislib/AbstractD3DAdapterInformation.h"
#include "vislib/PtrArray.h"


namespace vislib {
namespace graphics {
namespace d3d {


    /**
     * This class wraps all information about the Direct3D 9 adapters attached
     * to the system and their outputs 
     */
    class D3D9AdapterInformation : public AbstractD3DAdapterInformation {

    public:

        static void GetAdapterInformation(
            vislib::PtrArray<D3D9AdapterInformation>& outAdapterInformation,
            IDirect3D9 *d3d);

        /** Ctor. */
        D3D9AdapterInformation(IDirect3D9 *d3d, const UINT adapterOrdinal);

        inline D3D9AdapterInformation(const D3D9AdapterInformation& rhs) {
            *this = rhs;
        }

        /** Dtor. */
        virtual ~D3D9AdapterInformation(void);

        inline UINT GetAdapterOrdinal(const SIZE_T outputIdx) {
            VLSTACKTRACE("D3D9AdapterInformation::GetAdapterOrdinal", 
                __FILE__,__LINE__);
            return this->GetDirect3DCapabilites(outputIdx).AdapterOrdinal;
        }

        const D3DCAPS9& GetDirect3DCapabilites(const SIZE_T outputIdx);

        virtual SIZE_T GetOutputCount(void) const;

        D3D9AdapterInformation& operator =(const D3D9AdapterInformation& rhs);

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
            const SIZE_T outputIdx) const;

    private:

        /** Superclass typedef. */
        typedef AbstractD3DAdapterInformation Super;

        /** This structure groups all information about a single output. */
        typedef struct Output_t {
            D3DCAPS9 d3dCaps;       //< The D3D capabilities
            MONITORINFOEXW monInfo; //< Info about the monitor attached.

            inline bool operator ==(const struct Output_t& rhs) const {
                return ((::memcmp(&this->d3dCaps, &rhs.d3dCaps, 
                    sizeof(this->d3dCaps)) == 0)
                    && (::memcmp(&this->monInfo, &rhs.monInfo, 
                    sizeof(this->monInfo)) == 0));
            }

            inline struct Output_t& operator =(const struct Output_t& rhs) {
                ::memcpy(&this->d3dCaps, &rhs.d3dCaps, sizeof(this->d3dCaps));
                ::memcpy(&this->monInfo, &rhs.monInfo, sizeof(this->monInfo));
                return *this;
            }
        } Output;

        /** The actual information we need about an adapter. */
        vislib::Array<Output> infos;

    };
    
} /* end namespace d3d */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_D3D9ADAPTERINFORMATION_H_INCLUDED */

