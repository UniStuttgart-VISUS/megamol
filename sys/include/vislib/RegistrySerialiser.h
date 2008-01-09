/*
 * RegistrySerialiser.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_REGISTRYSERIALISER_H_INCLUDED
#define VISLIB_REGISTRYSERIALISER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifdef _WIN32

#include <windows.h>

#include "vislib/String.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {


    /**
     * TODO: comment class
     *
     * TODO: Think about a more general serialisation concept.
     */
    class RegistrySerialiser {

    public:

        /**
         * This class defines the interface of serialisable variables. It is 
         * abstract and only intended as superclass for integral, string, etc.
         * serialisable data wrappers.
         *
         * This is part of the "crowbar pattern".
         *
         * The template parameters must be the following:
         * C: A CharTrais implementation that is used for the value name 
         *    string.
         */
        template<class C> class Serialisable {

        public:

            /** The string type for the value names. */
            typedef String<C> String;

            /** Dtor. */
            virtual ~Serialisable(void);

            /**
             * Answer the name that is used for registry serialisation of the 
             * wrapped variable.
             *
             * @return The value name.
             */
            inline const String& GetName(void) const {
                return this->name;
            }

            /**
             * Change the serialisation name of the wrapped variable.
             *
             * @param name The new serialisation value name.
             */
            inline void SetName(const String& name) {
                this->name = name;
            }

        protected:

            /** 
             * Create a new instance with the specified value name.
             *
             * @param name The name of the value in the registry.
             */
            inline Serialisable(const String& name) : name(name) {}

            /**
             * Copy ctor.
             *
             * @param rhs The object to be cloned.
             */
            inline Serialisable(const Serialisable& rhs) : name(rhs.name) {}

            /**
             * Answer the size in bytes of the data.
             *
             * @return The size of the data in bytes.
             */
            virtual DWORD getSize(void) const = 0;

            /**
             * Answer the registry type that is to be used for serialisation.
             *
             * @return The registry type.
             */
            virtual DWORD getType(void) const = 0;

            /**
             * Answer a pointer to the data that are to be stored in the 
             * registry.
             *
             * @return A pointer to GetSize() bytes of data to be stored in the
             *         registry.
             */
            virtual BYTE *peekData(void) = 0;

            /**
             * Answer a pointer to the data that are to be stored in the 
             * registry.
             *
             * @return A pointer to GetSize() bytes of data to be stored in the
             *         registry.
             */
            virtual const BYTE *peekData(void) const = 0;

            /**
             * Assignment operator.
             *
             * THIS OPERATOR DOES NOT DO ANYTHING. THE VALUE NAME IS NOT PART OF
             * AN ASSIGNMENT!
             *
             * @param rhs The right hand side operand.
             *
             * @return *this.
             */
            inline Serialisable& operator =(const Serialisable& rhs) {
                // DO NOT COPY THE NAME!
                return *this;
            }

        private:

            /** The value name in the registry. */
            String name;

            /** The serialiser must access the data and the data size. */
            friend class RegistrySerialiser;

        }; /* end class Serialisable */


        /**
         * TODO documentation
         * 
         *
         * The template parameters must be the following:
         * T: 
         * U:
         * C: A CharTrais implementation that is used for the value name 
         *    string.
         * S:
         */
        template<class T, DWORD U, class C, DWORD S = sizeof(T)> 
        class GeneralSerialisable : public Serialisable<C> {

        public:

            /** 
             * Create a new instance with the specified value name.
             *
             * @param name The name of the value in the registry.
             */
            inline GeneralSerialisable(const String& name) : Super(name) {}

            inline GeneralSerialisable(const String& name, const T& data)
                : Super(name), data(data) {}

            /**
             * Copy ctor.
             *
             * @param rhs The object to be cloned.
             */
            inline GeneralSerialisable(const GeneralSerialisable& rhs) 
                : Super(rhs), data(rhs.data) {}

            /** Dtor. */
            virtual ~GeneralSerialisable(void);

            /**
             * Assignment operator.
             *
             * THE VALUE NAME IS NOT PART OF AN ASSIGNMENT!
             *
             * @param rhs The right hand side operand.
             *
             * @return *this.
             */
            inline GeneralSerialisable& operator =(
                    const GeneralSerialisable& rhs) {
                Super::operator =(rhs);
                this->data = rhs.data;
            }

            /**
             * Assignment operator.
             *
             * @param rhs The right hand side operand.
             *
             * @return *this.
             */
            inline Serialisable& operator =(const T& rhs) {
                this->data = rhs;
            }

            /**
             * Test for equality.
             *
             * THE VALUE NAME IS NOT PART OF THE COMPARISON!
             *
             * @param rhs The right hand side operand.
             *
             * @return true if the data are equal, false otherwise.
             */
            inline bool operator ==(const Serialisable& rhs) {
                return (this->data == rhs.data);
            }

            /**
             * Test for equality.
             *
             * @param rhs The right hand side operand.
             *
             * @return true if the data are equal, false otherwise.
             */
            inline bool operator ==(const T& rhs) {
                return (this->data == rhs);
            }

            /**
             * Test for inequality.
             *
             * THE VALUE NAME IS NOT PART OF THE COMPARISON!
             *
             * @param rhs The right hand side operand.
             *
             * @return true if the data are not equal, false otherwise.
             */
            inline bool operator !=(const Serialisable& rhs) {
                return (this->data != rhs.data);
            }

            /**
             * Test for inequality.
             *
             * @param rhs The right hand side operand.
             *
             * @return true if the data are not equal, false otherwise.
             */
            inline bool operator !=(const T& rhs) {
                return (this->data != rhs);
            }

            /**
             * Cast to internal data.
             *
             * @return A reference to the data.
             */
            inline operator T&(void) {
                return this->data;
            }

            /**
             * Cast to internal data.
             *
             * @return A reference to the data.
             */
            inline operator const T&(void) const {
                return this->data;
            }

        protected:

            /** Superclass typedef. */
            typedef Serialisable<C> Super;

            /**
             * Answer the size in bytes of the data.
             *
             * @return The size of the data in bytes.
             */
            virtual DWORD getSize(void) const;

            /**
             * Answer the registry type that is to be used for serialisation.
             *
             * @return The registry type.
             */
            virtual DWORD getType(void) const;

            /**
             * Answer a pointer to the data that are to be stored in the 
             * registry.
             *
             * @return A pointer to GetSize() bytes of data to be stored in the
             *         registry.
             */
            virtual BYTE *peekData(void);

            /**
             * Answer a pointer to the data that are to be stored in the 
             * registry.
             *
             * @return A pointer to GetSize() bytes of data to be stored in the
             *         registry.
             */
            virtual const BYTE *peekData(void) const;

            /** The actual piece of data. */
            T data;

        }; /* end class GeneralSerialisable */


        /** Serialisable specialisation for DWORD and ANSI charset name. */
        typedef GeneralSerialisable<DWORD, REG_DWORD, CharTraitsA> 
            SerialisableDwordA;

        /** Serialisable specialisation for DWORD and Unicode charset name. */
        typedef GeneralSerialisable<DWORD, REG_DWORD, CharTraitsW> 
            SerialisableDwordW;

        /** Serialisable specialisation for DWORD and TCHAR name. */
        typedef GeneralSerialisable<DWORD, REG_DWORD, TCharTraits>
            TSerialisableDword;

        /** Serialisable specialisation for int and ANSI charset name. */
        typedef GeneralSerialisable<float, REG_DWORD, CharTraitsA> 
            SerialisableFloatA;

        /** Serialisable specialisation for int and Unicode charset name. */
        typedef GeneralSerialisable<float, REG_DWORD, CharTraitsW> 
            SerialisableFloatW;

        /** Serialisable specialisation for DWORD and TCHAR name. */
        typedef GeneralSerialisable<float, REG_DWORD, TCharTraits>
            TSerialisableFloat;

        /** Serialisable specialisation for int and ANSI charset name. */
        typedef GeneralSerialisable<int, REG_DWORD, CharTraitsA> 
            SerialisableIntA;

        /** Serialisable specialisation for int and Unicode charset name. */
        typedef GeneralSerialisable<int, REG_DWORD, CharTraitsW> 
            SerialisableIntW;

        /** Serialisable specialisation for DWORD and TCHAR name. */
        typedef GeneralSerialisable<int, REG_DWORD, TCharTraits>
            TSerialisableInt;


        /**
         * TODO: documentation
         */
        RegistrySerialiser(const char *subKey, HKEY hKey = HKEY_CURRENT_USER);

        /**
         * TODO: documentation
         */
        RegistrySerialiser(const wchar_t *subKey, 
            HKEY hKey = HKEY_CURRENT_USER);

        /** Dtor. */
        ~RegistrySerialiser(void);

        /**
         * TODO: documentation
         */
        void Deserialise(Serialisable<CharTraitsA>& inOutSerialisable);

        /**
         * TODO: documentation
         */
        void Deserialise(Serialisable<CharTraitsW>& inOutSerialisable);

        /**
         * TODO: documentation
         */
        void Serialise(const Serialisable<CharTraitsA>& serialisable);

        /**
         * TODO: documentation
         */
        void Serialise(const Serialisable<CharTraitsW>& serialisable);

    private:

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        RegistrySerialiser(const RegistrySerialiser& rhs);

        /**
         * Forbidden assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throw IllegalParamException If (this != &rhs).
         */
        RegistrySerialiser& operator =(const RegistrySerialiser& rhs);

        /** Handle of the base key that is parent of the serialised elements. */
        HKEY hBaseKey;

    };

    /*
     * RegistrySerialiser::Serialisable<C>::~Serialisable
     */
    template<class C> RegistrySerialiser::Serialisable<C>::~Serialisable(void) {
    }


    /*
     * RegistrySerialiser::GeneralSerialisable<T, U, C, S>::~GeneralSerialisable
     */
    template<class T, DWORD U, class C, DWORD S> 
    RegistrySerialiser::GeneralSerialisable<T, U, C, S>::~GeneralSerialisable(
            void) {
    }


    /*
     * RegistrySerialiser::GeneralSerialisable<T, U, C, S>::getSize
     */
    template<class T, DWORD U, class C, DWORD S> 
    DWORD RegistrySerialiser::GeneralSerialisable<T, U, C, S>::getSize(
            void) const {
        return S;
    }

    /*
     * RegistrySerialiser::GeneralSerialisable<T, U, C, S>::getType
     */
    template<class T, DWORD U, class C, DWORD S> 
    DWORD RegistrySerialiser::GeneralSerialisable<T, U, C, S>::getType(
            void) const {
        return U;
    }


    /*
     * RegistrySerialiser::GeneralSerialisable<T, U, C, S>::peekData
     */
    template<class T, DWORD U, class C, DWORD S> 
    BYTE *RegistrySerialiser::GeneralSerialisable<T, U, C, S>::peekData(
            void) {
        return reinterpret_cast<BYTE *>(&this->data);
    }


    /*
     * RegistrySerialiser::GeneralSerialisable<T, U, C, S>::peekData
     */
    template<class T, DWORD U, class C, DWORD S> 
    const BYTE *RegistrySerialiser::GeneralSerialisable<T, U, C, S>::peekData(
            void) const {
        return reinterpret_cast<const BYTE *>(&this->data);
    }

} /* end namespace sys */
} /* end namespace vislib */

#endif /* _WIN32 */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_REGISTRYSERIALISER_H_INCLUDED */
