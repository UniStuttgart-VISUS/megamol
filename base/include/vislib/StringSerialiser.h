/*
 * StringSerialiser.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_STRINGSERIALISER_H_INCLUDED
#define VISLIB_STRINGSERIALISER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Array.h"
#include "vislib/NoSuchElementException.h"
#include "vislib/Pair.h"
#include "vislib/Serialiser.h"
#include "vislib/String.h"
#include "vislib/StringTokeniser.h"


namespace vislib {

#ifdef _WIN32
#define _I64_PRINTF "I64"
#else /* _WIN32 */
#define _I64_PRINTF "ll"
#endif /* _WIN32 */


    /**
     * Implementation of serialiser operating on a string. This serialiser uses
     * the names and the order, if no name is specified, of the elements to
     * serialise or deserialise 'Serialisable' objects. Be aware of the fact
     * that the strings might be copied often and that this serialiser might be
     * very slow because of this.
     */
    template <class T> class StringSerialiser : public Serialiser {
    public:

        /**
         * Ctor.
         *
         * @param str The default value of the string. This is usually used
         *            for deserialisation. To serialise data, leave this 
         *            parameter 'NULL'.
         */
        StringSerialiser(const typename T::Char *str = NULL);

        /**
         * Ctor.
         *
         * @param str The value of the string. This is used for 
         *            deserialisation.
         */
        StringSerialiser(const String<T>& str);

        /** Dtor. */
        virtual ~StringSerialiser(void);

        /**
         * Clears the internal data buffer of the serialiser.
         */
        inline void ClearData(void) {
            this->SetInputString(NULL);
        }

        virtual void Deserialise(bool& outValue, 
                const char *name = NULL) {
            outValue = T::ParseBool(this->value(name));
        }

        virtual void Deserialise(bool& outValue, 
                const wchar_t *name) {
            outValue = T::ParseBool(this->value(name));
        }

        virtual void Deserialise(wchar_t& outValue, 
                const char *name = NULL) {
            unsigned int u;
            this->Deserialise(u, name);
            outValue = static_cast<wchar_t>(u);
        }

        virtual void Deserialise(wchar_t& outValue, 
                const wchar_t *name) {
            unsigned int u;
            this->Deserialise(u, name);
            outValue = static_cast<wchar_t>(u);
        }

        virtual void Deserialise(INT8& outValue, 
                const char *name = NULL) {
            outValue = static_cast<INT8>(T::ParseInt(this->value(name)));
        }

        virtual void Deserialise(INT8& outValue, 
                const wchar_t *name) {
            outValue = static_cast<INT8>(T::ParseInt(this->value(name)));
        }

        virtual void Deserialise(UINT8& outValue, 
                const char *name = NULL) {
            outValue = static_cast<UINT8>(T::ParseInt(this->value(name)));
        }

        virtual void Deserialise(UINT8& outValue, 
                const wchar_t *name) {
            outValue = static_cast<UINT8>(T::ParseInt(this->value(name)));
        }

        virtual void Deserialise(INT16& outValue, 
                const char *name = NULL) {
            outValue = static_cast<INT16>(T::ParseInt(this->value(name)));
        }

        virtual void Deserialise(INT16& outValue, 
                const wchar_t *name) {
            outValue = static_cast<INT16>(T::ParseInt(this->value(name)));
        }

        virtual void Deserialise(UINT16& outValue, 
                const char *name = NULL) {
            outValue = static_cast<UINT16>(T::ParseInt(this->value(name)));
        }

        virtual void Deserialise(UINT16& outValue, 
                const wchar_t *name) {
            outValue = static_cast<UINT16>(T::ParseInt(this->value(name)));
        }

        virtual void Deserialise(INT32& outValue, 
                const char *name = NULL) {
            outValue = static_cast<INT32>(T::ParseInt(this->value(name)));
        }

        virtual void Deserialise(INT32& outValue, 
                const wchar_t *name) {
            outValue = static_cast<INT32>(T::ParseInt(this->value(name)));
        }

        virtual void Deserialise(UINT32& outValue, 
                const char *name = NULL) {
            outValue = static_cast<UINT32>(T::ParseUInt64(this->value(name)));
        }

        virtual void Deserialise(UINT32& outValue, 
                const wchar_t *name) {
            outValue = static_cast<UINT32>(T::ParseUInt64(this->value(name)));
        }

        virtual void Deserialise(INT64& outValue, 
                const char *name = NULL) {
            outValue = T::ParseInt64(this->value(name));
        }

        virtual void Deserialise(INT64& outValue, 
                const wchar_t *name) {
            outValue = T::ParseInt64(this->value(name));
        }

        virtual void Deserialise(UINT64& outValue, 
                const char *name = NULL) {
            outValue = T::ParseUInt64(this->value(name));
        }

        virtual void Deserialise(UINT64& outValue, 
                const wchar_t *name) {
            outValue = T::ParseUInt64(this->value(name));
        }

        virtual void Deserialise(float& outValue, 
                const char *name = NULL) {
            outValue = static_cast<float>(T::ParseDouble(this->value(name)));
        }

        virtual void Deserialise(float& outValue, 
                const wchar_t *name) {
            outValue = static_cast<float>(T::ParseDouble(this->value(name)));
        }

        virtual void Deserialise(double& outValue, 
                const char *name = NULL) {
            outValue = T::ParseDouble(this->value(name));
        }

        virtual void Deserialise(double& outValue, 
                const wchar_t *name) {
            outValue = T::ParseDouble(this->value(name));
        }

        virtual void Deserialise(StringA& outValue, 
                const char *name = NULL) {
            outValue = vislib::StringA(this->value(name));
        }

        virtual void Deserialise(StringA& outValue, 
                const wchar_t *name) {
            outValue = vislib::StringA(this->value(name));
        }

        virtual void Deserialise(StringW& outValue, 
                const char *name = NULL) {
            outValue = vislib::StringW(this->value(name));
        }

        virtual void Deserialise(StringW& outValue, 
                const wchar_t *name) {
            outValue = vislib::StringW(this->value(name));
        }

        virtual void Serialise(const bool value, 
                const char *name = NULL) {
            this->Serialise(vislib::StringA(value ? "1" : "0"), name);
        }

        virtual void Serialise(const bool value, 
                const wchar_t *name) {
            this->Serialise(vislib::StringA(value ? "1" : "0"), name);
        }

        virtual void Serialise(const wchar_t value,
                const char *name = NULL) {
            this->Serialise(static_cast<unsigned int>(value), name);
        }

        virtual void Serialise(const wchar_t value,
                const wchar_t *name) {
            this->Serialise(static_cast<unsigned int>(value), name);
        }

        virtual void Serialise(const INT8 value,
                const char *name = NULL) {
            this->Serialise(static_cast<int>(value), name);
        }

        virtual void Serialise(const INT8 value,
                const wchar_t *name) {
            this->Serialise(static_cast<int>(value), name);
        }

        virtual void Serialise(const UINT8 value,
                const char *name = NULL) {
            this->Serialise(static_cast<unsigned int>(value), name);
        }

        virtual void Serialise(const UINT8 value,
                const wchar_t *name) {
            this->Serialise(static_cast<unsigned int>(value), name);
        }

        virtual void Serialise(const INT16 value,
                const char *name = NULL) {
            this->Serialise(static_cast<int>(value), name);
        }

        virtual void Serialise(const INT16 value,
                const wchar_t *name) {
            this->Serialise(static_cast<int>(value), name);
        }

        virtual void Serialise(const UINT16 value,
                const char *name = NULL) {
            this->Serialise(static_cast<unsigned int>(value), name);
        }

        virtual void Serialise(const UINT16 value,
                const wchar_t *name) {
            this->Serialise(static_cast<unsigned int>(value), name);
        }

        virtual void Serialise(const INT32 value,
                const char *name = NULL) {
            vislib::StringA s;
            s.Format("%d", value);
            this->Serialise(s, name);
        }

        virtual void Serialise(const INT32 value,
                const wchar_t *name) {
            vislib::StringA s;
            s.Format("%d", value);
            this->Serialise(s, name);
        }

        virtual void Serialise(const UINT32 value,
                const char *name = NULL) {
            vislib::StringA s;
            s.Format("%u", value);
            this->Serialise(s, name);
        }

        virtual void Serialise(const UINT32 value,
                const wchar_t *name) {
            vislib::StringA s;
            s.Format("%u", value);
            this->Serialise(s, name);
        }

        virtual void Serialise(const INT64 value,
                const char *name = NULL) {
            vislib::StringA s;
            s.Format("%"_I64_PRINTF"d", value);
            this->Serialise(s, name);
        }

        virtual void Serialise(const INT64 value,
                const wchar_t *name) {
            vislib::StringA s;
            s.Format("%"_I64_PRINTF"d", value);
            this->Serialise(s, name);
        }

        virtual void Serialise(const UINT64 value,
                const char *name = NULL) {
            vislib::StringA s;
            s.Format("%"_I64_PRINTF"u", value);
            this->Serialise(s, name);
        }

        virtual void Serialise(const UINT64 value,
                const wchar_t *name) {
            vislib::StringA s;
            s.Format("%"_I64_PRINTF"u", value);
            this->Serialise(s, name);
        }

        virtual void Serialise(const float value,
                const char *name = NULL) {
            this->Serialise(static_cast<double>(value), name);
        }

        virtual void Serialise(const float value,
                const wchar_t *name) {
            this->Serialise(static_cast<double>(value), name);
        }

        virtual void Serialise(const double value,
                const char *name = NULL) {
            vislib::StringA s;
            s.Format("%.32g", value);
            this->Serialise(s, name);
        }

        virtual void Serialise(const double value,
                const wchar_t *name) {
            vislib::StringA s;
            s.Format("%.32g", value);
            this->Serialise(s, name);
        }

        virtual void Serialise(const StringA& value,
                const char *name = NULL) {
            if (name != NULL) {
                vislib::String<T> tname(name);
                for (unsigned int i = 0; i < this->data.Count(); i++) {
                    if (this->data[i].Key().Equals(tname)) {
                        this->data[i].Second() = vislib::String<T>(value);
                        return;
                    }
                }
                this->data.Append(
                    vislib::Pair<vislib::String<T>, vislib::String<T> >(
                    vislib::String<T>(name), vislib::String<T>(value)));
            } else {
                this->data.Append(
                    vislib::Pair<vislib::String<T>, vislib::String<T> >(
                    NULL, vislib::String<T>(value)));
            }
        }

        virtual void Serialise(const StringA& value,
                const wchar_t *name) {
            vislib::String<T> tname(name);
            for (unsigned int i = 0; i < this->data.Count(); i++) {
                if (this->data[i].Key().Equals(tname)) {
                    this->data[i].Second() = vislib::String<T>(value);
                    return;
                }
            }
            this->data.Append(
                vislib::Pair<vislib::String<T>, vislib::String<T> >(
                vislib::String<T>(name), vislib::String<T>(value)));
        }

        virtual void Serialise(const StringW& value,
                const char *name = NULL) {
            if (name != NULL) {
                vislib::String<T> tname(name);
                for (unsigned int i = 0; i < this->data.Count(); i++) {
                    if (this->data[i].Key().Equals(tname)) {
                        this->data[i].Second() = vislib::String<T>(value);
                        return;
                    }
                }
                this->data.Append(
                    vislib::Pair<vislib::String<T>, vislib::String<T> >(
                    vislib::String<T>(name), vislib::String<T>(value)));
            } else {
                this->data.Append(
                    vislib::Pair<vislib::String<T>, vislib::String<T> >(
                    NULL, vislib::String<T>(value)));
            }
        }

        virtual void Serialise(const StringW& value,
                const wchar_t *name) {
            vislib::String<T> tname(name);
            for (unsigned int i = 0; i < this->data.Count(); i++) {
                if (this->data[i].Key().Equals(tname)) {
                    this->data[i].Second() = vislib::String<T>(value);
                    return;
                }
            }
            this->data.Append(
                vislib::Pair<vislib::String<T>, vislib::String<T> >(
                vislib::String<T>(name), vislib::String<T>(value)));
        }

        /**
         * Parses an input string to the internal representation. This method
         * is usually used for deserialisation.
         *
         * @param str The input string.
         */
        void SetInputString(const typename T::Char *str);

        /**
         * Parses an input string to the internal representation. This method
         * is usually used for deserialisation.
         *
         * @param str The input string.
         */
        inline void SetInputString(const String<T> &str) {
            this->SetInputString(str);
        }

        /**
         * Sets the position of the next element to be deserialised if driven
         * by order. Usually this method is only used to jump to the first
         * serialised element to restart the deserialisation process.
         *
         * @param i The index of the next element to be deserialised.
         */
        void SetNextDeserialisePosition(unsigned int i = 0) {
            this->nextDePos = 0;
        }

        /**
         * Stores the internal data in a string. This method is usually used
         * after serialisation to receive the resulting string.
         *
         * @param outStr The string receiving the serialised data.
         */
        void GetString(String<T> &outStr) const;

        /**
         * Returns a string of the internal data. This method is usually used
         * after serialisation to receive the resulting string.
         *
         * @return The string holding the serialised data.
         */
        inline String<T> GetString(void) const {
            vislib::String<T> tmp;
            this->GetString(tmp);
            return tmp;
        }

    private:

        /**
         * Answers the value for a given name or the next value to be used for
         * deserialisation. As side effect 'nextDePos' is set to the next
         * element.
         *
         * @param name The name of the value to return.
         *
         * @return The found value.
         *
         * @throws Exception in case of an error.
         */
        inline const vislib::String<T>& value(const char *name) {
            if (name != NULL) {
                vislib::String<T> tname(name);
                for (unsigned int i = 0; i < this->data.Count(); i++) {
                    if (this->data[i].Key().Equals(tname)) {
                        this->nextDePos = i + 1;
                        return this->data[i].Value();
                    }
                }
            } else if (this->nextDePos < this->data.Count()) {
                return this->data[this->nextDePos++].Value();
            }
            throw NoSuchElementException("deserialisation failed", 
                __FILE__, __LINE__);
        }

        /**
         * Answers the value for a given name or the next value to be used for
         * deserialisation. As side effect 'nextDePos' is set to the next
         * element.
         *
         * @param name The name of the value to return.
         *
         * @return The found value.
         *
         * @throws Exception in case of an error.
         */
        inline const vislib::String<T>& value(const wchar_t *name) {
            if (name != NULL) {
                vislib::String<T> tname(name);
                for (unsigned int i = 0; i < this->data.Count(); i++) {
                    if (this->data[i].Key().Equals(tname)) {
                        this->nextDePos = i + 1;
                        return this->data[i].Value();
                    }
                }
            } else if (this->nextDePos < this->data.Count()) {
                return this->data[this->nextDePos++].Value();
            }
            throw NoSuchElementException("deserialisation failed", 
                __FILE__, __LINE__);
        }

        /** The internal data structure */
        vislib::Array<vislib::Pair<vislib::String<T>, vislib::String<T> > > 
            data;

        /** 
         * The position of the next value to be deserialised, used for order
         * driven deserialisation.
         */
        unsigned int nextDePos;

    };


    /*
     * StringSerialiser<T>::StringSerialiser
     */
    template<class T>
    StringSerialiser<T>::StringSerialiser(const typename T::Char *str)
            : Serialiser(SERIALISER_SUPPORTS_NAMES), data(), nextDePos(0) {
        this->SetInputString(str);
    }


    /*
     * StringSerialiser<T>::StringSerialiser
     */
    template<class T>
    StringSerialiser<T>::StringSerialiser(const vislib::String<T>& str)
            : Serialiser(SERIALISER_SUPPORTS_NAMES), data(), nextDePos(0) {
        this->SetInputString(str.PeekBuffer());
    }


    /*
     * StringSerialiser<T>::~StringSerialiser
     */
    template<class T> StringSerialiser<T>::~StringSerialiser(void) {
        // intentionally empty
    }


    /*
     * StringSerialiser<T>::parseInitString
     */
    template<class T>
    void StringSerialiser<T>::SetInputString(const typename T::Char *str) {
        this->data.Clear();
        this->nextDePos = 0;

        if ((str == NULL) || (*str == static_cast<typename T::Char>(0))) {
            return;
        }

        vislib::String<T> key;
        vislib::String<T> value;
        vislib::StringTokeniser<T> tokeniser(str, 
            static_cast<typename T::Char>('\n'));

        while (tokeniser.HasNext()) {
            const vislib::String<T>& line = tokeniser.Next();
            typename vislib::String<T>::Size pos;

            do {
                pos = line.Find(static_cast<typename T::Char>('='));
            } while ((pos != vislib::String<T>::INVALID_POS) && (pos > 0)
                && (line[pos - 1] == static_cast<typename T::Char>('\\')));

            key = line.Substring(0, pos);
            value = line.Substring(pos + 1);

            key.UnescapeCharacters(static_cast<typename T::Char>('\\'), 
                vislib::String<T>("\n\r="), vislib::String<T>("nr="));
            value.UnescapeCharacters(static_cast<typename T::Char>('\\'), 
                vislib::String<T>("\n\r="), vislib::String<T>("nr="));

            this->data.Append(
                vislib::Pair<vislib::String<T>, vislib::String<T> >(
                key, value));
        }
    }


    /*
     * StringSerialiser<T>::GetString
     */
    template<class T>
    void StringSerialiser<T>::GetString(vislib::String<T> &outStr) const {
        outStr.Clear();
        vislib::String<T> str;
        for (unsigned int i = 0; i < this->data.Count(); i++) {
            str = this->data[i].Key();
            str.EscapeCharacters(static_cast<typename T::Char>('\\'), 
                vislib::String<T>("\n\r="), vislib::String<T>("nr="));

            outStr.Append(str);
            outStr.Append(vislib::String<T>("="));
            str = this->data[i].Value();
            str.EscapeCharacters(static_cast<typename T::Char>('\\'), 
                vislib::String<T>("\n\r="), vislib::String<T>("nr="));

            outStr.Append(str);
            outStr.Append(vislib::String<T>("\n"));
        }
    }

#undef _I64_PRINTF

    /** Template instantiation for ANSI strings. */
    typedef StringSerialiser<CharTraitsA> StringSerialiserA;

    /** Template instantiation for wide strings. */
    typedef StringSerialiser<CharTraitsW> StringSerialiserW;

    /** Template instantiation for TCHARs. */
    typedef StringSerialiser<TCharTraits> TStringSerialiser;

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_STRINGSERIALISER_H_INCLUDED */
