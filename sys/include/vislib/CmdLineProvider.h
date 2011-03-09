/*
 * CmdLineProvider.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CMDLINEPROVIDER_H_INCLUDED
#define VISLIB_CMDLINEPROVIDER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/CharTraits.h"
#include "vislib/String.h"

#ifdef _WIN32
#include "vislib/SystemException.h"
#endif // _WIN32


namespace vislib {
namespace sys {

    /* forward declaration of CmdLineParser */
    template<class T> class CmdLineParser;

    /**
     * Helper class to split a single string command line into an array of zero
     * terminated c strings.
     *
     * Instantiate with CharTraits as template parameter.
     */
    template<class T> class CmdLineProvider {
    public:
        /** 
         * CmdLineParser is friend because it's able to output internal data 
         * structures compatible with CmdLineProvider.
         *  <Tf> should only be used when equal with <T>
         */
        template<class Tf> friend class vislib::sys::CmdLineParser;

        /** Define a local name for the character type. */
        typedef typename T::Char Char;

        /**
         * Convenience method for creating a command line provider from 
         * standard main ANSI input parameters regardless of the type of the
         * command line parser.
         *
         * This is primarily intended for Linux where the Unicode support is 
         * sh*** and Unicode applications will get ANSI input.
         *
         * @param argc The number of elements in 'argv'.
         * @param argv The array of command line arguments. The first should be
         *             the application name.
         *
         * @return A template CmdLineProvider representing the specified command
         *         line arguments.
         */
        static CmdLineProvider Create(int argc, char **argv);

#ifdef _WIN32
        /**
         * Receives the modules file name of the current processes.
         * Available only under Windows operating systems.
         *
         * TODO: Move to somewhere more appropriate!
         *
         * @return The module file name.
         *
         * @throw SystemException if the module file name cannot be received
         *        due to a system error.
         */
        static String<T> GetModuleName(void);
#endif // _WIN32

        /** Ctor. */
        CmdLineProvider(void);

        /**
         * copy Ctor. 
         *
         * @param rhs The source object to deep copy the data from
         */
        CmdLineProvider(const CmdLineProvider& rhs);

        /**
         * Ctor with initialization
         * See: CreateCmdLine for further Information
         *
         * @param cmdLine The single string command line including the 
         *                application name.
         */
        CmdLineProvider(const Char *cmdLine);

        /**
         * Ctor with initialization
         * See: CreateCmdLine for further Information
         *
         * @param cmdLine The single string command line including the 
         *                application name.
         */
        CmdLineProvider(const String<T>& cmdLine);

        /**
         * Create command line provider from C style parameters.
         *
         * @param argc An integer that contains the count of arguments that 
         *             follow in 'argv'. 
         * @param argv An array of null-terminated strings representing 
         *             command-line arguments entered by the user of the program. 
         */
        CmdLineProvider(int argc, Char **argv);

        /**
         * Ctor with initialization
         * See: CreateCmdLine for further Information
         *
         * @param appName The application name.
         * @param cmdLine The single string command line excluding the 
         *                application name.
         */
        CmdLineProvider(const Char *appName, const Char *cmdLine);

        /**
         * Ctor with initialization
         * See: CreateCmdLine for further Information
         *
         * @param appName The application name.
         * @param cmdLine The single string command line excluding the 
         *                application name.
         */
        CmdLineProvider(const Char *appName, const String<T>& cmdLine);

        /**
         * Ctor with initialization
         * See: CreateCmdLine for further Information
         *
         * @param appName The application name.
         * @param cmdLine The single string command line excluding the 
         *                application name.
         */
        CmdLineProvider(const String<T>& appName, const Char *cmdLine);

        /**
         * Ctor with initialization
         * See: CreateCmdLine for further Information
         *
         * @param appName The application name.
         * @param cmdLine The single string command line excluding the 
         *                application name.
         */
        CmdLineProvider(const String<T>& appName, const String<T>& cmdLine);

        /** Dtor. */
        ~CmdLineProvider(void);

        /** 
         * assignment operator 
         *
         * @param rhs The source object to deep copy the data from
         *
         * @return A reference to this object.
         */
        CmdLineProvider& operator=(const CmdLineProvider& rhs);

        /**
         * Splits the provided single string command line into an array of 
         * strings. Each string represents one command line argument. The
         * strings are splitted at whitespace characters outside of double 
         * qouted sections. Single double qout characters will be removed, 
         * since they either start or end a qouted section. If the command line
         * ends within a double qouted section, the last parameter will begin
         * with a qout character.
         *
         * Remarks: Since single and double qout characters are important 
         * input in the following paragraphs, input command lines are placed 
         * in angle brackets < >.
         *
         * To produce as double qout character inside a double qouted section, 
         * escape the character by another double qout character 
         * (i.e.: "1 "" 2" will be parsed as a single parameter string holding
         * <1 " 2>).
         * 
         * if appName is not NULL, this string becomes the first command line 
         * argument, followed by the strings created from cmdLine. The 
         * splitted substrings are copied into the new array.
         *
         * @warning 
         *  When using windows powershell there will be problem when parsing
         *  command lines holding parameters containing escaped double qout
         *  characters. 
         *
         *  When using a single string command line provided by the 
         *  operating system (e.g. cmd.exe or powershell.exe calling a windows
         *  application using WinMain) there may be problems with escaped
         *  double qout characters. I.e. calling <prog.exe p1 "p2 `" p3" p4> in
         *  the windows power shell will produce the command line string 
         *  <p1 "p2 " p3" p4>, which does not hold any information that the 
         *  second double qout character was escaped. Therefore CreateCmdLine
         *  will produce a non intuitive result: {"p1", "p2 ", "p3 p4"}.
         *
         *  Windows power shell also provides the possibility to mark 
         *  parameters containing spaces using single qouts <'>. This must also
         *  be considered with care, since these will be repaced with qouble 
         *  qout characters by the power shell. I.e. calling <prog.exe '1 " 2'>
         *  will produce <prog.exe "1 " 2">, which will then be parsed into the
         *  following result: {"1 ", "2"}.
         *
         * @param appName The application name.
         * @param cmdLine The single string command line excluding the 
         *                application name.
         */
        void CreateCmdLine(const Char *appName, const Char *cmdLine);

        /**
         * Splits the provided single string command line into an array of 
         * strings. 
         * See: CreateCmdLine for further Information
         *
         * @param appName The application name.
         * @param cmdLine The single string command line excluding the 
         *                application name.
         */
        inline void CreateCmdLine(const String<T>& appName, const Char *cmdLine) {
            this->CreateCmdLine(appName.PeekBuffer(), cmdLine);
        }

        /**
         * Splits the provided single string command line into an array of 
         * strings. 
         * See: CreateCmdLine for further Information
         *
         * @param appName The application name.
         * @param cmdLine The single string command line excluding the 
         *                application name.
         */
        inline void CreateCmdLine(const Char *appName, const String<T>& cmdLine) {
            this->CreateCmdLine(appName, cmdLine.PeekBuffer());
        }

        /**
         * Splits the provided single string command line into an array of 
         * strings. 
         * See: CreateCmdLine for further Information
         *
         * @param appName The application name.
         * @param cmdLine The single string command line excluding the 
         *                application name.
         */
        inline void CreateCmdLine(const String<T>& appName, const String<T>& cmdLine) {
            this->CreateCmdLine(appName.PeekBuffer(), cmdLine.PeekBuffer());
        }

        /**
         * Splits the provided single string command line into an array of 
         * strings. 
         * See: CreateCmdLine for further Information
         *
         * @param cmdLine The single string command line including the 
         *                application name.
         */
        inline void CreateCmdLine(const Char *cmdLine) {
            this->CreateCmdLine(NULL, cmdLine);
        }

        /**
         * Splits the provided single string command line into an array of 
         * strings. 
         * See: CreateCmdLine for further Information
         *
         * @param cmdLine The single string command line including the 
         *                application name.
         */
        inline void CreateCmdLine(const String<T>& cmdLine) {
            this->CreateCmdLine(NULL, cmdLine.PeekBuffer());
        }

        /**
         * Returns a reference to the number of arguments in the arguemnt list.
         * The value should be changed by the caller to reflect any changes 
         * made to array of arguments returned by ArgV
         *
         * @return Reference to the number of arguments
         */
        inline int& ArgC(void) {
            return this->argCount;
        }

        /**
         * Returns the argument list as array of zero terminated strings. The
         * caller may change the values of the array of strings, but must not
         * free the memory the array values points to, even if the pointers
         * are lost, since the memory will be handled by the CmdLineProvider
         * object. 
         *
         * The value of the reference returned by ArgC should also be changed
         * to reflect any changes to the array returned by ArgV.
         *
         * @return The argument list. Might be NULL, if ArgC returned Zero.
         */
        inline Char **ArgV(void) {
            return this->arguments;
        }

        /**
         * Appends an argument to the command line.
         *
         * @param arg The argument to be appenden to the command line.
         */
        void Append(const Char *arg);

        /**
         * Appends an argument to the command line.
         *
         * @param arg The argument to be appenden to the command line.
         */
        inline void Append(const String<T>& arg) {
            this->Append(arg.PeekBuffer());
        }

        /**
         * Prepends an argument to the command line.
         *
         * @param arg The argument to be prependen to the command line.
         */
        void Prepend(const Char *arg);

        /**
         * Prepends an argument to the command line.
         *
         * @param arg The argument to be prependen to the command line.
         */
        void Prepend(const String<T>& arg) {
            this->Prepend(arg.PeekBuffer());
        }

        /**
         * Generates and returns a single string containing the whole command
         * line. Calling "CreateCmdList()" with this string, will generate an
         * internal data structure equal to the one of this object.
         *
         * This member works with the same variables which are returned by
         * ArgC and ArgV. So if the caller has changes these values in any
         * inconsitent way, the return value of this member is undefined.
         *
         * @param includeFirst Flag whether to include the first argument of
         *                     the argument list. Useful to create a single
         *                     string command line excluding the application
         *                     name from a argument list containing the
         *                     application name.
         *
         * @return The generated single string command line.
         */
        String<T> SingleStringCommandLine(bool includeFirst);

    private:

        /** clears the argument list */
        void clearArgumentList(void);

        /**
         * Creates an argument from two given pointers.
         *
         * @param left The left pointer.
         * @param right The right pointer.
         *
         * @return The new created argument string.
         */
        Char *createArgument(Char *left, Char *right);

        /**
         * Reflects all changes done to the pointers returned by 'ArgC' and 
         * 'ArgV' by updating 'storeCount' and 'memoryAnchor'.
         */
        void reflectCallerChanges(void);

        /** 
         * Number of created arguments. This is not unsigned to be more 
         * compliant to the specification of "main".
         */
        int argCount;

        /** list of created arguments */
        Char **arguments;

        /**
         * Number of created arguments. This is not unsigned to be more 
         * compliant to the specification of "main". This member is
         * necessary to reflect changes to the argument pointer list returned 
         * by ArgV().
         */
        int storeCount;

        /** 
         * memory anchor of the list of created arguments. This member is 
         * necessary to aviod memory leaks if the argument pointer list 
         * returned by ArgV() is changed.
         */
        Char **memoryAnchor[2];

    };


    /*
     * vislib::sys::CmdLineProvider<T>::Create
     */
    template<class T>
    CmdLineProvider<T> CmdLineProvider<T>::Create(int argc, char **argv) {
        // TODO: This is not very efficient.
        String<T> tmpLine;

        for (int i = 0; i < argc; i++) {
            tmpLine += String<T>(argv[i]);
            tmpLine += static_cast<Char>(' ');
        }

        return CmdLineProvider(tmpLine.PeekBuffer());
    }


#ifdef _WIN32
    /*
     * CmdLineProvider<T>::GetModuleName
     */
    template<class T> String<T> CmdLineProvider<T>::GetModuleName(void) {
        String<T> str;
        DWORD len = 16;
        DWORD rlen;
        Char *buf = NULL;

        while(true) {
            buf = str.AllocateBuffer(len);

            if (T::CharSize() == sizeof(wchar_t)) { // not too secure, but it will do
                rlen = GetModuleFileNameW(NULL, reinterpret_cast<wchar_t*>(buf), len); // unicode string
            } else {
                rlen = GetModuleFileNameA(NULL, reinterpret_cast<char*>(buf), len); // ansi string
            }

            if (rlen == 0) {
                throw SystemException(__FILE__, __LINE__);
            }

            if (rlen < len) break;
            len *= 2;
        }

        return str;
    }

#endif


    /*
     * CmdLineProvider<T>::CmdLineProvider
     */
    template<class T> CmdLineProvider<T>::CmdLineProvider() : argCount(0), arguments(NULL), storeCount(0)/*, memoryAnchor(NULL)*/ {
        this->memoryAnchor[0] = NULL;
        this->memoryAnchor[1] = NULL;
    }


    /*
     * CmdLineProvider<T>::CmdLineProvider
     */
    template<class T> CmdLineProvider<T>::CmdLineProvider(const CmdLineProvider& rhs) 
            : argCount(0), arguments(NULL), storeCount(0)/*, memoryAnchor(NULL)*/ {
        this->memoryAnchor[0] = NULL;
        this->memoryAnchor[1] = NULL;
        (*this) = rhs; // call assignment operator
    }


    /*
     * CmdLineProvider<T>::CmdLineProvider
     */
    template<class T> CmdLineProvider<T>::CmdLineProvider(const Char *cmdLine) 
            : argCount(0), arguments(NULL), storeCount(0)/*, memoryAnchor(NULL)*/ {
        this->memoryAnchor[0] = NULL;
        this->memoryAnchor[1] = NULL;
        this->CreateCmdLine(NULL, cmdLine);
    }


    /*
     * CmdLineProvider<T>::CmdLineProvider
     */
    template<class T> CmdLineProvider<T>::CmdLineProvider(const String<T>& cmdLine) 
            : argCount(0), arguments(NULL), storeCount(0)/*, memoryAnchor(NULL)*/ {
        this->memoryAnchor[0] = NULL;
        this->memoryAnchor[1] = NULL;
        this->CreateCmdLine(NULL, cmdLine.PeekBuffer());
    }


    /*
     * vislib::sys::CmdLineProvider<T>::CmdLineProvider
     */
    template<class T> 
    CmdLineProvider<T>::CmdLineProvider(int argc, Char **argv) {
        this->argCount = this->storeCount = argc;
        this->memoryAnchor[0] = new Char *[this->storeCount];
        this->memoryAnchor[1] = new Char *[this->argCount];
        this->arguments = this->memoryAnchor[1];

        BECAUSE_I_KNOW(this->argCount == this->storeCount);
        for (int i = 0; i < this->argCount; i++) {
            typename T::Size len = T::SafeStringLength(argv[i]) + 1;
            this->arguments[i] = this->memoryAnchor[0][i] = new Char[len];
            ::memcpy(this->memoryAnchor[0][i], argv[i], len * T::CharSize());
        }
    }


    /*
     * CmdLineProvider<T>::CmdLineProvider
     */
    template<class T> CmdLineProvider<T>::CmdLineProvider(const Char *appName, const Char *cmdLine) 
            : argCount(0), arguments(NULL), storeCount(0)/*, memoryAnchor(NULL)*/ {
        this->memoryAnchor[0] = NULL;
        this->memoryAnchor[1] = NULL;
        this->CreateCmdLine(appName, cmdLine);
    }


    /*
     * CmdLineProvider<T>::CmdLineProvider
     */
    template<class T> CmdLineProvider<T>::CmdLineProvider(const Char *appName, const String<T>& cmdLine) 
            : argCount(0), arguments(NULL), storeCount(0)/*, memoryAnchor(NULL)*/ {
        this->memoryAnchor[0] = NULL;
        this->memoryAnchor[1] = NULL;
        this->CreateCmdLine(appName, cmdLine.PeekBuffer());
    }


    /*
     * CmdLineProvider<T>::CmdLineProvider
     */
    template<class T> CmdLineProvider<T>::CmdLineProvider(const String<T>& appName, const Char *cmdLine) 
            : argCount(0), arguments(NULL), storeCount(0)/*, memoryAnchor(NULL)*/ {
        this->memoryAnchor[0] = NULL;
        this->memoryAnchor[1] = NULL;
        this->CreateCmdLine(appName.PeekBuffer(), cmdLine);
    }


    /*
     * CmdLineProvider<T>::CmdLineProvider
     */
    template<class T> CmdLineProvider<T>::CmdLineProvider(const String<T>& appName, const String<T>& cmdLine) 
            : argCount(0), arguments(NULL), storeCount(0)/*, memoryAnchor(NULL)*/ {
        this->memoryAnchor[0] = NULL;
        this->memoryAnchor[1] = NULL;
        this->CreateCmdLine(appName.PeekBuffer(), cmdLine.PeekBuffer());
    }


    /*
     * CmdLineProvider<T>::~CmdLineProvider
     */
    template<class T> CmdLineProvider<T>::~CmdLineProvider() {
        this->clearArgumentList();

    }


    /*
     * CmdLineProvider<T>::operator=
     */
    template<class T> CmdLineProvider<T>& CmdLineProvider<T>::operator=(const CmdLineProvider<T>& rhs) {
        if (&rhs != this) {
            this->clearArgumentList();

            this->storeCount = rhs.storeCount;
            this->argCount = rhs.storeCount; // TODO: Maybe restore the (changed) arguments

            this->memoryAnchor[0] = new Char*[this->storeCount];
            this->memoryAnchor[1] = new Char*[this->storeCount];
            this->arguments = this->memoryAnchor[1];

            for (int i = 0; i < this->storeCount; i++) {
                unsigned int len = 1; // 1 because including the terminating zero
                Char *c = rhs.memoryAnchor[0][i];
                while (*c != 0) { len++; c++; }

                this->memoryAnchor[0][i] = new Char[len];
                ::memcpy(this->memoryAnchor[0][i], rhs.memoryAnchor[0][i], len * T::CharSize());

                this->arguments[i] = this->memoryAnchor[0][i]; // TODO: Maybe restore the (changed) arguments
            }

        }
        return *this;
    }


    /*
     * CmdLineProvider<T>::CreateCmdLine
     */
    template<class T> void CmdLineProvider<T>::CreateCmdLine(const Char *appName, const Char *cmdLine) {
        Char *ci;
        Char *start = NULL;
        unsigned int state;

        // clear old argument list
        this->clearArgumentList();

        // count strings in command line
        this->storeCount = (appName == NULL) ? 0 : 1;

        ci = const_cast<Char *>(cmdLine);
        state = 1;
        while (state > 0) {
            switch (state) {
                case 0: break; // end of string
                case 1:
                    if (*ci == 0) { // end of string
                        state = 0;
                    } else if (*ci == static_cast<Char>('"')) { // start of qouted parameter
                        this->storeCount++; 
                        state = 3;
                    } else if (!T::IsSpace(*ci)) { // start of parameter
                        this->storeCount++; 
                        state = 2;
                    }
                    break;
                case 2:
                    if (*ci == 0) { // end of parameter
                        state = 0;
                    } else if (*ci == static_cast<Char>('"')) { // start of qouted section
                        state = 3;
                    } else if (T::IsSpace(*ci)) { // end of parameter
                        state = 1;
                    }
                    break;
                case 3:
                    if (*ci == 0) { // end of parameter
                        state = 0;
                    } else if (*ci == static_cast<Char>('"')) { // possible end of qouted section/parameter
                        state = 4;
                    } 
                    break;
                case 4:
                    if (*ci == 0) { // end of parameter
                        state = 0;
                    } else if (*ci == static_cast<Char>('"')) { // escaped qout!
                        state = 3;
                    } else if (T::IsSpace(*ci)) { // end of parameter
                        state = 1; // truncate last space!
                    } else { // end of qouted section
                        state = 2;
                    }
                    break;
            }
            ci++;
        }

        // create argument list
        this->memoryAnchor[0] = new Char*[this->storeCount];
        this->memoryAnchor[1] = new Char*[this->storeCount];
        this->arguments = this->memoryAnchor[1];

        if (appName != NULL) {
            unsigned int len = 1; // 1 because including the terminating zero
            ci = const_cast<Char*>(appName);
            while (*ci != 0) { len++; ci++; }
            this->memoryAnchor[0][0] = new Char[len];
            ::memcpy(this->memoryAnchor[0][0], appName, len * T::CharSize());
            this->argCount = 1;
        } else {
            this->argCount = 0;
        }

        ci = const_cast<Char *>(cmdLine);
        state = 1;
        while (state > 0) {
            switch (state) {
                case 0: break; // end of string
                case 1:
                    if (*ci == 0) { // end of string
                        state = 0;
                    } else if (*ci == static_cast<Char>('"')) { // start of qouted parameter
                        start = ci + 1;
                        state = 3;
                    } else if (!T::IsSpace(*ci)) { // start of parameter
                        start = ci;
                        state = 2;
                    }
                    break;
                case 2:
                    if (*ci == 0) { // end of parameter
                        state = 0;
                        this->memoryAnchor[0][this->argCount++] = this->createArgument(start, (ci - 1));
                    } else if (*ci == static_cast<Char>('"')) { // start of qouted section
                        state = 3;
                    } else if (T::IsSpace(*ci)) { // end of parameter
                        state = 1;
                        this->memoryAnchor[0][this->argCount++] = this->createArgument(start, (ci - 1));
                    }
                    break;
                case 3:
                    if (*ci == 0) { // end of parameter
                        state = 0;
                        this->memoryAnchor[0][this->argCount++] = this->createArgument(start, (ci - 1));
                    } else if (*ci == static_cast<Char>('"')) { // possible end of qouted section/parameter
                        state = 4;
                    } 
                    break;
                case 4:
                    if (*ci == 0) { // end of parameter
                        state = 0;
                        this->memoryAnchor[0][this->argCount++] = this->createArgument(start, (ci - 2));
                    } else if (*ci == static_cast<Char>('"')) { // escaped qout!
                        state = 3;
                    } else if (T::IsSpace(*ci)) { // end of parameter
                        state = 1; // truncate last space!
                        this->memoryAnchor[0][this->argCount++] = this->createArgument(start, (ci - 2));
                    } else { // end of qouted section
                        state = 2;
                    }
                    break;
            }
            ci++;
        }

        ASSERT(this->storeCount == this->argCount);

        for (this->argCount = 0; this->argCount < this->storeCount; this->argCount++) {
            this->arguments[this->argCount] = this->memoryAnchor[0][this->argCount];
        }

    }


    /*
     * CmdLineProvider<T>::clearArgumentList
     */
    template<class T> void CmdLineProvider<T>::clearArgumentList(void) {
        if (this->memoryAnchor[0] != NULL) {
            for (int i = 0; i < this->storeCount; i++) {
                ARY_SAFE_DELETE(this->memoryAnchor[0][i]);
            }
            ARY_SAFE_DELETE(this->memoryAnchor[0]);
            ARY_SAFE_DELETE(this->memoryAnchor[1]);
            this->arguments = NULL;
        }

        this->storeCount = 0;
        this->argCount = 0;

    }


    /*
     * CmdLineProvider<T>::createArgument
     */
    template<class T> typename CmdLineProvider<T>::Char * CmdLineProvider<T>::createArgument(Char *left, Char *right) {
        Char *buf = NULL;
        int len = static_cast<int>(right - left) + 1;

        if (len <= 0) {
            buf = new Char[1];
            buf[0] = 0;
        } else {
            Char *ci = buf = new Char[len + 1];

            // copy characters individual to unescape quots
            for (ci = buf; left <= right; left++) {
                if ((*left != static_cast<Char>('"')) || (*(left - 1) != static_cast<Char>('"'))) {
                    *(ci++) = *left;
                }
            }
            *ci = 0;
        }        

        return buf;
    }


    /*
     * CmdLineProvider<T>::Append
     */
    template<class T> void CmdLineProvider<T>::Append(const Char *arg) {
        this->reflectCallerChanges();
        
        // and now, because i know that memory anchor and active argument 
        // list are equal ...
        delete[] this->memoryAnchor[0];
        this->memoryAnchor[0] = new Char*[this->storeCount + 1];

        // copy old pointers
        memcpy(this->memoryAnchor[0], this->arguments, sizeof(Char*) * this->storeCount);
        
        // copy new argument
        unsigned int len = T::SafeStringLength(arg) + 1;
        this->memoryAnchor[0][this->storeCount] = new Char[len];
        this->memoryAnchor[0][this->storeCount][0] = 0;
        this->memoryAnchor[0][this->storeCount][len - 1] = 0;
        memcpy(this->memoryAnchor[0][this->storeCount], arg, sizeof(Char) * len);

        this->storeCount++;

        // update active argument list
        delete[] this->memoryAnchor[1];
        this->memoryAnchor[1] = new Char*[this->storeCount + 1];
        this->arguments = this->memoryAnchor[1];
        memcpy(this->arguments, this->memoryAnchor[0], sizeof(Char*) * this->storeCount);

        this->argCount++;
    }


    /*
     * CmdLineProvider<T>::Prepend
     */
    template<class T> void CmdLineProvider<T>::Prepend(const Char *arg) {
        this->reflectCallerChanges();
        
        // and now, because i know that memory anchor and active argument 
        // list are equal ...
        delete[] this->memoryAnchor[0];
        this->memoryAnchor[0] = new Char*[this->storeCount + 1];

        // copy old pointers
        memcpy(this->memoryAnchor[0] + 1, this->arguments, sizeof(Char*) * this->storeCount);
        
        // copy new argument
        unsigned int len = T::SafeStringLength(arg) + 1;
        this->memoryAnchor[0][0] = new Char[len];
        this->memoryAnchor[0][0][0] = 0;
        this->memoryAnchor[0][0][len - 1] = 0;
        memcpy(this->memoryAnchor[0][0], arg, sizeof(Char) * len);

        this->storeCount++;

        // update active argument list
        delete[] this->memoryAnchor[1];
        this->memoryAnchor[1] = new Char*[this->storeCount + 1];
        this->arguments = this->memoryAnchor[1];
        memcpy(this->arguments, this->memoryAnchor[0], sizeof(Char*) * this->storeCount);

        this->argCount++;
    }


    /*
     * CmdLineProvider<T>::SingleStringCommandLine
     */
    template<class T> String<T> CmdLineProvider<T>::SingleStringCommandLine(bool includeFirst) {
        String<T> retval;
        int startIndex = includeFirst ? 0 : 1;

        if (this->argCount > startIndex) {
            // calculate length of string to be returned
            unsigned int len = this->argCount - (1 + startIndex); // separating spaces

            for (int i = startIndex; i < this->argCount; i++) {
                bool needQuots = (*this->arguments[i] == 0);
                for (Char *ci = this->arguments[i]; *ci != 0; ci++) {
                    len++; // character
                    if (T::IsSpace(*ci)) needQuots = true; // parameter with spaces
                    if (*ci == static_cast<Char>('"')) {
                        needQuots = true;
                        len++; // escape qouts
                    }
                }
                if (needQuots) len += 2;
            }

            // build string
            Char *data = retval.AllocateBuffer(len);

            for (int i = startIndex; i < this->argCount; i++) {
                if (i > startIndex) *(data++) = static_cast<Char>(' '); // seperating space

                // remember start if quots are needed
                bool needQuots = (*this->arguments[i] == 0);
                Char *cs = data;

                for (Char *ci = this->arguments[i]; *ci != 0; ci++) {
                    *(data++) = *ci;

                    if ((T::IsSpace(*ci)) || (*ci == static_cast<Char>('"'))) {
                        // quots are needed!
                        needQuots = true;
                        break;
                    }
                }    

                if (needQuots) {
                    data = cs;
                    *(data++) = static_cast<Char>('"'); // starting quot
                    for (Char *ci = this->arguments[i]; *ci != 0; ci++) {
                        if (*ci == static_cast<Char>('"')) { // escape quot
                            *(data++) = static_cast<Char>('"');
                        }
                        *(data++) = *ci;
                    }
                    *(data++) = static_cast<Char>('"'); // ending quot
                }
            }

            *(data++) = 0;
    
        } else {
            // no arguments, so return empty string
            retval.AllocateBuffer(0);

        }

        return retval;
    }


    /*
     * CmdLineProvider<T>::reflectCallerChanges
     */
    template<class T> void CmdLineProvider<T>::reflectCallerChanges(void) {

        // delete all entries in mV which are missing in aV
        for (int i = 0; i < this->storeCount; i++) {
            bool found = false;
            for (int j = 0; j < this->argCount; j++) {
                if (this->memoryAnchor[0][i] == this->arguments[j]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                delete[] this->memoryAnchor[0][i];
            }
        }

        // allocate new memory for the memory anchor
        if (this->storeCount != this->argCount) {
            delete[] this->memoryAnchor[0];
            this->storeCount = this->argCount;
            this->memoryAnchor[0] = new Char*[this->storeCount];
        }

        // copy remaining elements
        for (int i = 0; i < this->storeCount; i++) {
            this->memoryAnchor[0][i] = this->arguments[i];
        }

        // refresh arguments / memoryAnchor[1]
        delete[] this->memoryAnchor[1];
        this->memoryAnchor[1] = new Char*[this->storeCount];
        this->arguments = this->memoryAnchor[1];
        for (int i = 0; i < this->storeCount; i++) {
            this->arguments[i] = this->memoryAnchor[0][i];
        }

    }


    /** Template instantiation for ANSI strings. */
    typedef CmdLineProvider<CharTraitsA> CmdLineProviderA;

    /** Template instantiation for wide strings. */
    typedef CmdLineProvider<CharTraitsW> CmdLineProviderW;

    /** Template instantiation for TCHARs. */
    typedef CmdLineProvider<TCharTraits> TCmdLineProvider;

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CMDLINEPROVIDER_H_INCLUDED */
