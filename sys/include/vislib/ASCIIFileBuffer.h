/*
 * ASCIIFileBuffer.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ASCIIFILEBUFFER_H_INCLUDED
#define VISLIB_ASCIIFILEBUFFER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/Array.h"
#include "vislib/File.h"
#include "vislib/MemmappedFile.h"
#include "vislib/String.h"


namespace vislib {
namespace sys {


    /**
     * Buffer class loading a whole ASCII text file into memory and providing
     * a pointer array to access the lines.
     */
    class ASCIIFileBuffer {
    public:

        /** Ctor. */
        ASCIIFileBuffer(void);

        /** Dtor. */
        ~ASCIIFileBuffer(void);

        /** Clears the buffer */
        void Clear(void);

        /**
         * Answer the number of lines stored in the buffer
         *
         * @return The number of lines stored in the buffer
         */
        inline SIZE_T Count(void) const {
            return this->lines.Count();
        }

        /**
         * Answer the idx-th line of the buffer
         *
         * @param idx The zero-based index of the line to return
         *
         * @return The requested line
         *
         * @throw OutOfRangeException if a non-existing line is requested
         */
        inline const char * Line(SIZE_T idx) const {
            return this->lines[idx];
        }

        /**
         * Loads the whole file as ASCII text into the buffer and builds up
         * the array of lines for accessing the data.
         *
         * @param filename The path to the file to be loaded
         *
         * @return True on success, false on failure
         *
         * @throw vislib::Exception on any critical failure
         */
        inline bool LoadFile(const vislib::StringA& filename) {
            MemmappedFile file;
            if (!file.Open(filename, File::READ_ONLY, File::SHARE_READ,
                    File::OPEN_ONLY)) return false;
            return this->LoadFile(file);
        }

        /**
         * Loads the whole file as ASCII text into the buffer and builds up
         * the array of lines for accessing the data.
         *
         * @param filename The path to the file to be loaded
         *
         * @return True on success, false on failure
         *
         * @throw vislib::Exception on any critical failure
         */
        inline bool LoadFile(const char *filename) {
            MemmappedFile file;
            if (!file.Open(filename, File::READ_ONLY, File::SHARE_READ,
                    File::OPEN_ONLY)) return false;
            return this->LoadFile(file);
        }

        /**
         * Loads the whole file as ASCII text into the buffer and builds up
         * the array of lines for accessing the data.
         *
         * @param filename The path to the file to be loaded
         *
         * @return True on success, false on failure
         *
         * @throw vislib::Exception on any critical failure
         */
        inline bool LoadFile(const vislib::StringW& filename) {
            MemmappedFile file;
            if (!file.Open(filename, File::READ_ONLY, File::SHARE_READ,
                    File::OPEN_ONLY)) return false;
            return this->LoadFile(file);
        }

        /**
         * Loads the whole file as ASCII text into the buffer and builds up
         * the array of lines for accessing the data.
         *
         * @param filename The path to the file to be loaded
         *
         * @return True on success, false on failure
         *
         * @throw vislib::Exception on any critical failure
         */
        inline bool LoadFile(const wchar_t *filename) {
            MemmappedFile file;
            if (!file.Open(filename, File::READ_ONLY, File::SHARE_READ,
                    File::OPEN_ONLY)) return false;
            return this->LoadFile(file);
        }

        /**
         * Loads the whole file as ASCII text into the buffer and builds up
         * the array of lines for accessing the data. The current position in
         * 'file' is irrelevant. The file will not be closed, but the position
         * within will be undefined.
         *
         * @param file The file to be loaded
         *
         * @return True on success, false on failure
         *
         * @throw vislib::Exception on any critical failure
         */
        bool LoadFile(File& file);

        /**
         * Answer the idx-th line of the buffer
         *
         * @param idx The zero-based index of the line to return
         *
         * @return The requested line
         *
         * @throw OutOfRangeException if a non-existing line is requested
         */
        inline const char * operator[](SIZE_T idx) const {
            return this->lines[idx];
        }

    private:

        /** The buffer holding the whole file */
        char *buffer;

        /** Access to the lines in 'buffer' */
        vislib::Array<char *> lines;

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ASCIIFILEBUFFER_H_INCLUDED */

