/*
 * TextFileReader.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_TextFileReader_H_INCLUDED
#define VISLIB_TextFileReader_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/File.h"
#include "vislib/String.h"


namespace vislib {
namespace sys {


    /**
     * Utility class to read a text file a line at a time.
     *
     *     CURRENTLY ONLY ASCII TEXT FILES ARE SUPPORTED!
     *
     * All methods with unicode string parameters will work on ascii strings
     * and will convert their results to unicode!
     */
    class TextFileReader {
    public:

        /**
         * Ctor.
         *
         * @param file The file object to read from. The reader will not take
         *             the ownership of the file object specified. The caller
         *             must ensure that the file object remains valid as long
         *             as it is used by this reader. The reader will also not
         *             close or open the file!
         * @param bufferSize The size of the line buffer to be used in bytes.
         */
        TextFileReader(File *file = NULL, unsigned int bufferSize = 10240);

        /** Dtor. */
        ~TextFileReader(void);

        /**
         * Synchronises the reader position based on the position of the file
         * pointer. You must not call this method if no file object has been
         * set.
         */
        void FilePositionToReaderPosition(void);

        /**
         * Gets the size of the read buffer in bytes.
         *
         * @return The size of the read buffer in bytes.
         */
        inline unsigned int GetBufferSize(void) const {
            return this->bufSize;
        }

        /**
         * Gets the file object from which the reader reads.
         *
         * @return The file object from which the reader reads.
         */
        inline const File * GetFile(void) const {
            return this->file;
        }

        /**
         * Synchronises the position of the file pointer based on the reader
         * position. You must not call this method if no file object has been
         * set.
         */
        void ReaderPositionToFilePosition(void);

        /**
         * Reads the next line from the text file. You must not call this
         * method if no file object has been set.
         *
         * @param outLine The string to receive the content of the newly read
         *                line.
         * @param maxSize The maximum number of characters to be read from the
         *                file
         *
         * @return 'true' if a line could be successfully read, 'false'
         *         otherwise (usually eof). If 'false' is returned the value
         *         of outLine is undefined.
         */
        bool ReadLine(StringA& outLine, unsigned int maxSize = 1024);

        /**
         * Reads the next line from the text file. You must not call this
         * method if no file object has been set.
         *
         * @param outLine The string to receive the content of the newly read
         *                line.
         * @param maxSize The maximum number of characters to be read from the
         *                file
         *
         * @return 'true' if a line could be successfully read, 'false'
         *         otherwise (usually eof). If 'false' is returned the value
         *         of outLine is undefined.
         */
        bool ReadLine(StringW& outLine, unsigned int maxSize = 1024);

        /**
         * Reads the next line from the text file. You must not call this
         * method if no file object has been set.
         *
         * @param maxSize The maximum number of characters to be read from the
         *                file
         *
         * @return The content of the newly read line.
         */
        inline StringA ReadLineA(unsigned int maxSize = 1024) {
            StringA line;
            if (!this->ReadLine(line, maxSize)) {
                line.Clear();
            }
            return line;
        }

        /**
         * Reads the next line from the text file. You must not call this
         * method if no file object has been set.
         *
         * @param success Receives the return value of 'ReadLine': 'true' if a
         *                line had be read successfully, 'false' otherwise.
         * @param maxSize The maximum number of characters to be read from the
         *                file
         *
         * @return The content of the newly read line.
         */
        inline StringA ReadLineA(bool& success, unsigned int maxSize = 1024) {
            StringA line;
            if ((success = this->ReadLine(line, maxSize)) == false) {
                line.Clear();
            }
            return line;
        }

        /**
         * Reads the next line from the text file. You must not call this
         * method if no file object has been set.
         *
         * @param maxSize The maximum number of characters to be read from the
         *                file
         *
         * @return The content of the newly read line.
         */
        inline StringW ReadLineW(unsigned int maxSize = 1024) {
            StringW line;
            if (!this->ReadLine(line, maxSize)) {
                line.Clear();
            }
            return line;
        }

        /**
         * Reads the next line from the text file. You must not call this
         * method if no file object has been set.
         *
         * @param maxSize The maximum number of characters to be read from the
         *                file
         *
         * @return The content of the newly read line.
         */
        inline StringW ReadLineW(bool& success, unsigned int maxSize = 1024) {
            StringW line;
            if ((success = this->ReadLine(line, maxSize)) == false) {
                line.Clear();
            }
            return line;
        }

        /**
         * Sets the size of the read buffer in bytes.
         *
         * @param bufferSize The new size of the read buffer in bytes.
         */
        void SetBufferSize(unsigned int bufferSize);

        /**
         * Sets the file to be read from, replacing any previously set file
         * object. The object being replace will not be closed or deleted by
         * the reader, nor will the reader take ownership of the new file
         * object. The caller must ensure that the new file object remains
         * valid as long as it is used by the reader. This implicitly calls
         * 'ReaderPositionToFilePosition' on the old file object if necessary
         * and 'FilePositionToReaderPosition' on the new file object.
         *
         * @param file The new file object to read from.
         */
        void SetFile(File *file);

    private:

        /** Forbidden copy ctor. */
        TextFileReader(const TextFileReader& src);

        /** Forbidden assignment operator */
        TextFileReader& operator=(const TextFileReader& src);

        /** The read buffer */
        char *buf;

        /** The position of the file pointer inside the buffer */
        unsigned int bufPos;

        /** The size of the buffer memory */
        unsigned int bufSize;

        /** The starting point of the buffer in the file */
        File::FileSize bufStart;

        /** The file object to be used. */
        File *file;

        /** The size of the valid portion of the buffer */
        unsigned int validBufSize;

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_TextFileReader_H_INCLUDED */

