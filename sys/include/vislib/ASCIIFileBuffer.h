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
#include "vislib/assert.h"
#include "vislib/File.h"
#include "vislib/IllegalStateException.h"
#include "vislib/MemmappedFile.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/String.h"


namespace vislib {
namespace sys {


    /**
     * Buffer class loading a whole ASCII text file into memory and providing
     * a pointer array to access the lines.
     */
    class ASCIIFileBuffer {
    public:

        /** Possible parsing elements */
        enum ParsingElement {
            PARSING_DEFAULT,
            PARSING_LINES,
            PARSING_WORDS
        };

        /**
         * Class storing all information about a single line
         */
        class LineBuffer {
        public:

            /**
             * Ctor
             */
            LineBuffer();

            /**
             * copy ctor
             *
             * @param src The object to clone from
             */
            LineBuffer(const LineBuffer& src);

            /** Dtor */
            ~LineBuffer(void);

            /**
             * Answer the number of word in this line. This value is zero if
             * the parsing element were lines and thus no tokens were
             * identified or if the line is empty (except for whitespaces).
             *
             * @return The number of words in this line
             */
            inline SIZE_T Count(void) const {
                return this->cnt;
            }

            /**
             * Answer the pointer to the string of the line. Do not call when
             * the parsing elements were words and if 'Count' returns a value
             * larger than zero.
             *
             * @return The pointer to the string of the line
             */
            inline const char * Pointer(void) const {
                if (this->cnt > 0) {
                    throw vislib::IllegalStateException(
                        "ASCIIFileBuffer was parsed for words. "
                        "Requesting lines is thus illegal",
                        __FILE__, __LINE__);
                }
                return this->ptr.line;
            }

            /**
             * Answer the idx-th word of the line
             *
             * @param idx The zero-based index of the word to return
             *
             * @return The requested word
             *
             * @throw OutOfRangeException if a non-existing line is requested
             */
            inline const char * Word(SIZE_T idx) const {
                if (idx >= this->cnt) {
                    throw vislib::OutOfRangeException(static_cast<int>(idx),
                        0, static_cast<int>(this->cnt - 1),
                        __FILE__, __LINE__);
                }
                return this->ptr.words[idx];
            }

            /**
             * Assignment operator
             *
             * @param rhs The right hand side operand
             *
             * @return A reference to 'this'
             */
            LineBuffer& operator=(const LineBuffer& rhs);

            /**
             * Test for equality
             *
             * @param rhs The right hand side operand
             *
             * @return True if this and rhs are equal
             */
            inline bool operator==(const LineBuffer& rhs) const {
                return (this->cnt == rhs.cnt)
                    && (this->ptr.line == rhs.ptr.line);
            }

            /**
             * Answer the pointer to the string of the line. Do not call when
             * the parsing elements were words and if 'Count' returns a value
             * larger than zero.
             *
             * @return The pointer to the string of the line
             */
            operator const char *(void) const {
                if (this->cnt > 0) {
                    throw vislib::IllegalStateException(
                        "ASCIIFileBuffer was parsed for words. "
                        "Requesting lines is thus illegal",
                        __FILE__, __LINE__);
                }
                return this->ptr.line;
            }

        private:

            /**
             * Ctor
             *
             * @param line The line to set
             */
            LineBuffer(char *line);

            /**
             * Ctor
             *
             * @param words The words to set
             */
            LineBuffer(vislib::Array<char *>& words);

            /**
             * Assignment operator
             *
             * @param line The line to set
             *
             * @return A reference to 'this'
             */
            LineBuffer& operator=(char *line);

            /**
             * Assignment operator
             *
             * @param words The words to set
             *
             * @return A reference to 'this'
             */
            LineBuffer& operator=(vislib::Array<char *>& words);

            /** The number of tokens, or Zero if only storing the line */
            SIZE_T cnt;

            /** The pointers */
            union _pointers_t {

                /** Pointer to the beginning of the line if 'cnt' is zero */
                char *line;

                /**
                 * Array of the pointers to the beginnings of 'cnt' tokens, if
                 * the 'cnt' is larger than zero
                 */
                char **words;

            } ptr;

            /** Friend class for creation */
            friend class ASCIIFileBuffer;

        };

        /**
         * Ctor.
         *
         * @param elements The elements to be parsed. 'PARSING_DEFAULT' is not
         *                 a legal value and will be changed to
         *                 'PARSING_LINES'
         */
        ASCIIFileBuffer(ParsingElement elements = PARSING_LINES);

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
         * Answer the parsing element which will be parsed when no other
         * element is requested specifically.
         *
         * @return The default parsing element
         */
        inline ParsingElement GetParsingElements(void) const {
            return this->defElements;
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
        inline const LineBuffer&  Line(SIZE_T idx) const {
            return this->lines[idx];
        }

        /**
         * Loads the whole file as ASCII text into the buffer and builds up
         * the array of lines for accessing the data.
         *
         * @param filename The path to the file to be loaded
         * @param elements The elements to be parsed.
         *
         * @return True on success, false on failure
         *
         * @throw vislib::Exception on any critical failure
         */
        inline bool LoadFile(const vislib::StringA& filename,
                ParsingElement elements = PARSING_DEFAULT) {
            MemmappedFile file;
            if (!file.Open(filename, File::READ_ONLY, File::SHARE_READ,
                    File::OPEN_ONLY)) return false;
            return this->LoadFile(file, elements);
        }

        /**
         * Loads the whole file as ASCII text into the buffer and builds up
         * the array of lines for accessing the data.
         *
         * @param filename The path to the file to be loaded
         * @param elements The elements to be parsed.
         *
         * @return True on success, false on failure
         *
         * @throw vislib::Exception on any critical failure
         */
        inline bool LoadFile(const char *filename,
                ParsingElement elements = PARSING_DEFAULT) {
            MemmappedFile file;
            if (!file.Open(filename, File::READ_ONLY, File::SHARE_READ,
                    File::OPEN_ONLY)) return false;
            return this->LoadFile(file, elements);
        }

        /**
         * Loads the whole file as ASCII text into the buffer and builds up
         * the array of lines for accessing the data.
         *
         * @param filename The path to the file to be loaded
         * @param elements The elements to be parsed.
         *
         * @return True on success, false on failure
         *
         * @throw vislib::Exception on any critical failure
         */
        inline bool LoadFile(const vislib::StringW& filename,
                ParsingElement elements = PARSING_DEFAULT) {
            MemmappedFile file;
            if (!file.Open(filename, File::READ_ONLY, File::SHARE_READ,
                    File::OPEN_ONLY)) return false;
            return this->LoadFile(file, elements);
        }

        /**
         * Loads the whole file as ASCII text into the buffer and builds up
         * the array of lines for accessing the data.
         *
         * @param filename The path to the file to be loaded
         * @param elements The elements to be parsed.
         *
         * @return True on success, false on failure
         *
         * @throw vislib::Exception on any critical failure
         */
        inline bool LoadFile(const wchar_t *filename,
                ParsingElement elements = PARSING_DEFAULT) {
            MemmappedFile file;
            if (!file.Open(filename, File::READ_ONLY, File::SHARE_READ,
                    File::OPEN_ONLY)) return false;
            return this->LoadFile(file, elements);
        }

        /**
         * Loads the whole file as ASCII text into the buffer and builds up
         * the array of lines for accessing the data. The current position in
         * 'file' is irrelevant. The file will not be closed, but the position
         * within will be undefined.
         *
         * @param file The file to be loaded
         * @param elements The elements to be parsed.
         *
         * @return True on success, false on failure
         *
         * @throw vislib::Exception on any critical failure
         */
        bool LoadFile(File& file, ParsingElement elements = PARSING_DEFAULT);

        /**
         * Sets the parsing element which will be parsed when no other element
         * is requested specifically.
         *
         * @param elements The new default parsing element
         */
        void SetParsingElements(ParsingElement elements);

        /**
         * Answer the idx-th line of the buffer
         *
         * @param idx The zero-based index of the line to return
         *
         * @return The requested line
         *
         * @throw OutOfRangeException if a non-existing line is requested
         */
        inline const LineBuffer& operator[](SIZE_T idx) const {
            return this->lines[idx];
        }

    private:

        /** The buffer holding the whole file */
        char *buffer;

        /** Access to the lines in 'buffer' */
        vislib::Array<LineBuffer> lines;

        /** The default parsing elements */
        ParsingElement defElements;

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ASCIIFILEBUFFER_H_INCLUDED */

