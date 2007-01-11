/*
 * ColumnFormatter.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_COLUMNFORMATTER_H_INCLUDED
#define VISLIB_COLUMNFORMATTER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include <vislib/CharTraits.h>
#include <vislib/String.h>
#include <vislib/UnsupportedOperationException.h>
#include <vislib/IllegalParamException.h>


namespace vislib {


    /**
     * Helper class formating strings into columns. The class can basically be 
     * instantiated for vislib::CharTraits subclasses that implement all the 
     * required operations.
     */
    template<class T> class ColumnFormatter {

    public:

        /** Define a local name for the character type. */
        typedef typename T::Char Char;

        /** Define a local name for the string type. */
        typedef typename vislib::String<T> String;

        /**
         * Nested class representing a single column.
         */
        class Column {
        public:
            /** allow parser to access the members directly */
            friend class ColumnFormatter<T>;

            /** dtor. */
            ~Column(void);

            /**
             * Sets the width value of the column. A width of Zero (default 
             * value) tells the formater to produce a column with the size of 
             * the contained text. However, GetWidth will continue to return 
             * zero.
             *
             * @param width The new width of the column.
             */
            inline void SetWidth(unsigned int width) {
                this->width = width;
            }

            /**
             * Answer the size of the column.
             *
             * @return The size of the column.
             */
            inline unsigned int GetWidth(void) const {
                return this->width;
            }

            /**
             * Disables wrapping of the column text. If the column text is 
             * larger then width, the text will go into the space of the next
             * columns. However, this will not prohibit wrapping of the text
             * when reaching max width.
             */
            inline void DisableWrapping(void) {
                this->noWrap = true;
            }

            /**
             * Enables or disables wrapping of the column text, depening on the
             * value of enable.
             * @see DisableWrapping for further information.
             *
             * @param enable If true enables wrapping, if false disables
             *               wrapping of the column text into a new line.
             */
            inline void EnableWrapping(bool enable = true) {
                this->noWrap = !enable;
            }

            /**
             * Answer whether wrapping of the column text is enabled or not.
             *
             * @return true if wrapping is disabled, false if enabled.
             */
            inline bool IsWrappingDisabled(void) const {
                return this->noWrap;
            }

            /**
             * Sets the text of the column. The text is copied. Special 
             * formating characters (e.g. '\n', '\t', '\r', '\a', or '\b') will
             * corrupt the output of FormatColumns and must not be used.
             *
             * @param text The new text of the column.
             */
            inline void SetText(const String& text) {
                this->text = text;
            }

            /**
             * Answer the text of the column.
             *
             * @return the text of the column.
             */
            inline const String& GetText(void) const {
                return this->text;
            }

        private:
            
            /** private default ctor. */
            Column(void);

            /** forbidden copy ctor. */
            Column(const Column& rhs);

            /** assignment operator */
            Column& operator=(const Column& rhs);

            /** width of the column */
            unsigned int width;

            /** flag to prohibit wrap arounds */
            bool noWrap;

            /** text of the column */
            String text;

            /** internal width value during the calculation*/
            unsigned int realWidth;

        };

        /** 
         * Default Ctor. Initializes column-count to zero, the maximum width
         * to zero, and the separator to two spaces.
         */
        ColumnFormatter(void);

        /** 
         * Ctor. Initializes column-count to colCount and initializes all 
         * columns with the width defWidth. The maximum width is set to zero,
         * and the separator to two spaces.
         *
         * @param colCount The initial number of columns.
         * @param defWidth The initial width value of each column.
         */
        ColumnFormatter(unsigned int colCount, unsigned int defWidth = 0);

        /** 
         * Copy ctor.
         *
         * @param rhs The right hand side operand
         */
        ColumnFormatter(const ColumnFormatter& rhs);

        /** Dtor. */
        ~ColumnFormatter(void);

        /**
         * Sets the number of columns to colCount. If the column number is
         * decreased, the remaining columns will keep their values. If the 
         * number ins increased, the orignial columns will also keep their
         * values and the new columns will be initialized with default values.
         *
         * @param colCount The new number of columns.
         */
        void SetColumnCount(unsigned int colCount);

        /**
         * Answer the number of columns.
         *
         * @return The number of columns.
         */
        inline unsigned int GetColumnCount(void) {
            return this->colCount;
        }

        /**
         * Returns the reference to the Column object specified by col.
         *
         * @param col The number of the Column.
         *
         * @return The reference to the Column object specified by col.
         *
         * @throw IllegalParamException if col is greater or equal the number
         *        of columns.
         */
        Column& AccessColumn(unsigned int col);

        /**
         * Returns the reference to the Column object specified by col.
         *
         * @param col The number of the Column.
         *
         * @return The reference to the Column object specified by col.
         *
         * @throw IllegalParamException if col is greater or equal the number
         *        of columns.
         */
        inline Column& operator[](unsigned int col) {
            return this->AccessColumn(col);
        }

        /**
         * Sets the maximum width for all columns altogether. The text produced
         * by the formatter will never exceed this width. A value of zero
         * (default value) disables this functionality. A value larger then the
         * maximum width will also be interpreted like zero.
         *
         * Note: Keep in mind that each line will be ended with an additional 
         * "New line" character ('\n') not counting to this maximum width. So 
         * if you want to fill a 80 character width console windows, you must 
         * set the maximum width to 79 to avoid double line-breaks, if the new 
         * line character would be the 81sd character and so the console
         * performs an automated new line.
         *
         * @param width The new value for the maximum width.
         */
        inline void SetMaxWidth(unsigned int width) {
            this->maxWidth = width;
        }

        /**
         * Answer the maximum width for all columns altogether.
         *
         * @return The maximum width.
         */
        inline unsigned int GetMaxWidth(void) const {
            return this->maxWidth;
        }

        /**
         * Sets the separator to sep. This separator will be output between 
         * each two columns. Keep in mind that these 
         * "(columnCount - 1) * length of separator" characters also count to
         * the maximum width of the text line.
         *
         * @param sep The new value for the separator.
         */
        inline void SetSeparator(const String& sep) {
            this->separator = sep;
        }

        /**
         * Answer the separator value.
         *
         * @return The separator value.
         */
        inline const String& GetSeparator(void) {
            return this->separator;
        }

        /**
         * Performs the formatting and writes the output to the string object
         * specified by outString. Each line of the string will contain maximum
         * width characters (plus a new line character if another line 
         * follows). If the text is too short, it will be filled up with white
         * spaces.
         *
         * @param outString String receiving the formatted output.
         *
         * @return A reference to outString.
         */
        String& FormatColumns(String &outString);

        /**
         * Output operator. Behaves like "FormatColumns".
         * @see FormatColumns
         *
         * @param outString String receiving the formatted output.
         *
         * @return A reference to outString.
         */
        String& operator>>(String &outString) {
            return this->FormatColumns(outString);
        }

        /** 
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return Reference of this object
         */
        ColumnFormatter& operator=(const ColumnFormatter& rhs);

    private:

        /** number of columns */
        unsigned int colCount;

        /** column objects */
        Column *cols;

        /** maximal width for all columns altogether. */
        unsigned int maxWidth;

        /** seperator string to be output between the columns */
        String separator;

    };


    /*
     * CmdLineParser<T>::Argument::Argument
     */
    template<class T>
    ColumnFormatter<T>::Column::Column(void) : width(0), noWrap(false), text() {
    }


    /*
     * CmdLineParser<T>::Argument::~Argument
     */
    template<class T>
    ColumnFormatter<T>::Column::~Column(void) {
    }


    /*
     * CmdLineParser<T>::Argument::Argument
     */
    template<class T>
    ColumnFormatter<T>::Column::Column(const Column& rhs) {
        throw UnsupportedOperationException("Column copy ctor", __FILE__, __LINE__);
    }


    /*
     * CmdLineParser<T>::Argument::operator=
     */
    template<class T>
    typename ColumnFormatter<T>::Column& ColumnFormatter<T>::Column::operator=(const Column& rhs) {
        if (&rhs != this) {
            this->width = rhs.width;
            this->noWrap = rhs.noWrap;
            this->text = rhs.text;
        }

        return *this;
    }


    /*
     * ColumnFormatter::ColumnFormatter
     */
    template<class T>
    ColumnFormatter<T>::ColumnFormatter(void) 
            : colCount(0), cols(NULL), maxWidth(0), separator(static_cast<Char>(' '), 2) {
    }


    /*
     * ColumnFormatter::ColumnFormatter
     */
    template<class T>
    ColumnFormatter<T>::ColumnFormatter(const ColumnFormatter& rhs) {
        *this = rhs;
    }


    /*
     * ColumnFormatter::ColumnFormatter
     */
    template<class T>
    ColumnFormatter<T>::ColumnFormatter(unsigned int colCount, unsigned int defWidth) 
            : maxWidth(0), separator(static_cast<Char>(' '), 2) {
        this->colCount = colCount;

        this->cols = new Column[this->colCount];
        for (unsigned int i = 0; i < this->colCount; i++) {
            this->cols[i].width = defWidth;
        }
    }


    /*
     * ColumnFormatter::~ColumnFormatter
     */
    template<class T>
    ColumnFormatter<T>::~ColumnFormatter(void) {
        ARY_SAFE_DELETE(this->cols);
    }

    
    /*
     * ColumnFormatter<T>::SetColumnCount
     */
    template<class T>
    void ColumnFormatter<T>::SetColumnCount(unsigned int colCount) {
        if (colCount != this->colCount) {
            Column *oc = this->cols;
            this->cols = new Column[colCount];

            // copy old values
            if (this->colCount > colCount) this->colCount = colCount;
            for (unsigned int i = 0; i < this->colCount; i++) {
                this->cols[i] = oc[i];
            }

            this->colCount = colCount;
            ARY_SAFE_DELETE(oc);
        }
    }


    /*
     * ColumnFormatter<T>::AccessColumn
     */
    template<class T>
    typename ColumnFormatter<T>::Column& ColumnFormatter<T>::AccessColumn(unsigned int col) {
        if (col >= this->colCount) {
            throw IllegalParamException("col", __FILE__, __LINE__);
        }

        return this->cols[col];
    }


    /*
     * ColumnFormatter<T>::AccessColumn
     */
    template<class T>
    typename ColumnFormatter<T>::String& ColumnFormatter<T>::FormatColumns(String &outString) {

        // some sort of debug implementation to be able to write the online help producer for the CmdLineParser
        if (this->colCount > 0) {
            outString = this->cols[0].GetText();
            for (unsigned int i = 1; i < this->colCount; i++) {
                outString += this->separator;
                outString += this->cols[i].GetText();
            }

        } else {
            outString.Clear();

        }

        // TODO: Implement real functionallity

        return outString;
    }


    /*
     * ColumnFormatter<T>::operator=
     */
    template<class T>
    ColumnFormatter<T>& ColumnFormatter<T>::operator=(const ColumnFormatter<T>& rhs) {
        if (this != &rhs) {
            this->colCount = rhs.colCount;
            this->cols = new Column[this->colCount];
            for (unsigned int i = 0; i < this->colCount; i++) {
                this->cols[i] = rhs.cols[i];
            }

            this->maxWidth = rhs.maxWidth;
            this->separator = rhs.separator;
        }

        return *this;
    }

    
    /** Template instantiation for ANSI strings. */
    typedef ColumnFormatter<CharTraitsA> ColumnFormatterA;

    /** Template instantiation for wide strings. */
    typedef ColumnFormatter<CharTraitsW> ColumnFormatterW;

    /** Template instantiation for TCHARs. */
    typedef ColumnFormatter<TCharTraits> TColumnFormatter;

} /* end namespace vislib */

#endif /* VISLIB_COLUMNFORMATTER_H_INCLUDED */
