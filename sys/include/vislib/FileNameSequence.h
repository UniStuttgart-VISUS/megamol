/*
 * FileNameSequence.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_FILENAMESEQUENCE_H_INCLUDED
#define VISLIB_FILENAMESEQUENCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/CharTraits.h"
#include "vislib/Array.h"
#include "vislib/SmartPtr.h"
#include "vislib/String.h"
#include <memory.h>


namespace vislib {
namespace sys {


    /**
     * Utility class for file sequences identified through growing numbers in
     * the file names. This class is used for creating such file name
     * sequences as well as for autodetecting such sequences based on the name
     * for the first file of the sequence.
     *
     * The file name is build up by several strings and number sequences with
     * fixed minimum number of digits.
     */
    class FileNameSequence {
    public:

        /**
         * Abstract base class for the individual file name elements.
         */
        class FileNameElement {
        public:

            /** Ctor. */
            FileNameElement(void);

            /** Dtor. */
            virtual ~FileNameElement(void);

            /**
             * Answers the number of different states for this file name
             * element.
             *
             * @return The number of different states for this file name
             *         element
             */
            virtual unsigned int Count(void) const = 0;

            /**
             * Returns the textual representation of this file name element as
             * ASCII string.
             *
             * @return A textual representation of this file name element.
             */
            const char * TextA(void) const;

            /**
             * Returns the textual representation of this file name element as
             * unicode string.
             *
             * @return A textual representation of this file name element.
             */
            const wchar_t * TextW(void) const;

        private:

            /**
             * Returns the textual representation of this file name element as
             * ASCII string. The memory returned must be allocated using the
             * default array 'new' operator.
             *
             * @return A textual representation of this file name element.
             */
            virtual char * makeTextA(void) const = 0;

            /**
             * Returns the textual representation of this file name element as
             * unicode string. The memory returned must be allocated using the
             * default array 'new' operator.
             *
             * @return A textual representation of this file name element.
             */
            virtual wchar_t * makeTextW(void) const = 0;

            /** The output buffer */
            mutable char *buffer;

            /** Flag if buffer points to a unicode buffer */
            mutable bool unicodeBuffer;

        };

        /**
         * File name elements of fixed strings.
         * Use a CharTraits class as template parameter.
         */
        template<class T>
        class FileNameStringElement: public FileNameElement {
        public:

            /** Ctor. */
            FileNameStringElement(void) : FileNameElement(), text() {
                // Intentionally empty
            }

            /**
             * Ctor. Initialises the file name element with the given text.
             *
             * @param text The string to be used for file name elements.
             */
            FileNameStringElement(const vislib::String<T>& text)
                    : FileNameElement(), text(text) {
                // Intentionally empty
            }

            /** Dtor. */
            virtual ~FileNameStringElement(void) {
                // Intentionally empty
            }

            /**
             * Answers the number of different states for this file name
             * element.
             *
             * @return The number of different states for this file name
             *         element
             */
            virtual unsigned int Count(void) const {
                return 1;
            }

            /**
             * Sets the fixed string for this file name element.
             *
             * @param text The string for this file name element.
             */
            void SetText(const String<T>& text) {
                this->text = text;
            }

            /**
             * Returns the text set to be the fixed string for this file name
             * element.
             *
             * @return The text of this file name element.
             */
            const String<T>& Text(void) const {
                return this->text;
            }

        private:

            /**
             * Returns the textual representation of this file name element as
             * ASCII string. The memory returned must be allocated using the
             * default array 'new' operator.
             *
             * @return A textual representation of this file name element.
             */
            virtual char * makeTextA(void) const {
                unsigned int len = this->text.Length();
                char *c = new char[len + 1];
                StringA str(text);
                c[len] = static_cast<char>(0);
                memcpy(c, str.PeekBuffer(), len * sizeof(char));
                return c;
            }

            /**
             * Returns the textual representation of this file name element as
             * unicode string. The memory returned must be allocated using the
             * default array 'new' operator.
             *
             * @return A textual representation of this file name element.
             */
            virtual wchar_t * makeTextW(void) const {
                unsigned int len = this->text.Length();
                wchar_t *c = new wchar_t[len + 1];
                StringW str(text);
                c[len] = static_cast<wchar_t>(0);
                memcpy(c, str.PeekBuffer(), len * sizeof(wchar_t));
                return c;
            }

            /** The fixed string a file name element */
            String<T> text;

        };

        /** Typedef for ANSI string file name elements */
        typedef FileNameStringElement<CharTraitsA> FileNameStringElementA;

        /** Typedef for unicode string file name elements */
        typedef FileNameStringElement<CharTraitsW> FileNameStringElementW;

        /** Typedef for autotype string file name elements */
        typedef FileNameStringElement<TCharTraits> TFileNameStringElement;

        /**
         * Class for counting file name elements
         */
        class FileNameCountElement : public FileNameElement {
        public:

            /**
             * Ctor. Initialises the counter to count from 1 to 1 with
             * stepsize 1 and output at least 1 digit.
             */
            FileNameCountElement(void);

            /**
             * Ctor.
             *
             * @param digits The number of minimum digits to output. If the
             *               current value of the counter does not produce
             *               enough digits leadings zeros will be prepended.
             * @param minVal The minimum and starting value for the counter.
             * @param maxVal The maximum value for the counter.
             * @param step   The increase step size for the counter.
             */
            FileNameCountElement(unsigned int digits, unsigned int minVal,
                unsigned int maxVal, unsigned int step = 1);

            /** Dtor. */
            virtual ~FileNameCountElement(void);

            /**
             * Answers the number of different states for this file name
             * element.
             *
             * @return The number of different states for this file name
             *         element
             */
            virtual unsigned int Count(void) const;

            /**
             * Answers the index of the current counter.
             *
             * @return The index of the current counter.
             */
            virtual unsigned int CounterIndex(void) const;

            /**
             * Answers the current value of the counter.
             *
             * @return The current value of the counter.
             */
            inline unsigned int CounterValue(void) const {
                return this->value;
            }

            /**
             * Gets the number of digits to be output at least.
             *
             * @return The number of digits to be output at least.
             */
            inline unsigned int Digits(void) const {
                return this->digits;
            }

            /**
             * Increases the counter if possible.
             *
             * @return 'true' if the counter has been increased, 'false' if it
             *         was not possible to increase the counter because
             *         otherwise its value would be greater than the maximum
             *         value allowed.
             */
            virtual bool IncreaseCounter(void);

            /**
             * Answer the maximum value allowed for the counter.
             *
             * @return The maximum value allowed for the counter.
             */
            inline unsigned int MaxValue(void) const {
                return this->maxVal;
            }

            /**
             * Answers the minimum and starting value for the counter.
             *
             * @return The minimum and starting value for the counter.
             */
            inline unsigned int MinValue(void) const {
                return this->minVal;
            }

            /**
             * Resets the counter to the starting value.
             */
            inline void ResetCounter(void) {
                this->SetCounterIndex(0);
            }

            /**
             * Sets the counter to it's 'idx'-th value.
             *
             * @param idx The index value to set the counter to.
             */
            virtual void SetCounterIndex(unsigned int idx);

            /**
             * Sets the minimum number of digits to output.
             *
             * @param digits The minimum number of digits to output.
             */
            void SetDigits(unsigned int digits);

            /**
             * Sets the range of the values of the counter.
             *
             * @param minVal The minimum and starting value for the counter.
             * @param maxVal The maximum value for the counter.
             * @param step   The increase step size for the counter.
             */
            void SetRange(unsigned int minVal, unsigned int maxVal,
                unsigned int step = 1);

            /**
             * Sets the increase step size for the counter.
             *
             * @param step The increase step size for the counter.
             */
            void SetStepSize(unsigned int step);

            /**
             * Gets the increase step size for the counter.
             *
             * @return The increase step size for the counter.
             */
            inline unsigned int StepSize(void) const {
                return this->step;
            }

        private:

            /**
             * Returns the textual representation of this file name element as
             * ASCII string. The memory returned must be allocated using the
             * default array 'new' operator.
             *
             * @return A textual representation of this file name element.
             */
            virtual char * makeTextA(void) const;

            /**
             * Returns the textual representation of this file name element as
             * unicode string. The memory returned must be allocated using the
             * default array 'new' operator.
             *
             * @return A textual representation of this file name element.
             */
            virtual wchar_t * makeTextW(void) const;

            /** The number of minimum output digits for the string output */
            unsigned int digits;

            /** The maximum value for the counter */
            unsigned int maxVal;

            /** The minimum value for the counter */
            unsigned int minVal;

            /** The step for the counter */
            unsigned int step;

            /** The value of the counter */
            unsigned int value;

        };

        /** Ctor. */
        FileNameSequence(void);

        /** Dtor. */
        ~FileNameSequence(void);

        /**
         * Performs an autodetection for a file name sequence based on the
         * file name of the first file. Note that all files has to be
         * existing.
         *
         * @param firstFileName The file name of the first file of the
         *                      potential file sequence to be autodetected.
         */
        void Autodetect(const StringA& firstFileName);

        /**
         * Performs an autodetection for a file name sequence based on the
         * file name of the first file. Note that all files has to be
         * existing.
         *
         * @param firstFileName The file name of the first file of the
         *                      potential file sequence to be autodetected.
         */
        void Autodetect(const StringW& firstFileName);

        /**
         * Answers the number of file names defined by this sequence.
         *
         * @return The number of file names defined by this sequence.
         */
        unsigned int Count(void);

        /**
         * Answers the 'idx'-th file name element of the sequence.
         *
         * @param idx The index of the file name element to be returned.
         *
         * @return The requested file name element.
         */
        inline const SmartPtr<FileNameElement>& Element(unsigned int idx)
                const {
            return this->elements[idx];
        }

        /**
         * Answer the number of file name elements for this sequence.
         *
         * @return The number of file name elements for this sequence.
         */
        inline unsigned int ElementCount(void) const {
            return static_cast<unsigned int>(this->elements.Count());
        }

        /**
         * Answer the 'idx'-th file name of the sequence. You must not call
         * this method on invalid sequences!
         *
         * @return The 'idx'-th file name of the sequence.
         */
        StringA FileNameA(unsigned int idx);

        /**
         * Answer the 'idx'-th file name of the sequence. You must not call
         * this method on invalid sequences!
         *
         * @return The 'idx'-th file name of the sequence.
         */
        StringW FileNameW(unsigned int idx);

        /**
         * Checks whether this sequence is valid or not.
         *
         * @return 'true' if this sequence is valid, 'false' if not.
         */
        bool IsValid(void) const;

        /**
         * Answer wether the file name counter element are priorisised in
         * reversed order.
         *
         * @return 'true' if the counter elements are priorisised in reversed
         *         order, 'false' otherwise.
         */
        inline bool ReversedCounterPriority(void) const {
            return this->reversedPriority;
        }

        /**
         * Sets the 'idx'-th file name element of the sequence.
         *
         * @param idx The index of the element to be set.
         * @param e   Pointer to the new element. The memory of this object
         *            will be owned by the sequence object after this call
         *            successfully returns. So the caller must not free the
         *            memory of this object.
         */
        inline void SetElement(unsigned int idx, FileNameElement *e) {
            this->elements[idx] = e;
        }

        /**
         * Sets the 'idx'-th file name element of the sequence.
         *
         * @param idx The index of the element to be set.
         * @param e   Pointer to the new element. The memory of this object
         *            is owned by the SmartPtr object.
         */
        inline void SetElement(unsigned int idx,
                const SmartPtr<FileNameElement>& e) {
            this->elements[idx] = e;
        }

        /**
         * Sets the number of file name elements for this sequence.
         *
         * @param cnt The number of file name elements for this sequence.
         */
        inline void SetElementCount(unsigned int cnt) {
            this->elements.Resize(cnt);
            this->elements.SetCount(cnt);
        }

        /**
         * Answer wether the file name counter element are priorisised in
         * reversed order.
         *
         * @param rev The flag whether the file name counter element are
         *            priorisised in reversed order.
         */
        inline void SetReversedCounterPriority(bool rev) {
            this->reversedPriority = rev;
        }

    private:

        /**
         * Implementation of the autodetection code.
         *
         * @param filename The file name of the first file of the potential
         *                 file sequence to be autodetected.
         */
        template<class T>
        void autodetect(const vislib::String<T>& filename);

        /**
         * Sets the indices of all elements so that they form the requested
         * index for the whole sequence.
         *
         * @param idx The requested index.
         */
        void setIndex(unsigned int idx);

        /** The elements of the file names */
        Array<SmartPtr<FileNameElement> > elements;

        /** Flag for reversing the counter element priority */
        bool reversedPriority;

    };

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_FILENAMESEQUENCE_H_INCLUDED */

