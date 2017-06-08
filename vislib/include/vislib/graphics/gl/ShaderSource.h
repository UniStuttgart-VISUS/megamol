/*
 * ShaderSource.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SHADERSOURCE_H_INCLUDED
#define VISLIB_SHADERSOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/SmartPtr.h"
#include "vislib/Array.h"
#include "vislib/Map.h"
#include "vislib/String.h"


namespace vislib {
namespace graphics {
namespace gl {



    /**
     * Class of sources of a shader.
     *
     * Use each one object to define a vertex, fragment, or geometry shader. To
     * construct a GLSLShader, for example, you will need two object of this 
     * class. However the user must ensure that the different shader source 
     * objects used to build up the shader program are compatible.
     */
    class ShaderSource {
    public:

        /**
         * Nested abstract base class of shader source code snippets.
         */
        class Snippet {
        public:

#ifndef _WIN32
            /** stupid dtor for stupid gcc */
            virtual ~Snippet(void) { }
#endif /* !_WIN32 */

            /**
             * Returns the content of the snippet.
             */
            virtual const char *PeekCode(void) const = 0;

        };

        /**
         * Nested class of simple string snippets.
         */
        class StringSnippet : public Snippet {
        public:

            /** ctor. Creates an empty snippet. */
            StringSnippet(void);

            /** 
             * ctor.
             *
             * @param str The text of the snippet.
             */
            StringSnippet(const vislib::StringA& str);

            /**
             * ctor.
             *
             * @param str The text of the snippet.
             */
            StringSnippet(const char *str);

            /** dtor */
            virtual ~StringSnippet(void);

            /**
             * Loads an ANSI text file into the snippet.
             *
             * @param filename The name of the ANSI text file.
             *
             * @return true if the file was successfully read, otherwise false.
             *
             * @throw IOException If reading from the file failed.
             */
            bool LoadFile(const char* filename);

            /**
             * Returns the content of the snippet.
             */
            virtual const char *PeekCode(void) const;

            /**
             * Sets the text of the snippet.
             *
             * @param str The text of the snippet.
             */
            void Set(const vislib::StringA& str);

            /**
             * Sets the text of the snippet.
             *
             * @param str The text of the snippet.
             */
            void Set(const char *str);

            /**
             * Answer the text of the snippet.
             *
             * @return The text of the snippet.
             */
            const vislib::StringA& String(void) const;

            /**
             * Assignment operator from vislib string.
             *
             * @param str The text of the snippet.
             *
             * @return A reference to this.
             */
            inline StringSnippet& operator=(const vislib::StringA& str) {
                this->Set(str);
                return *this;
            }

            /**
             * Assignment operator from string.
             *
             * @param str The text of the snippet.
             *
             * @return A reference to this.
             */
            inline StringSnippet& operator=(const char *str) {
                this->Set(str);
                return *this;
            }

        protected:

            /** The text of the snippet. */
            vislib::StringA txt;
        };

        /**
         * Nested class of a version snippet. This snippet returns a textual
         * representation of a numeric version in the form:
         * '#version NUMBER'
         */
        class VersionSnippet : public Snippet {
        public:

            /** 
             * ctor.
             *
             * @version The numeric version value.
             */
            VersionSnippet(int version = 110);

            /** dtor */
            virtual ~VersionSnippet(void);

            /**
             * Returns the content of the snippet.
             */
            virtual const char *PeekCode(void) const;

            /**
             * Sets the version
             *
             * @param version The numeric version value.
             */
            void Set(int version);

            /**
             * Answer the version
             *
             * @return The version.
             */
            int Version(void) const;

        private:

            /** The string representation of the version */
            vislib::StringA verStr;
        };

        /** ctor. Creates empty shader source. */
        ShaderSource(void);

        /**
         * Copy ctor.
         * Creates a shallow copy of 'src' only copying the smart pointers, but
         * not the snippet objects themself.
         *
         * @param src The object to clone from.
         */
        ShaderSource(const ShaderSource& src);

        /** dtor */
        ~ShaderSource(void);

        /**
         * Appends one code snipped to the shader source.
         *
         * @param code The code snipped to be added.
         *
         * @return The snipped added.
         *
         * @throw IllegalParamException if code is <NULL>.
         */
        const vislib::SmartPtr<Snippet>& Append(
            const vislib::SmartPtr<Snippet>& code);

        /**
         * Appends one code snipped to the shader source.
         *
         * @param name The name to be used to identify this snippet.
         * @param code The code snipped to be added.
         *
         * @return The snipped added.
         *
         * @throw AlreadyExistsException if there already is a snippet with
         *                               this name.
         * @throw IllegalParamException if code is <NULL>.
         */
        const vislib::SmartPtr<Snippet>& Append(
            const vislib::StringA& name,
            const vislib::SmartPtr<Snippet>& code);

        /**
         * Clears the shader source by removing all snippets.
         */
        void Clear(void);

        /**
         * Answers the code array of the shader source. The array returned is 
         * guaranteed to be valid until the next call of any member of this 
         * object.
         *
         * @return The code array of this shader source.
         */
        const char ** Code(void) const;

        /**
         * Answers the number of code snippets of the shader source.
         *
         * @return The number of code snippets.
         */
        SIZE_T Count(void) const;

        /**
         * Inserts one code snipped into the shader source at the given index.
         *
         * @param idx The index where to insert the snippet.
         * @param code The snippet to insert.
         *
         * @return The snippet inserted.
         *
         * @throw IllegalParamException if code is <NULL>.
         * @throw OutOfRangeException If 'idx' is not within 
         *                            [0, this->Count()[.
         */
        const vislib::SmartPtr<Snippet>& Insert(const SIZE_T idx, 
            const vislib::SmartPtr<Snippet>& code);

        /**
         * Inserts one code snipped into the shader source at the given index.
         *
         * @param idx The index where to insert the snippet.
         * @param name The name to be used to identify this snippet.
         * @param code The snippet to insert.
         *
         * @return The snippet inserted.
         *
         * @throw IllegalParamException if code is <NULL>.
         * @throw AlreadyExistsException if there already is a snippet with
         *                               this name.
         * @throw OutOfRangeException If 'idx' is not within 
         *                            [0, this->Count()[.
         */
        const vislib::SmartPtr<Snippet>& Insert(const SIZE_T idx, 
            const vislib::StringA& name,
            const vislib::SmartPtr<Snippet>& code);

        /**
         * Answer the name index.
         *
         * @param name The name to return the index.
         *
         * @throw NoSuchElementException if there is no index with this name.
         */
        SIZE_T NameIndex(const vislib::StringA& name);

        /**
         * Adds one code snippet to the beginning of the shader source.
         *
         * @param code The snippet to be prepended.
         *
         * @return The snippet prepended.
         *
         * @throw IllegalParamException if code is <NULL>.
         */
        const vislib::SmartPtr<Snippet>& Prepend(
            const vislib::SmartPtr<Snippet>& code);

        /**
         * Adds one code snippet to the beginning of the shader source.
         *
         * @param name The name to be used to identify this snippet.
         * @param code The snippet to be prepended.
         *
         * @return The snippet prepended.
         *
         * @throw AlreadyExistsException if there already is a snippet with
         *                               this name.
         * @throw IllegalParamException if code is <NULL>.
         */
        const vislib::SmartPtr<Snippet>& Prepend(
            const vislib::StringA& name,
            const vislib::SmartPtr<Snippet>& code);

        /**
         * Removes one snippet from the shader source.
         *
         * @param idx The index of the snipped to be removed.
         *
         * @throw OutOfRangeException If 'idx' is not within 
         *                            [0, this->Count()[.
         */
        void Remove(SIZE_T idx);

        /**
         * Removes the name 'name', if it exists.
         *
         * @param name The name to be removed.
         */
        void RemoveName(vislib::StringA& name);

        /**
         * Replaces a snippet with a new one.
         *
         * @param idx The index of the snipped to be replace.
         * @param code The new snippet to be inserted.
         *
         * @return The snippet replaced by the new one.
         */
        vislib::SmartPtr<Snippet> Replace(SIZE_T idx, 
            const vislib::SmartPtr<Snippet>& code);

        /**
         * Sets the shader source to the given snippet.
         *
         * @param code The snippet to be set as shader source.
         *
         * @return The snipped set as shader source.
         *
         * @throw IllegalParamException if code is <NULL>.
         */
        const vislib::SmartPtr<Snippet>& Set(
            const vislib::SmartPtr<Snippet>& code);

        /**
         * Sets a name for a given index.
         *
         * @param name The name to be set.
         * @param idx The index to be set.
         *
         * @throw AlreadyExistsException if there already is a snippet with
         *                               this name.
         * @throw OutOfRangeException If 'idx' is not within 
         *                            [0, this->Count()[.
         */
        void SetName(const vislib::StringA& name, SIZE_T idx);

        /**
         * Creates a single string containing the whole shader source code.
         *
         * @return The string containing the whole shader source code.
         */
        vislib::StringA WholeCode(void) const;

        /**
         * Answers one snippet of the shader source.
         *
         * @param idx The zero based index of the snippet to return.
         *
         * @return The idx-th snippet of the shader source.
         *
         * @throw OutOfRangeException If 'idx' is not within 
         *                            [0, this->Count()[.
         */
        const vislib::SmartPtr<Snippet>& operator[](const SIZE_T idx) const;

        /**
         * Assignment operator.
         * Creates a shallow copy of 'rhs' only copying the smart pointers, but
         * not the snippet objects themself.
         *
         * @param rhs The right hand side operand.
         *
         * @return A reference to 'this'.
         */
        ShaderSource& operator=(const ShaderSource& rhs);

    private:

        /** the constructed source code */
        vislib::Array<vislib::SmartPtr<Snippet> > snippets;

        /** the names of some of the shader snippets */
        vislib::Map<vislib::StringA, SIZE_T> names;

        /** the code output array */
        mutable const char **code;

    };

    
} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SHADERSOURCE_H_INCLUDED */

