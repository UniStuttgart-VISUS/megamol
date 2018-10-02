/*
 * BTFParser.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_BTFPARSER_INCLUDED
#define MEGAMOLCORE_BTFPARSER_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/xml/ConditionalParser.h"
#include "mmcore/utility/xml/XmlReader.h"
#include "vislib/Map.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/SmartPtr.h"
#include "vislib/Stack.h"


namespace megamol {
namespace core {
namespace utility {

    /** forward declaration */
    class ShaderSourceFactory;

    /**
     * Class for the MegaMol™ baschtel-together-file xml parser.
     *
     * See documentation for specification of BTF xml. Note that forward
     * references are not possible.
     */
    class BTFParser : public utility::xml::ConditionalParser {
    public:

        /**
         * Nested base class for btf elements
         */
        class MEGAMOLCORE_API BTFElement {
        public:

            /**
             * Ctor.
             */
            BTFElement(void) : name() {
                // intentionally empty
            }

            /**
             * Ctor.
             *
             * @param name The new name for this element
             */
            BTFElement(const vislib::StringA& name) : name(name) {
                // intentionally empty
            }

            /**
             * Dtor.
             */
            virtual ~BTFElement(void) {
                // intentionally empty
            }

            /**
             * Answer the name of the element.
             *
             * @return The name of the element.
             */
            const vislib::StringA& Name(void) const {
                return this->name;
            }

            /**
             * Sets the name of the element.
             *
             * @param name The new name of the element.
             */
            void SetName(const vislib::StringA& name) {
                this->name = name;
            }

            /**
             * Test for equality
             *
             * @param rhs The right hand side operand.
             *
             * @return true if equal, false otherwise.
             */
            virtual bool operator==(const BTFElement& rhs) {
                return this->name.Equals(rhs.name);
            }

        private:

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
            /** The name of the element */
            vislib::StringA name;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

        };

        /**
         * Nested class of a btf namespace
         */
        class MEGAMOLCORE_API BTFNamespace : public BTFElement {
        public:

            /**
             * Ctor.
             */
            BTFNamespace(void) : BTFElement(), children() {
                // intentionally empty
            }

            /**
             * Ctor.
             *
             * @param name The name of the namespace
             */
            BTFNamespace(const vislib::StringA& name) : BTFElement(name),
                    children() {
                // intentionally empty
            }

            /**
             * Dtor.
             */
            virtual ~BTFNamespace(void) {
                this->children.Clear();
            }

            /**
             * Gets the list of the children.
             *
             * @return The list of the children.
             */
            virtual vislib::SingleLinkedList<vislib::SmartPtr<BTFElement> >&
                    Children(void) {
                return this->children;
            }

            /**
             * Gets the list of the children.
             *
             * @return The list of the children.
             */
            virtual const vislib::SingleLinkedList<
                    vislib::SmartPtr<BTFElement> >& Children(void) const {
                return this->children;
            }

            /**
             * Gets the child element with the given name.
             *
             * @param name The name to search for.
             *
             * @return The found child element or NULL.
             */
            virtual vislib::SmartPtr<BTFElement> FindChild(
                    const vislib::StringA& name) const {
                vislib::ConstIterator<vislib::SingleLinkedList<
                    vislib::SmartPtr<BTFElement> >::Iterator>
                    i = this->children.GetConstIterator();
                while (i.HasNext()) {
                    const vislib::SmartPtr<BTFElement>& p = i.Next();
                    if (p.IsNull()) continue;
                    if (p->Name().Equals(name)) return p;
                }
                return NULL;
            }

            /**
             * Test for equality
             *
             * @param rhs The right hand side operand.
             *
             * @return true if equal, false otherwise.
             */
            virtual bool operator==(const BTFElement& rhs) {
                // we only test the name, because this is sufficient!
                return BTFElement::operator==(rhs);
            }

        private:

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
            /** The children of the namespace */
            vislib::SingleLinkedList<vislib::SmartPtr<BTFElement> > children;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

        };

        /**
         * Nested class of a btf code snippet
         */
        class BTFSnippet : public BTFElement {
        public:

            /** possible values of the snippet type */
            enum SnippetType {
                EMPTY_SNIPPET,
                STRING_SNIPPET,
                VERSION_SNIPPET,
                FILE_SNIPPET
            };

            /**
             * Ctor.
             */
            BTFSnippet(void) : BTFElement(), type(EMPTY_SNIPPET), content() {
                // intentionally empty
            }

            /**
             * Ctor.
             *
             * @param name The name of the namespace
             */
            BTFSnippet(const vislib::StringA& name) : BTFElement(name),
                    type(EMPTY_SNIPPET), content() {
                // intentionally empty
            }

            /**
             * Dtor.
             */
            virtual ~BTFSnippet(void) {
                // intentionally empty
            }

            /**
             * Gets the content of the snippet.
             *
             * @return The content of the snippet.
             */
            inline const vislib::StringA& Content(void) const {
                return this->content;
            }

            /**
             * Gets the file of the snippet.
             *
             * @return The file of the snippet.
             */
            inline const vislib::StringA& File(void) const {
                return this->file;
            }

            /**
             * Gets the line of the start tag of the snippet.
             *
             * @return The line of the start tag of the snippet.
             */
            inline SIZE_T Line(void) const {
                return this->line;
            }

            /**
             * Sets the content of the snippet.
             *
             * @param content The content of the snippet.
             */
            inline void SetContent(const vislib::StringA& content) {
                this->content = content;
            }

            /**
             * Sets the file of the snippet.
             *
             * @param file The file of the snippet.
             */
            inline void SetFile(const vislib::StringA& file) {
                this->file = file;
            }

            /**
             * Sets the line of the start tag of the snippet.
             *
             * @param line The line of the start tag of the snippet.
             */
            inline void SetLine(SIZE_T line) {
                this->line = line;
            }

            /**
             * Sets the type of the snippet.
             *
             * @param type The new type of the snippet.
             */
            inline void SetType(SnippetType type) {
                this->type = type;
            }

            /**
             * Gets the type of the snippet.
             *
             * @return The type of the snippet.
             */
            inline SnippetType Type(void) const {
                return this->type;
            }

            /**
             * Test for equality
             *
             * @param rhs The right hand side operand.
             *
             * @return true if equal, false otherwise.
             */
            virtual bool operator==(const BTFElement& rhs) {
                // we only test the name, because this is sufficient!
                return BTFElement::operator==(rhs);
            }

        private:

            /** The snippet type */
            SnippetType type;

            /** The content of the snippet */
            vislib::StringA content;

            /** The original file of the snippet */
            vislib::StringA file;

            /** The line of the snippets start tag */
            SIZE_T line;

        };

        /**
         * Nested class of a btf shader
         */
        class BTFShader : public BTFNamespace {
        public:

            /**
             * Ctor.
             */
            BTFShader(void) : BTFNamespace(), ids() {
                // intentionally empty
            }

            /**
             * Ctor.
             *
             * @param name The name of the namespace
             */
            BTFShader(const vislib::StringA& name) : BTFNamespace(name),
                    ids() {
                // intentionally empty
            }

            /**
             * Dtor.
             */
            virtual ~BTFShader(void) {
                // intentionally empty
            }

            /**
             * Test for equality
             *
             * @param rhs The right hand side operand.
             *
             * @return true if equal, false otherwise.
             */
            virtual bool operator==(const BTFElement& rhs) {
                // we only test the name, because this is sufficient!
                return BTFElement::operator==(rhs);
            }

            /**
             * Access to the name id map of the shader.
             *
             * @return The name id map of the shader.
             */
            inline vislib::Map<vislib::StringA, unsigned int>& NameIDs(void) {
                return this->ids;
            }

            /**
             * Access to the name id map of the shader.
             *
             * @return The name id map of the shader.
             */
            inline const vislib::Map<vislib::StringA, unsigned int>&
            NameIDs(void) const {
                return this->ids;
            }

        private:

            /** The name id map */
            vislib::Map<vislib::StringA, unsigned int> ids;

        };

        /**
         * Ctor.
         *
         * @param core The core instance
         */
        BTFParser(ShaderSourceFactory& factory);

        /** Dtor. */
        virtual ~BTFParser(void);

        /**
         * Sets the root element to be used by the parser. This must be done
         * before 'Parse' is called. The root element must use the correct file
         * root namespace name. The root element must not have any children.
         *
         * @param root The root element to use.
         */
        void SetRootElement(const vislib::SmartPtr<BTFElement>& root,
            const vislib::SmartPtr<BTFElement>& masterroot);

    protected:

        /**
         * Callback method for character data.
         *
         * @param level The level of the current xml tag.
         * @param text Pointer to the character data. Note that this is not
         *             zero-terminated.
         * @param len Length of the character data.
         * @param state The current state of the reader.
         */
        virtual void CharacterData(unsigned int level, const XML_Char *text,
            int len, xml::XmlReader::ParserState state);

        /**
         * Callback method for comment data.
         * This implementation ignores all data.
         *
         * @param level The level of the current xml tag.
         * @param text Pointer to the character data.
         * @param state The current state of the reader.
         */
        virtual void Comment(unsigned int level, const XML_Char *text,
             xml::XmlReader::ParserState state);

        /**
         * Checks whether the xml file is a 'config', '1.0' MegaMol™ xml file.
         *
         * @param reader The current xml reader.
         *
         * @return 'true' if the file of the reader is compatible with this 
         *         parser, 'false' otherwise.
         */
        virtual bool CheckBaseTag(const xml::XmlReader& reader);

        /**
         * Callback method for starting xml tags.
         *
         * @param num The number of the current xml tag.
         * @param level The level of the current xml tag.
         * @param name The name of the current xml tag. This string should be
         *             considdered utf8.
         * @param attrib The attributes of the current xml tag. These strings
         *               should be considdered utf8.
         * @param state The current state of the reader.
         * @param outChildState The ParserState used by the reader for the 
         *                      first child of the current xml tag. The default
         *                      value is the current state of the reader.
         * @param outEndTagState The ParserState to be set when the end tag
         *                       callback is called for the end tag of this
         *                       start tag. The default value is the current
         *                       state of the reader.
         * @param outPostEndTagState The ParserState to be set after the end
         *                           tag callback has been returned. This value
         *                           may be altered in the end tag callback
         *                           method. The default value is the current 
         *                           state of the reader.
         *
         * @return 'true' if the tag has been handled by this implementation.
         *         'false' if the tag was not handled.
         */
        virtual bool StartTag(unsigned int num, unsigned int level,
            const XML_Char * name, const XML_Char ** attrib,
            xml::XmlReader::ParserState state,
            xml::XmlReader::ParserState& outChildState,
            xml::XmlReader::ParserState& outEndTagState,
            xml::XmlReader::ParserState& outPostEndTagState);

        /**
         * Callback method for the ending xml tags.
         *
         * @param num The number of the current xml tag (this is the same 
         *            number used when 'startTag' was called).
         * @param level The level of the current xml tag.
         * @param name The name of the current xml tag. This string should be
         *             considdered utf8.
         * @param state The current state of the reader.
         * @param outPostEndTagState The state to be set after this callback
         *                           returned. The default value is the value
         *                           set by 'startTag'.
         *
         * @return 'true' if the tag has been handled by this implementation.
         *         'false' if the tag was not handled.
         */
        virtual bool EndTag(unsigned int num, unsigned int level,
            const XML_Char * name, utility::xml::XmlReader::ParserState state,
            xml::XmlReader::ParserState& outPostEndTagState);

    private:

        /** The xml reader state for accepting content */
        static unsigned int XMLSTATE_CONTENT;

        /** The shader source factory object */
        ShaderSourceFactory& factory;

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
        /** the root namespace */
        vislib::SmartPtr<BTFElement> root;

        /** The stack of parsing elements */
        vislib::Stack<vislib::SmartPtr<BTFElement> > stack;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

    };

} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_BTFPARSER_INCLUDED */
