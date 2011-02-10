/*
 * CmdLineParser.h
 *
 * Copyright (C) 2006-2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CMDLINEPARSER_H_INCLUDED
#define VISLIB_CMDLINEPARSER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/types.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/IllegalParamException.h"
#include "vislib/FormatException.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/assert.h"
#include "vislib/CharTraits.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/Iterator.h"
#include "vislib/CmdLineProvider.h"
#include "vislib/ColumnFormatter.h"
#include "vislib/Console.h"


namespace vislib {
namespace sys {


    /**
     * Parser class for command line arguments.
     *
     * Usage:
     * Create an object of this class. Then register all options which should 
     * be supporteded. After this, the command line can be parsed with one of 
     * the "parse"-methodes. This will build up the command line list, which
     * then can be accessed through the corresponding getters. In addition the
     * option objects will receive the values of their first occurrences in the
     * command line. You can select elements from the command line list and
     * recreate a command line, only holding these elements, to pass it to the
     * initialization functions of other libraries.
     *
     * Instantiate with CharTraits as template parameter.
     */
    template<class T> class CmdLineParser {
    public:

        /** Define a local name for the character type. */
        typedef typename T::Char Char;

        /** forward declaration of Argument nested class */
        class Argument;

        /**
         * Nested class for command line options.
         *
         * An option is defined by a short name, a long name, and a value type.
         * The parser will check the short names case sensitive but the long 
         * names case insensitive.
         */
        class Option {
        public:
            /** allow parser to access the members directly */
            friend class CmdLineParser<T>;

            /** 
             * possible flags for the ctors. These are bits an can be combined
             * using the or operator "|".
             */
            enum Flags {
                /** Symbolic constant for no flags */
                FLAG_NULL = 0,
                /**
                 * Flag indicating that this option must not appear more then 
                 * once in the command line. If the option appears more then 
                 * once, only the first one is correctly parsed to be this 
                 * option. The other appearences will be ignored and there will
                 * be a warning.
                 */
                FLAG_UNIQUE = 1, 
                /**
                 * Flag indicating that this option is required. If at least 
                 * one required option is missing in the command line there 
                 * will be an error.
                 */
                FLAG_REQUIRED = 2,
                /**
                 * Flag indicating that this option may only appear as first 
                 * option. Any following parameters will be classified as
                 * "TYPE_UNKNOWN". If this option is used not as first option,
                 * an error is generated. (This implies "FLAG_UNIQUE").
                 */
                FLAG_EXCLUSIVE = 4
            };

            /** possible variable types */
            enum ValueType {
                NO_VALUE,
                STRING_VALUE,
                INT_VALUE,
                DOUBLE_VALUE,
                BOOL_VALUE,
                INT_OR_STRING_VALUE,
                DOUBLE_OR_STRING_VALUE,
                BOOL_OR_STRING_VALUE
            };

            /**
             * Nested class representing an option value descriptor.
             */
            class ValueDesc {
            public:
                /** allow parser to access the members directly */
                friend class CmdLineParser<T>;

                /** dtor */
                ~ValueDesc(void) {
                    ValueDesc *t;
                    while (this->next) {
                        t = this->next->next;
                        this->next->next = NULL;
                        delete this->next;
                        this->next = t;
                    }
                }

                /**
                 * Construction function for the beginning of a list of values.
                 *
                 * @param type The type of the value to append to the list of 
                 *             values. Must not be NO_VALUE!
                 * @param name The name of the value. If NULL a name like 
                 *             "value i" is generated.
                 * @param desc The desctiption of the value or NULL if no 
                 *             description should be available.
                 */
                static inline ValueDesc * ValueList(ValueType type, const Char *name = NULL, const Char *desc = NULL) {
                    ValueDesc *t = new ValueDesc(type);
                    if (name != NULL) {
                        t->name = name;
                    }
                    if (desc != NULL) {
                        t->description = desc;
                    }
                    return t;
                }

                /**
                 * Construction function for the beginning of a list of values.
                 *
                 * @param type The type of the value to append to the list of 
                 *             values. Must not be NO_VALUE!
                 * @param name The name of the value. If NULL a name like 
                 *             "value i" is generated.
                 * @param desc The desctiption of the value or NULL if no 
                 *             description should be available.
                 */
                static inline ValueDesc * ValueList(ValueType type, const String<T>& name, const Char *desc = NULL) {
                    return ValueList(type, name.PeekBuffer(), desc);
                }

                /**
                 * Construction function for the beginning of a list of values.
                 *
                 * @param type The type of the value to append to the list of 
                 *             values. Must not be NO_VALUE!
                 * @param name The name of the value. If NULL a name like 
                 *             "value i" is generated.
                 * @param desc The desctiption of the value or NULL if no 
                 *             description should be available.
                 */
                static inline ValueDesc * ValueList(ValueType type, const Char *name, const String<T>& desc) {
                    return ValueList(type, name, desc.PeekBuffer());
                }

                /**
                 * Construction function for the beginning of a list of values.
                 *
                 * @param type The type of the value to append to the list of 
                 *             values. Must not be NO_VALUE!
                 * @param name The name of the value. If NULL a name like 
                 *             "value i" is generated.
                 * @param desc The desctiption of the value or NULL if no 
                 *             description should be available.
                 */
                static inline ValueDesc * ValueList(ValueType type, const String<T>& name, const String<T>& desc) {
                    return ValueList(type, name.PeekBuffer(), desc.PeekBuffer());
                }

                /**
                 * Construction function to build up the list of values.
                 *
                 * @param type The type of the value to append to the list of 
                 *             values. Must not be NO_VALUE!
                 * @param name The name of the value. If NULL a name like 
                 *             "value i" is generated.
                 * @param desc The desctiption of the value or NULL if no 
                 *             description should be available.
                 */
                inline ValueDesc * Add(ValueType type, const Char *name = NULL, const Char *desc = NULL) {
                    ValueDesc *t = this;
                    while (t->next) { t = t->next; }
                    t->next = new ValueDesc(type);
                    t = t->next;
                    if (name != NULL) {
                        t->name = name;
                    }
                    if (desc != NULL) {
                        t->description = desc;
                    }
                    return this;
                }

                /**
                 * Construction function to build up the list of values.
                 *
                 * @param type The type of the value to append to the list of 
                 *             values. Must not be NO_VALUE!
                 * @param name The name of the value. If NULL a name like 
                 *             "value i" is generated.
                 * @param desc The desctiption of the value or NULL if no 
                 *             description should be available.
                 */
                inline ValueDesc * Add(ValueType type, const String<T>& name, const Char *desc = NULL) {
                    return Add(type, name.PeekBuffer(), desc);
                }

                /**
                 * Construction function to build up the list of values.
                 *
                 * @param type The type of the value to append to the list of 
                 *             values. Must not be NO_VALUE!
                 * @param name The name of the value. If NULL a name like 
                 *             "value i" is generated.
                 * @param desc The desctiption of the value or NULL if no 
                 *             description should be available.
                 */
                inline ValueDesc * Add(ValueType type, const Char *name, const String<T>& desc) {
                    return Add(type, name, desc.PeekBuffer());
                }

                /**
                 * Construction function to build up the list of values.
                 *
                 * @param type The type of the value to append to the list of 
                 *             values. Must not be NO_VALUE!
                 * @param name The name of the value. If NULL a name like 
                 *             "value i" is generated.
                 * @param desc The desctiption of the value or NULL if no 
                 *             description should be available.
                 */
                inline ValueDesc * Add(ValueType type, const String<T>& name, const String<T>& desc) {
                    return Add(type, name.PeekBuffer(), desc.PeekBuffer());
                }

                /** 
                 * Answer the number of values.
                 *
                 * @return the number of values.
                 */
                unsigned int Count(void) const;

                /** 
                 * Answer the name of the i-th value.
                 *
                 * @return the name of the i-th value.
                 *
                 * @throw OutOfRangeException if i is out of range.
                 */
                inline const String<T>& Name(unsigned int i) const {
                    return this->element(i).name;
                }

                /** 
                 * Answer the description of the i-th value.
                 *
                 * @return the description of the i-th value.
                 *
                 * @throw OutOfRangeException if i is out of range.
                 */
                inline const String<T>& Description(unsigned int i) const {
                    return this->element(i).description;
                }

                /** 
                 * Answer the type of the i-th value.
                 *
                 * @return the type of the i-th value.
                 *
                 * @throw OutOfRangeException if i is out of range.
                 */
                inline ValueType Type(unsigned int i) const {
                    return this->element(i).type;
                }

            private:

                /** private ctor */
                ValueDesc(ValueType type) 
                    : name(), description(), type(type), next(NULL) {
                    ASSERT(this->type != NO_VALUE);
                }

                /**
                 * Answer the i-th value object.
                 *
                 * @return the i-th value object.
                 *
                 * @throw OutOfRangeException if i is out of range.
                 */
                const ValueDesc &element(unsigned int i) const;

                /** the name of the value */
                String<T> name;

                /** the description of the value */
                String<T> description;

                /** the type of the value */
                ValueType type;

                /** the pointer to the next element in the value list */
                ValueDesc *next;
            };

            /** 
             * ctor.
             * 
             * @param shortName The single character name of the option, or 0
             *                  if no short name should be used.
             * @param longName The long name string of the option. Can be NULL
             *                 if no long name should be used. Otherwise the
             *                 string will be copied into an internal buffer.
             * @param description The description string of the option. Can be
             *                    NULL if no description string is provided.
             *                    Otherwise the string will be copied into an
             *                    internal buffer.
             * @param flags The flags of the option.
             * @param valueList The list of values expected to follow this 
             *                  option directly in the command line.
             */
            inline Option(const Char shortName, const Char *longName = NULL, 
                    const Char *description = NULL, Flags flags = FLAG_NULL, 
                    ValueDesc *valueList = NULL) {
                this->initObject(shortName, longName, description, flags, 
                    valueList);
            }

            /** 
             * ctor.
             * 
             * @param shortName The single character name of the option, or 0
             *                  if no short name should be used.
             * @param longName The long name string of the option. Can be NULL
             *                 if no long name should be used. Otherwise the
             *                 string will be copied into an internal buffer.
             * @param description The description string of the option. Can be
             *                    NULL if no description string is provided.
             *                    Otherwise the string will be copied into an
             *                    internal buffer.
             * @param flags The flags of the option.
             * @param valueList The list of values expected to follow this 
             *                  option directly in the command line.
             */
            inline Option(const Char shortName, const String<T>& longName, 
                    const Char *description = NULL, Flags flags = FLAG_NULL, 
                    ValueDesc *valueList = NULL) {
                this->initObject(shortName, longName.PeekBuffer(), description,
                    flags, valueList);
            }            
            
            /** 
             * ctor.
             * 
             * @param shortName The single character name of the option, or 0
             *                  if no short name should be used.
             * @param longName The long name string of the option. Can be NULL
             *                 if no long name should be used. Otherwise the
             *                 string will be copied into an internal buffer.
             * @param description The description string of the option. Can be
             *                    NULL if no description string is provided.
             *                    Otherwise the string will be copied into an
             *                    internal buffer.
             * @param flags The flags of the option.
             * @param valueList The list of values expected to follow this 
             *                  option directly in the command line.
             */
            inline Option(const Char shortName, const Char *longName, 
                    const String<T>& description, Flags flags = FLAG_NULL, 
                    ValueDesc *valueList = NULL) {
                this->initObject(shortName, longName, description.PeekBuffer(), 
                    flags, valueList);
            }

            /** 
             * ctor.
             * 
             * @param shortName The single character name of the option, or 0
             *                  if no short name should be used.
             * @param longName The long name string of the option. Can be NULL
             *                 if no long name should be used. Otherwise the
             *                 string will be copied into an internal buffer.
             * @param description The description string of the option. Can be
             *                    NULL if no description string is provided.
             *                    Otherwise the string will be copied into an
             *                    internal buffer.
             * @param flags The flags of the option.
             * @param valueList The list of values expected to follow this 
             *                  option directly in the command line.
             */
            inline Option(const Char shortName, const String<T>& longName, 
                    const String<T>& description = NULL, Flags flags = FLAG_NULL, 
                    ValueDesc *valueList = NULL) {
                this->initObject(shortName, longName.PeekBuffer(), 
                    description.PeekBuffer(), flags, valueList);
            }

            /** 
             * ctor.
             * 
             * @param longName The long name string of the option. Can be NULL
             *                 if no long name should be used. Otherwise the
             *                 string will be copied into an internal buffer.
             * @param description The description string of the option. Can be
             *                    NULL if no description string is provided.
             *                    Otherwise the string will be copied into an
             *                    internal buffer.
             * @param flags The flags of the option.
             * @param valueList The list of values expected to follow this 
             *                  option directly in the command line.
             */
            inline Option(const Char *longName = NULL, 
                    const Char *description = NULL, Flags flags = FLAG_NULL, 
                    ValueDesc *valueList = NULL) {
                this->initObject(static_cast<Char>(0), longName, description, 
                    flags, valueList);
            }

            /** 
             * ctor.
             * 
             * @param longName The long name string of the option. Can be NULL
             *                 if no long name should be used. Otherwise the
             *                 string will be copied into an internal buffer.
             * @param description The description string of the option. Can be
             *                    NULL if no description string is provided.
             *                    Otherwise the string will be copied into an
             *                    internal buffer.
             * @param flags The flags of the option.
             * @param valueList The list of values expected to follow this 
             *                  option directly in the command line.
             */
            inline Option(const String<T>& longName, 
                    const Char *description = NULL, Flags flags = FLAG_NULL, 
                    ValueDesc *valueList = NULL) {
                this->initObject(static_cast<Char>(0), longName.PeekBuffer(), 
                    description, flags, valueList);
            }            
            
            /** 
             * ctor.
             * 
             * @param longName The long name string of the option. Can be NULL
             *                 if no long name should be used. Otherwise the
             *                 string will be copied into an internal buffer.
             * @param description The description string of the option. Can be
             *                    NULL if no description string is provided.
             *                    Otherwise the string will be copied into an
             *                    internal buffer.
             * @param flags The flags of the option.
             * @param valueList The list of values expected to follow this 
             *                  option directly in the command line.
             */
            inline Option(const Char *longName, const String<T>& description, 
                    Flags flags = FLAG_NULL, ValueDesc *valueList = NULL) {
                this->initObject(static_cast<Char>(0), longName, 
                    description.PeekBuffer(), flags, valueList);
            }

            /** 
             * ctor.
             * 
             * @param longName The long name string of the option. Can be NULL
             *                 if no long name should be used. Otherwise the
             *                 string will be copied into an internal buffer.
             * @param description The description string of the option. Can be
             *                    NULL if no description string is provided.
             *                    Otherwise the string will be copied into an
             *                    internal buffer.
             * @param flags The flags of the option.
             * @param valueList The list of values expected to follow this 
             *                  option directly in the command line.
             */
            inline Option(const String<T>& longName, 
                    const String<T>& description = NULL, 
                    Flags flags = FLAG_NULL, ValueDesc *valueList = NULL) {
                this->initObject(static_cast<Char>(0), longName.PeekBuffer(), 
                    description.PeekBuffer(), flags, valueList);
            }

            /** 
             * ctor.
             * 
             * @param shortName The single character name of the option, or 0
             *                  if no short name should be used.
             * @param longName The long name string of the option. Can be NULL
             *                 if no long name should be used. Otherwise the
             *                 string will be copied into an internal buffer.
             * @param description The description string of the option. Can be
             *                    NULL if no description string is provided.
             *                    Otherwise the string will be copied into an
             *                    internal buffer.
             * @param flags The flags of the option.
             * @param valueList The list of values expected to follow this 
             *                  option directly in the command line.
             */
            inline Option(const Char shortName, const Char *longName, 
                    const Char *description, int flags, 
                    ValueDesc *valueList = NULL) {
                this->initObject(shortName, longName, description, 
                    static_cast<Flags>(flags), valueList);
            }

            /** 
             * ctor.
             * 
             * @param shortName The single character name of the option, or 0
             *                  if no short name should be used.
             * @param longName The long name string of the option. Can be NULL
             *                 if no long name should be used. Otherwise the
             *                 string will be copied into an internal buffer.
             * @param description The description string of the option. Can be
             *                    NULL if no description string is provided.
             *                    Otherwise the string will be copied into an
             *                    internal buffer.
             * @param flags The flags of the option.
             * @param valueList The list of values expected to follow this 
             *                  option directly in the command line.
             */
            inline Option(const Char shortName, const String<T>& longName, 
                    const Char *description, int flags,
                    ValueDesc *valueList = NULL) {
                this->initObject(shortName, longName.PeekBuffer(), description,
                    static_cast<Flags>(flags), valueList);
            }            
            
            /** 
             * ctor.
             * 
             * @param shortName The single character name of the option, or 0
             *                  if no short name should be used.
             * @param longName The long name string of the option. Can be NULL
             *                 if no long name should be used. Otherwise the
             *                 string will be copied into an internal buffer.
             * @param description The description string of the option. Can be
             *                    NULL if no description string is provided.
             *                    Otherwise the string will be copied into an
             *                    internal buffer.
             * @param flags The flags of the option.
             * @param valueList The list of values expected to follow this 
             *                  option directly in the command line.
             */
            inline Option(const Char shortName, const Char *longName, 
                    const String<T>& description, int flags, 
                    ValueDesc *valueList = NULL) {
                this->initObject(shortName, longName, description.PeekBuffer(), 
                    static_cast<Flags>(flags), valueList);
            }

            /** 
             * ctor.
             * 
             * @param shortName The single character name of the option, or 0
             *                  if no short name should be used.
             * @param longName The long name string of the option. Can be NULL
             *                 if no long name should be used. Otherwise the
             *                 string will be copied into an internal buffer.
             * @param description The description string of the option. Can be
             *                    NULL if no description string is provided.
             *                    Otherwise the string will be copied into an
             *                    internal buffer.
             * @param flags The flags of the option.
             * @param valueList The list of values expected to follow this 
             *                  option directly in the command line.
             */
            inline Option(const Char shortName, const String<T>& longName, 
                    const String<T>& description, int flags, 
                    ValueDesc *valueList = NULL) {
                this->initObject(shortName, longName.PeekBuffer(), 
                    description.PeekBuffer(), static_cast<Flags>(flags), 
                    valueList);
            }

            /** 
             * ctor.
             * 
             * @param longName The long name string of the option. Can be NULL
             *                 if no long name should be used. Otherwise the
             *                 string will be copied into an internal buffer.
             * @param description The description string of the option. Can be
             *                    NULL if no description string is provided.
             *                    Otherwise the string will be copied into an
             *                    internal buffer.
             * @param flags The flags of the option.
             * @param valueList The list of values expected to follow this 
             *                  option directly in the command line.
             */
            inline Option(const Char *longName, const Char *description, 
                    int flags, ValueDesc *valueList = NULL) {
                this->initObject(static_cast<Char>(0), longName, description, 
                    static_cast<Flags>(flags), valueList);
            }

            /** 
             * ctor.
             * 
             * @param longName The long name string of the option. Can be NULL
             *                 if no long name should be used. Otherwise the
             *                 string will be copied into an internal buffer.
             * @param description The description string of the option. Can be
             *                    NULL if no description string is provided.
             *                    Otherwise the string will be copied into an
             *                    internal buffer.
             * @param flags The flags of the option.
             * @param valueList The list of values expected to follow this 
             *                  option directly in the command line.
             */
            inline Option(const String<T>& longName, const Char *description, 
                    int flags, ValueDesc *valueList = NULL) {
                this->initObject(static_cast<Char>(0), longName.PeekBuffer(), 
                    description, static_cast<Flags>(flags), valueList);
            }            
            
            /** 
             * ctor.
             * 
             * @param longName The long name string of the option. Can be NULL
             *                 if no long name should be used. Otherwise the
             *                 string will be copied into an internal buffer.
             * @param description The description string of the option. Can be
             *                    NULL if no description string is provided.
             *                    Otherwise the string will be copied into an
             *                    internal buffer.
             * @param flags The flags of the option.
             * @param valueList The list of values expected to follow this 
             *                  option directly in the command line.
             */
            inline Option(const Char *longName, const String<T>& description, 
                    int flags, ValueDesc *valueList = NULL) {
                this->initObject(static_cast<Char>(0), longName, 
                    description.PeekBuffer(), static_cast<Flags>(flags), 
                    valueList);
            }

            /** 
             * ctor.
             * 
             * @param longName The long name string of the option. Can be NULL
             *                 if no long name should be used. Otherwise the
             *                 string will be copied into an internal buffer.
             * @param description The description string of the option. Can be
             *                    NULL if no description string is provided.
             *                    Otherwise the string will be copied into an
             *                    internal buffer.
             * @param flags The flags of the option.
             * @param valueList The list of values expected to follow this 
             *                  option directly in the command line.
             */
            inline Option(const String<T>& longName, 
                    const String<T>& description, int flags, 
                    ValueDesc *valueList = NULL) {
                this->initObject(static_cast<Char>(0), longName.PeekBuffer(), 
                    description.PeekBuffer(), static_cast<Flags>(flags), 
                    valueList);
            }

            /** dtor. */
            ~Option(void);

            /**
             * Answer the number of values of this option.
             *
             * @return The number of values of this option.
             */
            inline unsigned int GetValueCount(void) const {
                return (this->values) ? this->values->Count() : 0;
            }

            /**
             * Answer the type of the i-th value of this option.
             *
             * @return The type of the i-th value.
             *
             * @throws OutOfRangeException if i larger or equal value count.
             */
            inline ValueType GetValueType(unsigned int i) const {
                return this->values->Type(i);
            }

            /**
             * Answer the type of the i-th value of this option.
             *
             * @return The type of the i-th value.
             *
             * @throws OutOfRangeException if i larger or equal value count.
             */
            inline const String<T>& GetValueName(unsigned int i) const {
                return this->values->Name(i);
            }

            /**
             * Answer the type of the i-th value of this option.
             *
             * @return The type of the i-th value.
             *
             * @throws OutOfRangeException if i larger or equal value count.
             */
            inline const String<T>& GetValueDescription(unsigned int i) const {
                return this->values->Description(i);
            }

            /**
             * Returns the first argument in the parsers argument list which
             * matches this option, or NULL if this option was not found.
             *
             * @return Pointer to the first argument matching this option, or
             *         NULL. The argument object will be valid until the parser
             *         is destroied or the Parse methode of the parser is 
             *         called.
             */
            inline Argument *GetFirstOccurrence(void) const {
                return this->firstArg;
            }

            /**
             * Returns the first argument following 'cur' in the parsers
             * argument list which matches this option, or NULL if there is no
             * such argument.
             *
             * @return Pointer to the first argument following 'cur' matching
             *         this option, or NULL.
             */
            Argument *GetNextOccurrence(Argument *cur) const;

            /**
             * Answer the short name of the option.
             *
             * @return the short name.
             */
            inline const Char GetShortName(void) const {
                return this->shortName;
            }

            /**
             * Answer the long name of the option.
             *
             * @return The long name.
             */
            inline const String<T>& GetLongName(void) const {
                return this->longName;
            }

            /**
             * Answer the description of the option.
             *
             * @return The description.
             */
            inline const String<T>& GetDescription(void) const {
                return this->description;
            }

            /**
             * Compare operator. This is purely based on the short and the 
             * long name of the option.
             *
             * @param rhs The right hand side operand
             *
             * @return if the objects are equal
             */
            inline bool operator==(const Option& rhs) const {
                return (this->shortName == rhs.shortName)
                    && (this->longName == rhs.longName);
            }

        private:

            /**
             * Initializes the class. This methode is called from the different
             * inline constructors.
             *
             * @param shortName The single character name of the option, or 0
             *                  if no short name should be used.
             * @param longName The long name string of the option. Can be NULL
             *                 if no long name should be used. Otherwise the
             *                 string will be copied into an internal buffer.
             * @param description The description string of the option. The 
             *                    string will be copied into an internal 
             *                    buffer.
             * @param flags The flags of the option.
             * @param valueList The list of values expected to follow this 
             *                  option directly in the command line.
             */
            void initObject(const Char shortName, const Char *longName, 
                const Char *description, Flags flags, ValueDesc *valueList);

            /** forbidden copy ctor. */
            Option(const Option& rhs);

            /** forbidden assignment operator. */
            Option& operator=(const Option& rhs);

            /** access to the parser object */
            CmdLineParser<T> *parser;

            /** short name of the option */
            Char shortName;

            /** long name of the option */
            String<T> longName;

            /** description string of the option */
            String<T> description;

            /** flag indicating that this option is unique */
            bool unique;

            /** flag indicating that this option is required */
            bool required;

            /** flag indicating that this option is exclusive */
            bool exclusive;

            /** the value list of the option */
            ValueDesc *values;

            /** first occurance of this option */
            Argument *firstArg;

        };

        /**
         * typedef for SingleLinkedList of pointers to the Option objects. Be 
         * careful not to delete the object the pointer are pointing to, since
         * normally you wont own their memory.
         */
        typedef vislib::SingleLinkedList<Option*> OptionPtrList;

    private:
            
        /**
         * private iterator for the list of registered options. Neccessary
         * to avoid a gcc compiler bug not recognizing the type.
         */
        typedef typename OptionPtrList::Iterator OptionPtrIterator;

    public:

        /**
         * Nested class for parsed command line arguments building up the
         * command line list. The parser is generating an array of these
         * objects representing the interpreted command line arguments. The
         * objects memory is owned by the parser, so the objects will be valid
         * until the parser is destroied or the parsers "Parse" methode is
         * subsequently called.
         */
        class Argument {
        public:
            /** allow parser to access the members directly */
            friend class CmdLineParser<T>;

            /** possible values for type */
            enum Type {
                TYPE_UNKNOWN, // might be parameter or unknown option
                TYPE_OPTION_LONGNAME,
                TYPE_OPTION_SHORTNAMES,
                TYPE_OPTION_VALUE
            };

            /** dtor */
            ~Argument(void);

            /**
             * Answer the type of the argument.
             *
             * @return the type of the argument.
             */
            inline Type GetType(void) const {
                return this->type;
            }

            /**
             * If the type of this argument is not TYPE_UNKNOWN, this method
             * returns a pointer to the option object this argument matches.
             */
            inline Option* GetOption(void) const {
                return this->option;
            }
            
            /**
             * Answer the value type of the option of this argument, or 
             * Option::NO_VALUE if there is no option this argument matches.
             *
             * @return The value type of the option of this argument or
             *         Option::NO_VALUE.
             */
            inline typename Option::ValueType GetValueType(void) const {
                return this->valueType;
            }

            /**
             * Answer whether this argument is selected.
             *
             * @return true if this argument is selected, false otherwise.
             */
            inline bool IsSelected(void) const {
                return this->selected;
            }

            /**
             * Selectes this argument.
             */
            inline void Select(void) {
                this->selected = true;
            }

            /**
             * Deselects this argument.
             */
            inline void Deselect(void) {
                this->selected = false;
            }

            /**
             * Toggels the selection of this argument. If it was selected, it
             * becomes deselected and vis-à-vis.
             */
            inline void ToggleSelect(void) {
                this->selected = !this->selected;
            }

            /**
             * Returns the original argument string.
             *
             * @return The original argument string.
             */
            inline const Char* GetInputString(void) const {
                return this->arg;
            }

            /**
             * Returns the string value of this argument.
             *
             * @return The string value of this argument.
             *
             * @throw UnsupportedOperationException if the value type of the 
             *        option is not string type or if this argument does not
             *        have a value.
             */
            const Char* GetValueString(void) const;

            /**
             * Returns the integer value of this argument.
             *
             * @return The integer value of this argument.
             *
             * @throw UnsupportedOperationException if the value type of the 
             *        option is not integer type or if this argument does not
             *        have a value.
             * @throw FormatException if the argument string could not be
             *        converted to an integer value.
             */
            const int GetValueInt(void) const;

            /**
             * Returns the floating point value of this argument.
             *
             * @return The floating point value of this argument.
             *
             * @throw UnsupportedOperationException if the value type of the 
             *        option is not floating point type or if this argument 
             *        does not have a value.
             * @throw FormatException if the argument string could not be
             *        converted to an floating point value.
             */
            const double GetValueDouble(void) const;

            /**
             * Returns the bool value of this argument.
             *
             * @return The bool value of this argument.
             *
             * @throw UnsupportedOperationException if the value type of the 
             *        option is not boolean type or if this argument does not 
             *        have a value.
             */
            const bool GetValueBool(void) const;

        private:

            /** private ctor */
            Argument(void);

            /** forbidden copy ctor */
            Argument(const Argument& rhs);

            /** forbidden assignment operator */
            Argument& operator=(const Argument& rhs);

            /** argument string */
            Char *arg;

            /** argument id */
            unsigned int argid;

            /** character position */
            unsigned int pos;

            /** the arguments type */
            Type type;

            /** the option associated with this argument */
            Option *option;

            /** selection flag */
            bool selected;

            /** argument string of the value */
            Char *valueArg;

            /** the value type of this value */
            typename Option::ValueType valueType;

        };

        /**
         * Nested class for errors.
         */
        class Error {
        public:

            /** possible values for errorcode */
            enum ErrorCode {
                NONE = 0,
                UNKNOWN,
                NEGATIVE_ARGC,
                MISSING_VALUE,
                INVALID_VALUE,
                MISSING_REQUIRED_OPTIONS,
                MISSPLACED_EXCLUSIVE_OPTION
            };

            /**
             * Returns a human readable ansi string for the specified error code.
             *
             * @param errorcode The error code
             *
             * @return A human readable ansi string for the specified error code.
             */
            static const char * GetErrorString(ErrorCode errorcode) {
                const char *retval;
                switch(errorcode) {
                    case NONE: 
                        retval = "No Error"; 
                        break;
                    case NEGATIVE_ARGC: 
                        retval = "Parameter argc must not be negative."; 
                        break;
                    case MISSING_VALUE: 
                        retval = "Option value is missing."; 
                        break;
                    case INVALID_VALUE:
                        retval = "Option value is invalid.";
                        break;
                    case MISSING_REQUIRED_OPTIONS:
                        retval = "At least one required option is missing.";
                        break;
                    case MISSPLACED_EXCLUSIVE_OPTION:
                        retval = "An exclusive option was mixed up with other options. The exclusive option must be specified as first parameter.";
                        break;
                    case UNKNOWN: 
                        // no break
                    default: 
                        retval = "Unknown Error"; 
                        break;
                }
                return retval;
            }

            /** ctor */
            Error(void) {
                this->errorcode = UNKNOWN;
                this->argument = 0;
            }

            /**
             * ctor
             *
             * @param errorcode The error code
             * @param argument The arguments number (beginning with 0)
             */
            Error(ErrorCode errorcode, unsigned int argument) {
                this->errorcode = errorcode;
                this->argument = argument;
            }

            /**
             * Answer the error code.
             *
             * @return The error code.
             */
            inline ErrorCode GetErrorCode(void) {
                return this->errorcode;
            }

            /**
             * Answer the argument number (beginning with 0).
             *
             * @return The argument number.
             */
            inline unsigned int GetArgument(void) {
                return this->argument;
            }

            /**
             * Compare operator.
             *
             * @param rhs The right hand side operand
             *
             * @return if the objects are equal
             */
            inline bool operator==(const Error& rhs) const {
                return (this->errorcode == rhs.errorcode) && (this->argument == rhs.argument);
            }

        private:
            
            /** error code */
            ErrorCode errorcode;

            /** argument */
            unsigned int argument;
        };

        /**
         * Nested class for warnings.
         */
        class Warning {
        public:

            /** possible values for warncode */
            enum WarnCode {
                NONE = 0,
                UNKNOWN_SHORT_NAMES,
                UNIQUE_OPTION_MORE_THEN_ONCE,
                UNKNOWN
            };

            /**
             * Returns a human readable ansi string for the specified warning code.
             *
             * @param warncode The warning code
             *
             * @return A human readable ansi string for the specified warning code.
             */
            static const char * GetWarningString(WarnCode warncode) {
                const char *retval;
                switch(warncode) {
                    case NONE: 
                        retval = "No Warning"; 
                        break;
                    case UNKNOWN_SHORT_NAMES: 
                        retval = "At least one short name not recognized. Whole argument ignored."; 
                        break;
                    case UNIQUE_OPTION_MORE_THEN_ONCE: 
                        retval = "A Unique option appears more then once in the argument list.";
                        break;
                    case UNKNOWN: 
                        // no break
                    default: 
                        retval = "Unknown Warning"; 
                        break;
                }
                return retval;
            }

            /** ctor */
            Warning(void) {
                this->warncode = UNKNOWN;
                this->argument = 0;
            }

            /**
             * ctor
             *
             * @param warncode The warning code
             * @param argument The arguments number (beginning with 0)
             */
            Warning(WarnCode warncode, unsigned int argument) {
                this->warncode = warncode;
                this->argument = argument;
            }

            /**
             * Answer the error code.
             *
             * @return The error code.
             */
            inline WarnCode GetWarnCode(void) {
                return this->warncode;
            }

            /**
             * Answer the argument number (beginning with 0).
             *
             * @return The argument number.
             */
            inline unsigned int GetArgument(void) {
                return this->argument;
            }

            /**
             * Compare operator.
             *
             * @param rhs The right hand side operand
             *
             * @return if the objects are equal
             */
            inline bool operator==(const Warning& rhs) const {
                return (this->warncode == rhs.warncode) && (this->argument == rhs.argument);
            }

        private:
            
            /** warning code */
            WarnCode warncode;

            /** argument */
            unsigned int argument;
        };

        /**
         * Nested class used for returning lines of the automatically generated
         * online help on the registered options.
         */
        class OptionDescIterator: public Iterator<Char*> {
        public:
            friend class CmdLineParser<T>;

            /** 
             * copy ctor 
             *
             * @param rhs Reference to the source object
             */
            OptionDescIterator(const OptionDescIterator& rhs);

            /**
             * dtor 
             */
            virtual ~OptionDescIterator(void);

            /** 
             * assignment operator 
             *
             * @param rhs Reference to the source object
             *
             * @return Reference to this.
             */
            OptionDescIterator& operator=(const OptionDescIterator& rhs);

            /** Behaves like Iterator<T>::HasNext */
            virtual bool HasNext(void) const;

            /** 
             * Behaves like Iterator<T>::Next 
             *
             * @throw IllegalStateException if there is no next element
             */
            virtual Char*& Next(void);

        private:

            /** private init ctor */
            OptionDescIterator(OptionPtrList &opts, bool withValues);

            /** pointer to the list list of the options of the parser */
            OptionPtrList *options;

            /** iterator before next option */
            OptionPtrIterator option;

            /** formatter object */
            vislib::ColumnFormatter<T> formatter;

            /** output string */
            Char *output;

            /** flag indicating to include the value descriptions */
            bool withValues;

        };

        /** typedef for returning errors */
        typedef typename vislib::SingleLinkedList<Error>::Iterator ErrorIterator;

        /** typedef for returning warnings */
        typedef typename vislib::SingleLinkedList<Warning>::Iterator WarningIterator;

        /** Ctor. */
        CmdLineParser(void);

        /** Dtor. */
        ~CmdLineParser(void);

        /**
         * Adds the option opt to the option list of the parser. The parser 
         * will not take the ownership of opt, so the caller must ensure, that
         * the object opt is valid at this memory location, as long as it is
         * used by this parser. The object may only be freed after it has been
         * removed from the parsers option list by calling "RemoveOption(opt)",
         * or after the parser object itself has been destroied.
         *
         * @param opt The option to be added to the parsers option list. If opt
         *            is NULL, nothing happens.
         *
         * @throw IllegalParamException If there already is an option in the 
         *                              parser option list, with the same 
         *                              shortname or longname as opt, if opt
         *                              is NULL, if opt already is added to
         *                              another parser, or if opt has neither
         *                              a shortname or a longname.
         */
        void AddOption(const Option* opt);

        /**
         * Removes the option opt from the option list of the parser. If opt
         * is not on the option list, nothing is done. A pointer check is used
         * so other option objects with same short or long names wont be 
         * removed from the option list. Since the parser does not own the
         * memory of opt, it is not freed!
         *
         * @param opt The option to be removed from the parsers option list.
         *
         * @throw IllegalParamException If is NULL, or if opt is added to
         *                              another parser.
         */
        void RemoveOption(const Option* opt);

        /**
         * Cleans the option list by removing all options from the option list.
         * Since the parser does not own the memory of the options, they are 
         * not freed!
         */
        void CleanOptionList(void);          

        /**
         * Finds an option in the option list with the given short name. Short
         * name compares are case sensitive.
         *
         * @param shortname The short name of the option searching for.
         *
         * @return Pointer to the found option or NULL if no option in the
         *         parsers option list has this short name.
         */
        const Option* FindOption(const Char shortname) const;

        /**
         * Finds an option in the option list with the given long name. Long
         * name compares are case insensitive.
         *
         * @param longname The long name of the option searching for.
         *
         * @return Pointer to the found option or NULL if no option in the
         *         parsers option list has this long name.
         */
        const Option* FindOption(const Char *longname) const;

        /**
         * Finds an option in the option list with the given long name. Long
         * name compares are case insensitive.
         *
         * @param longname The long name of the option searching for.
         *
         * @return Pointer to the found option or NULL if no option in the
         *         parsers option list has this long name.
         */
        const Option* FindOption(const String<T>& longname) const;

        /**
         * Parses the command line specified by argv and argc, builds up the
         * command line list. The command line list is built up as array of 
         * Argument objects. These objects will store the interpretation of the
         * corresponding argument string, and provide direct access to 
         * recognized data.
         *
         * The Argument objects only store pointers to the argument strings
         * specified by the argv parameter. Thus the caller must ensure that
         * the memory of the argv parameter is valid and unchanged as long as
         * the Argument objects are used.
         *
         * Options can be specified by their long names using two minus 
         * characters (e.g. "--help"), which will match case insensitive, or 
         * by their short names using a single minus character (e.g. "-h"), 
         * which will match case sensitive. When multiple options are 
         * specified with their short names, they can be concatenated using 
         * only one minus character (e.g. "-a -b -c" can be written "-abc").
         * However, when specifying multiple short names in a single argument
         * all short names must match registered options, or the whole
         * argument will be classified as Argument::TYPE_UNKNOWN. If all
         * short names matches options, there will be one Argument object
         * for each short name in the command line list. Only the last short
         * name of such an concatenated argument string can support a value.
         *
         * The parser also selectes all Arguments which were classified as
         * Argument::TYPE_UNKNOWN. You can use a subsequent call to 
         * "ExtractSelectedArguments" to generate a command line holding only
         * all these arguments.
         *
         * @param argc The number of arguments in argv.
         * @param argv The array of the argument strings.
         * @param includeFirstArgument Indicates if the first argument should
         *                             be included in the parsing process. If
         *                             the first argument is the path of the
         *                             started application module it might be
         *                             helpful to ignore this first argument by
         *                             providing a value of false to avoid
         *                             missinterpretations. However, if the
         *                             first argument is not included, it will
         *                             not be part of the created command line
         *                             list of Argument objects!
         *
         * @return 0 on success, 1 if warnings are present, or -1 if errors 
         *         are present. Information about the errors and warnings can
         *         be received using "GetErrors()", and "GetWarnings()".
         */
        int Parse(int argc, Char **argv, bool includeFirstArgument = true);

        /**
         * Returns the Errors created by the last call of "Parse()".
         *
         * @return The Errors of last "Parse()"
         */
        inline ErrorIterator GetErrors() {
            return this->errors.GetIterator();
        }

        /**
         * Returns the Warnings created by the last call of "Parse()".
         *
         * @return The Warnings of last "Parse()"
         */
        inline WarningIterator GetWarnings() {
            return this->warnings.GetIterator();
        }

        /**
         * Returns the number of arguments in the command line list.
         *
         * @return The number of arguments in the command line list.
         */
        inline unsigned int ArgumentCount(void) const {
            return this->arglistSize;
        }

        /**
         * Returns a pointer to the i-th Argument object of the command line 
         * list.
         *
         * @return A pointer to the i-th argument, or NULL if i is out of 
         *         range.
         */
        inline Argument * GetArgument(unsigned int i) const {
            return (i < this->arglistSize) ? &this->arglist[i] : NULL;
        }

        /**
         * Returns a pointer to the Argument object directly following a given
         * Argument object.
         *
         * @param arg The argument provided.
         *
         * @return The directly following argument, or NULL if the given
         *         argument arg is the last one or NULL.
         */
        inline Argument * NextArgument(const Argument *arg) const {
            for (unsigned int i = 1; i < this->arglistSize; i++) {
                if (&this->arglist[i - 1] == arg) {
                    return &this->arglist[i];
                }
            }
            return NULL;
        }

        /**
         * Selects all arguments of the command line list.
         */
        inline void SelectAllArguments(void) {
            for (unsigned int i = 0; i < this->arglistSize; i++) {
                this->arglist[i].Select();
            }
        }

        /**
         * Deselects all arguments of the command line list.
         */
        inline void DeselectAllArguments(void) {
            for (unsigned int i = 0; i < this->arglistSize; i++) {
                this->arglist[i].Deselect();
            }
        }

        /**
         * Toggels selection of all arguments of the command line list. 
         * Previously selected arguments become deselected and vis-à-vis.
         */
        inline void InvertArgumentSelection(void) {
            for (unsigned int i = 0; i < this->arglistSize; i++) {
                this->arglist[i].ToggelSelect();
            }
        }

        /**
         * Creates a new command line, placed in the out parameter 
         * CmdLineProvider object, by copying all selected arguments to the 
         * out parameter.
         *
         * This CmdLineProvider object must not be the same, delivering the
         * data for a previous call to "Parse". Since "Parse" does not copy
         * the data, calling ExtractSelectedArguments would destroy the data 
         * before being able to copy it to the CmdLineProvider object.
         *
         * @param outCmdLine CmdLineProvider object to receive the selected
         *                   arguments.
         */
        void ExtractSelectedArguments(CmdLineProvider<T> &outCmdLine);

        /**
         * Creates an option description iterator which can be used to create 
         * an online help of the registered options of this parser. The 
         * iterator will enter an invalid state if the option list of the 
         * parser is changes (e.g. if the parser object ist destroied).
         * 
         * @param withValues If true, the descriptions of the option values are
         *                   also included. If false, only the descriptions of
         *                   the options are shown.
         *
         * @return The option description iterator.
         */
        inline OptionDescIterator OptionDescriptions(bool withValues = false) {
            return OptionDescIterator(this->options, withValues);
        }

        /**
         * Answer a single linked list of all missing required options. Be 
         * careful not to delete the objects the pointers in this list are
         * pointing to, since you normally wont own their memory. If there are 
         * no missing required options, a empty list will be returned.
         *
         * @return A single linked list of pointers to all missing required
         *         options.
         */
        OptionPtrList ListMissingRequiredOptions(void);

    private:

        /** forbidden copy Ctor. */
        CmdLineParser(const CmdLineParser& rhs);

        /** forbidden assignment operator. */
        CmdLineParser& operator=(const CmdLineParser& rhs);

        /** list of options */
        OptionPtrList options;

        /** the command line list created by parse */
        Argument *arglist;

        /** number of items in the command line list */
        unsigned int arglistSize;

        /** list of errors */
        vislib::SingleLinkedList<Error> errors;

        /** list of warnings */
        vislib::SingleLinkedList<Warning> warnings;
    };

    
    /*
     * CmdLineParser<T>::Option::ValueDesc::Count
     */
    template<class T> 
    unsigned int CmdLineParser<T>::Option::ValueDesc::Count(void) const {
        unsigned int i = 0;
        for (const ValueDesc *t = this; t != NULL; t = t->next) { i++; }
        return i;
    }


    /*
     * CmdLineParser<T>::Option::ValueDesc::Element
     */
    template<class T>
    const typename CmdLineParser<T>::Option::ValueDesc& 
            CmdLineParser<T>::Option::ValueDesc::element(unsigned int i) const {
        const ValueDesc *rv = this;
        while (i > 0) {
            if (!rv) {
                throw vislib::OutOfRangeException(i, 0, this->Count(), __FILE__, __LINE__);
            }
            rv = rv->next;
            i--;
        }
        return *rv;
    }


    /*
     * CmdLineParser<T>::Option::Option
     */
    template<class T> 
    CmdLineParser<T>::Option::Option(const Option& rhs) {
        throw vislib::UnsupportedOperationException("Option Copy Ctor", __FILE__, __LINE__);
    }


    /*
     * CmdLineParser<T>::Option::~Option
     */
    template<class T>
    CmdLineParser<T>::Option::~Option(void) {
        SAFE_DELETE(this->values); // paranoia
    }


    /*
     * CmdLineParser<T>::Option::initObject
     */
    template<class T>
    void CmdLineParser<T>::Option::initObject(const Char shortName, 
            const Char *longName, const Char *description, Flags flags, 
            ValueDesc *valueList) {
        this->parser = NULL;
        this->unique = ((flags & FLAG_UNIQUE) == FLAG_UNIQUE);
        this->required = ((flags & FLAG_REQUIRED) == FLAG_REQUIRED);
        this->exclusive = ((flags & FLAG_EXCLUSIVE) == FLAG_EXCLUSIVE);
        this->values = valueList;
        this->firstArg = NULL;
        this->shortName = shortName;
        this->longName = longName;
        this->description = description;
    }


    /*
     * CmdLineParser<T>::Option::GetNextOccurrence
     */
    template<class T>
    typename CmdLineParser<T>::Argument *
    CmdLineParser<T>::Option::GetNextOccurrence(
            typename CmdLineParser<T>::Argument *cur) const {
        if (this->parser == NULL) return NULL;
        if (cur == NULL) return NULL;
        unsigned int p = 0;
        while ((p < this->parser->arglistSize) && (&this->parser->arglist[p] != cur)) {
            p++;
        }
        p++;
        if (p >= this->parser->arglistSize) return NULL;
        while (p < this->parser->arglistSize) {
            if (((this->parser->arglist[p].GetType() == Argument::TYPE_OPTION_LONGNAME)
                    || (this->parser->arglist[p].GetType() == Argument::TYPE_OPTION_SHORTNAMES))
                    && (this->parser->arglist[p].GetOption() == this)) {
                return &this->parser->arglist[p];
            }
            p++;
        }
        return NULL;
    }


    /*
     * CmdLineParser<T>::Option::operator=
     */
    template<class T>
    typename CmdLineParser<T>::Option& CmdLineParser<T>::Option::operator=(const Option& rhs) {
        if (&rhs != this) {
            throw vislib::UnsupportedOperationException("Option::operator=", __FILE__, __LINE__);
        }
    }


    /*
     * CmdLineParser<T>::Argument::Argument
     */
    template<class T>
    CmdLineParser<T>::Argument::Argument(void) : arg(NULL), argid(0), pos(0), 
            type(CmdLineParser<T>::Argument::TYPE_UNKNOWN), option(NULL), 
            selected(false), valueArg(NULL), valueType(CmdLineParser<T>::Option::NO_VALUE) {
    }


    /*
     * CmdLineParser<T>::Argument::~Argument
     */
    template<class T>
    CmdLineParser<T>::Argument::~Argument(void) {
        // DO NOT DELETE option, arg OR valueArg, SINCE I DO NOT OWN THE MEMORY
    }


    /*
     * CmdLineParser<T>::Argument::Argument
     */
    template<class T>
    CmdLineParser<T>::Argument::Argument(const Argument& rhs) {
        throw UnsupportedOperationException("Argument copy ctor", __FILE__, __LINE__);
    }


    /*
     * CmdLineParser<T>::Argument::operator=
     */
    template<class T>
    typename CmdLineParser<T>::Argument& CmdLineParser<T>::Argument::operator=(const Argument& rhs) {
        if (&rhs != this) {
            throw vislib::UnsupportedOperationException("Argument::operator=", __FILE__, __LINE__);
        }
    }


    /*
     * CmdLineParser<T>::Argument::GetValueString
     */
    template<class T> 
    const typename CmdLineParser<T>::Char* CmdLineParser<T>::Argument::GetValueString(void) const {
        if (this->GetValueType() != Option::STRING_VALUE) {
            throw vislib::UnsupportedOperationException("Option value of incompatible type", __FILE__, __LINE__);
        }
        ASSERT(this->type != TYPE_UNKNOWN);

        return this->valueArg;
    }


    /*
     * CmdLineParser<T>::Argument::GetValueInt
     */
    template<class T> 
    const int CmdLineParser<T>::Argument::GetValueInt(void) const {
        if (this->GetValueType() != Option::INT_VALUE) {
            throw vislib::UnsupportedOperationException("Option value of incompatible type", __FILE__, __LINE__);
        }
        ASSERT(this->type != TYPE_UNKNOWN);

        return T::ParseInt(this->valueArg); // throws FormatException on failure
    }


    /*
     * CmdLineParser<T>::Argument::GetValueDouble
     */
    template<class T> 
    const double CmdLineParser<T>::Argument::GetValueDouble(void) const {
        if (this->GetValueType() != Option::DOUBLE_VALUE) {
            throw vislib::UnsupportedOperationException("Option value of incompatible type", __FILE__, __LINE__);
        }
        ASSERT(this->type != TYPE_UNKNOWN);

        return T::ParseDouble(this->valueArg); // throws FormatException on failure
    }

    
    /*
     * CmdLineParser<T>::Argument::GetValueBool
     */
    template<class T> 
    const bool CmdLineParser<T>::Argument::GetValueBool(void) const {
        if (this->GetValueType() != Option::BOOL_VALUE) {
            throw vislib::UnsupportedOperationException("Option value of incompatible type", __FILE__, __LINE__);
        }
        ASSERT(this->type != TYPE_UNKNOWN);

        return T::ParseBool(this->valueArg); // throws FormatException on failure
    }


    /*
     * CmdLineParser<T>::OptionDescIterator::OptionDescIterator
     */
    template<class T> 
    CmdLineParser<T>::OptionDescIterator::OptionDescIterator(const OptionDescIterator& rhs) : output(NULL) {
        *this = rhs;
    }


    /*
     * CmdLineParser<T>::OptionDescIterator::OptionDescIterator
     */
    template<class T> 
    CmdLineParser<T>::OptionDescIterator::~OptionDescIterator(void) {
        ARY_SAFE_DELETE(this->output);
    }


    /*
     * CmdLineParser<T>::OptionDescIterator::operator=
     */
    template<class T> 
    typename CmdLineParser<T>::OptionDescIterator& CmdLineParser<T>::OptionDescIterator::operator=(const OptionDescIterator& rhs) {
        if (this != &rhs) {
            this->options = rhs.options;
            this->option = rhs.option;
            this->formatter = rhs.formatter;

            ARY_SAFE_DELETE(this->output);
            unsigned int len = T::SafeStringLength(rhs.output);
            this->output = new Char[len + 1];
            this->output[len] = 0;
            if (len > 0) {
                ::memcpy(this->output, rhs.output, len * T::CharSize());
            }
        }

        return *this;
    }


    /*
     * CmdLineParser<T>::OptionDescIterator::HasNext
     */
    template<class T> 
    bool CmdLineParser<T>::OptionDescIterator::HasNext(void) const {
        return this->option.HasNext();
    }


    /*
     * CmdLineParser<T>::OptionDescIterator::Next
     */
    template<class T> 
    typename CmdLineParser<T>::Char*& CmdLineParser<T>::OptionDescIterator::Next(void) {
        Option *opt = this->option.Next();
        ARY_SAFE_DELETE(this->output);

        vislib::String<T> str;
        if (!opt->GetLongName().IsEmpty()) {
            str = vislib::String<T>(static_cast<Char>('-'), 2) + opt->GetLongName();
        }
        if (opt->GetShortName() != 0) {
            if (!str.IsEmpty()) {
                str += static_cast<Char>(' ');
            }
            str += static_cast<Char>('-');
            str += opt->GetShortName();
        }

        this->formatter[0].SetText(str);
        if (this->withValues) {
            this->formatter[1].SetText(T::EMPTY_STRING);
            this->formatter[2].SetText(opt->GetDescription());
        } else {
            this->formatter[1].SetText(opt->GetDescription());
        }
        this->formatter >> str;

        if (this->withValues) {
            if (opt->GetValueCount() > 0) {
                str += static_cast<Char>('\n');
                vislib::String<T> str2;

                this->formatter[1].SetWidth(this->formatter[0].GetWidth() - 4);
                this->formatter[0].SetWidth(2);
                this->formatter[0].SetText(T::EMPTY_STRING);
                this->formatter.SetSeparator(vislib::String<T>(static_cast<Char>(' '), 2));
                
                unsigned int cnt = opt->GetValueCount();
                for (unsigned int i = 0; i < cnt; i++) {
                    str2 = StringConverter<CharTraitsA, T>("<");
                    str2 += opt->GetValueName(i);
                    str2 += static_cast<Char>('>');
                    this->formatter[1].SetText(str2);
                    this->formatter[2].SetText(opt->GetValueDescription(i));
                    this->formatter >> str2;
                    str += str2;
                    str += static_cast<Char>('\n');
                }

                this->formatter.SetSeparator(vislib::String<T>(static_cast<Char>(' '), 1));
                this->formatter[0].SetWidth(this->formatter[1].GetWidth() + 4);
                this->formatter[1].SetWidth(0);

            } else {
                str += static_cast<Char>('\n');
            }
        }

        unsigned int len = str.Length();
        this->output = new Char[len + 1];
        this->output[len] = 0;
        if (len > 0) {
            ::memcpy(this->output, str.PeekBuffer(), len * T::CharSize());
        }

        return this->output;
    }


    /*
     * CmdLineParser<T>::OptionDescIterator::OptionDescIterator
     */
    template<class T> 
    CmdLineParser<T>::OptionDescIterator::OptionDescIterator(OptionPtrList &opts, bool withValues) 
            : formatter(withValues ? 3 : 2), output(NULL), withValues(withValues) {
        this->options = &opts;
        this->formatter.SetSeparator(vislib::String<T>(static_cast<Char>(' '), withValues ? 1 : 2));
        UINT consoleWidth = vislib::sys::Console::GetWidth();
        if (consoleWidth > 0) {
            this->formatter.SetMaxWidth(consoleWidth - 1);
        } else {
            // If retrieving the console width fails, the code above would
            // create an overflow and the formatter would consequently crash.
            this->formatter.SetMaxWidth(80);
        }
        this->formatter[1].SetWidth(0);
        if (this->withValues) {
            this->formatter[2].SetWidth(0);
        }

        unsigned int optnamelen = 0;
        this->option = this->options->GetIterator();
        while (this->option.HasNext()) {
            Option *opt = this->option.Next();
            unsigned int len = opt->GetLongName().Length();
            if (len > 0) len += 2;
            if (opt->GetShortName() != 0) {
                len += ((len > 0) ? 3 : 2);
            }

            if (len > optnamelen) optnamelen = len;

            if (this->withValues) {
                for (int i = opt->GetValueCount() - 1; i >= 0; i--) {
                    len = opt->GetValueName(i).Length() + 6;
                    if (len > optnamelen) optnamelen = len;
                }
            }
        }
        if (optnamelen > ((this->formatter.GetMaxWidth() - 2) / 4)) {
            optnamelen = ((this->formatter.GetMaxWidth() - 2) / 4);
        }
        this->formatter[0].SetWidth(optnamelen);
        this->formatter[0].DisableWrapping();

        this->option = this->options->GetIterator();

    }


    /*
     * CmdLineParser<T>::CmdLineParser
     */
    template<class T>
    CmdLineParser<T>::CmdLineParser() : arglist(NULL), arglistSize(0) {
    }
 

    /*
     * CmdLineParser<T>::CmdLineParser
     */
    template<class T>
    CmdLineParser<T>::CmdLineParser(const CmdLineParser<T>& rhs) {
        throw vislib::UnsupportedOperationException("Copy Ctor", __FILE__, __LINE__);
    }
 

    /*
     * CmdLineParser<T>::~CmdLineParser
     */
    template<class T>
    CmdLineParser<T>::~CmdLineParser() {
        ARY_SAFE_DELETE(this->arglist);
        this->CleanOptionList();
    }
 

    /*
     * CmdLineParser<T>::operator=
     */
    template<class T>
    CmdLineParser<T>& CmdLineParser<T>::operator=(const CmdLineParser<T>& rhs) {
        if (&rhs != this) {
            throw vislib::UnsupportedOperationException("operator=", __FILE__, __LINE__);
        }
    }


    /*
     * CmdLineParser<T>::AddOption
     */
    template<class T>
    void CmdLineParser<T>::AddOption(const Option* opt) {
        if ((opt == NULL) || ((opt->parser != NULL) && (opt->parser != this)) 
                || (this->FindOption(opt->shortName) != NULL) 
                || (this->FindOption(opt->longName) != NULL)
                || ((opt->shortName == 0) && (opt->longName == NULL))) {
            throw vislib::IllegalParamException("opt", __FILE__, __LINE__);
        }

        this->options.Append(const_cast<Option*>(opt));
        const_cast<Option*>(opt)->parser = this;
        const_cast<Option*>(opt)->firstArg = NULL;
    }


    /*
     * CmdLineParser<T>::RemoveOption
     */
    template<class T>
    void CmdLineParser<T>::RemoveOption(const Option* opt) {
        if ((opt == NULL) || (opt->parser != this)) {
            throw vislib::IllegalParamException("opt", __FILE__, __LINE__);
        }

        this->options.RemoveAll(const_cast<Option*>(opt));
        const_cast<Option*>(opt)->parser = NULL;
        const_cast<Option*>(opt)->firstArg = NULL;
    }


    /*
     * CmdLineParser<T>::CleanOptionList
     */
    template<class T>
    void CmdLineParser<T>::CleanOptionList(void) {
        // DO NOT FREE THE MEMORY OF THE OPTIONS !!!
        // because they might reside on the stack.
        // It's documented that the user of the class manages the memory.
        this->options.Clear();
    }


    /*
     * CmdLineParser<T>::FindOption
     */
    template<class T>
    const typename CmdLineParser<T>::Option* CmdLineParser<T>::FindOption(const Char shortname) const {
        if (shortname == 0) return NULL;

        OptionPtrIterator it = const_cast<CmdLineParser<T>*>(this)->options.GetIterator();

        while(it.HasNext()) {
            Option *o = it.Next();
            if (o->shortName == shortname) return o;
        }

        return NULL;
    }


    /*
     * CmdLineParser<T>::FindOption
     */
    template<class T>
    const typename CmdLineParser<T>::Option* CmdLineParser<T>::FindOption(
            const Char *longname) const {
        OptionPtrIterator it 
            = const_cast<CmdLineParser<T>*>(this)->options.GetIterator();

        while(it.HasNext()) {
            Option *o = it.Next();
            if (!o->longName.IsEmpty()
                    && o->longName.Equals(longname, false)) return o;
        }

        return NULL;
    }


    /*
     * CmdLineParser<T>::FindOption
     */
    template<class T>
    const typename CmdLineParser<T>::Option* CmdLineParser<T>::FindOption(
            const String<T>& longname) const {
        OptionPtrIterator it 
            = const_cast<CmdLineParser<T>*>(this)->options.GetIterator();

        while(it.HasNext()) {
            Option *o = it.Next();
            if (!o->longName.IsEmpty()
                    && o->longName.Equals(longname, false)) return o;
        }

        return NULL;
    }


    /*
     * CmdLineParser<T>::Parse
     */
    template<class T>
    int CmdLineParser<T>::Parse(int argc, Char **argv, bool includeFirstArgument) {
        ARY_SAFE_DELETE(this->arglist);
        this->arglistSize = 0;
        this->errors.Clear();
        this->warnings.Clear();

        if (argc < 0) {
            this->errors.Append(Error(Error::NEGATIVE_ARGC, 0));
            return -1; // Error: negative argc.
        }

        // temporary variables keeping record of found informations
        typename Argument::Type *argTypes = new typename Argument::Type[argc];

        OptionPtrIterator opti;

        // identify known options, values and calculate the length of the 
        // argument list to be created.
        // Also check for simple errors or warnings
        for (int i = includeFirstArgument ? 0 : 1; i < argc; i++) {
            unsigned int multi;

            if (argv[i][0] == static_cast<Char>('-')) { // option
                if (argv[i][1] == static_cast<Char>('-')) { // long name option
                    Option *o = NULL;

                    opti = this->options.GetIterator();
                    while (opti.HasNext()) {
                        Option *opt = opti.Next();
                        ASSERT(opt != NULL);
                        if (opt->longName.Equals(&argv[i][2], false)) {
                            o = opt; 
                            break;
                        }
                    }

                    if (o != NULL) { // known option
                        argTypes[i] = Argument::TYPE_OPTION_LONGNAME;
                        multi = 1;
                        bool exclusive = false;

                        if (o->exclusive) {
                            if (i == (includeFirstArgument ? 0 : 1)) {
                                // exclusive option starting the parameter list
                                for (int j = i + 1; j < argc; j++) {
                                    argTypes[j] = Argument::TYPE_UNKNOWN;
                                    this->arglistSize++;
                                }
                                exclusive = true;

                            } else {
                                // exclusive option not on first position.
                                this->errors.Append(Error(Error::MISSPLACED_EXCLUSIVE_OPTION, i));
                                argTypes[i] = Argument::TYPE_UNKNOWN;

                            }
                        }
                        if (o->GetValueCount() > 0) { // known option with value
                            if (i + static_cast<int>(o->GetValueCount()) >= argc) { 
                                // this arg is near end of list and there are not enought values!
                                this->errors.Append(Error(Error::MISSING_VALUE, i));
                                argTypes[i] = Argument::TYPE_UNKNOWN;

                            } else {
                                for (int j2 = o->GetValueCount() - 1; j2 >= 0; j2--) {
                                    i++;
                                    argTypes[i] = Argument::TYPE_OPTION_VALUE;
                                    multi++;
                                }
                            }
                        }

                        if (exclusive) i = argc;

                    } else { // unknown option
                        argTypes[i] = Argument::TYPE_UNKNOWN; // parameter or value to unknown option
                        multi = 1;

                    }

                } else { // short name option

                    // check if all short names are known!
                    bool cleanList = true;
                    bool someKnown = false;
                    bool exclusive = false;

                    multi = 0;

                    for (Char *sn = &argv[i][1]; *sn != 0; sn++) {
                        bool found = false;

                        multi++; // count short names

                        opti = this->options.GetIterator();
                        while (opti.HasNext()) {
                            Option *opt = opti.Next();
                            ASSERT(opt != NULL);
                            if (opt->shortName == *sn) {

                                if (opt->exclusive) {
                                    exclusive = true;
                                    if (i == (includeFirstArgument ? 0 : 1)) {
                                        if (sn == &argv[i][1]) {
                                            if (*(sn + 1) != 0) {
                                                // exclusive short name mixed up with other short names. This is not allowed!
                                                this->errors.Append(Error(Error::MISSPLACED_EXCLUSIVE_OPTION, i));
                                                cleanList = false;

                                            } else {
                                                // solitary exclusive short name.
                                                for (int j = i + 1; j < argc; j++) {
                                                    argTypes[j] = Argument::TYPE_UNKNOWN;
                                                    this->arglistSize++;
                                                }

                                            }

                                        } else {
                                            // short name of exclusive option not on first position
                                            this->errors.Append(Error(Error::MISSPLACED_EXCLUSIVE_OPTION, i));
                                            cleanList = false;

                                        }
                                    } else {
                                        // short name of exclusive option not on first position
                                        this->errors.Append(Error(Error::MISSPLACED_EXCLUSIVE_OPTION, i));
                                        cleanList = false;

                                    }
                                }

                                found = true;
                                someKnown = true;
                                break;
                            }
                        }

                        if (!found) cleanList = false;
                    }

                    if (cleanList) { // all short names are known
                        argTypes[i] = Argument::TYPE_OPTION_SHORTNAMES;

                        bool errmissingvalues = false;

                        for (unsigned int j = 1; j <= multi; j++) {
                            opti = this->options.GetIterator();
                            while (opti.HasNext()) {
                                Option *opt = opti.Next();
                                ASSERT(opt != NULL);
                                if (opt->shortName == argv[i][j]) {
                                    
                                    if (opt->GetValueCount() > 0) {
                                        if (j == multi) { // last of the short names
                                            if (i + static_cast<int>(opt->GetValueCount()) >= argc) { 
                                                // this arg is near end of list and there are not enought values!
                                                this->errors.Append(Error(Error::MISSING_VALUE, i));

                                            } else {
                                                for (int j2 = opt->GetValueCount() - 1; j2 >= 0; j2--) {
                                                    i++;
                                                    argTypes[i] = Argument::TYPE_OPTION_VALUE;
                                                    multi++; // use multi to count the i skipped here
                                                    j = multi; // break for the outer loop
                                                }
                                            }
                                        } else {
                                            errmissingvalues = true;
                                        }
                                    }

                                    break;
                                }
                            } /* while (opti.HasNext()) */
                        } /* for (unsigned int j = 1; j <= multi; j++) */

                        if (errmissingvalues) {
                            this->errors.Append(Error(Error::MISSING_VALUE, i));
                        }

                        if (exclusive) i = argc;

                    } else {
                        if (someKnown && !exclusive) { // some but not all short names are known
                            this->warnings.Append(Warning(Warning::UNKNOWN_SHORT_NAMES, i));
                        }

                        argTypes[i] = Argument::TYPE_UNKNOWN; // parameter or value to unknown option
                        multi = 1;

                    }

                } /* if (argv[i][1] == static_cast<Char>('-')) */

            } else {
                argTypes[i] = Argument::TYPE_UNKNOWN; // parameter or value to unknown option
                multi = 1;

            } /* if (argv[i][0] == static_cast<Char>('-')) */

            this->arglistSize += multi;

        } /* for (int i = 0; i < argc; i++) */

        // create argument list
        this->arglist = new Argument[this->arglistSize];
        this->arglistSize = 0;

        // fill argument list
        for (int i = includeFirstArgument ? 0 : 1; i < argc; i++) {
            this->arglist[this->arglistSize].arg = argv[i];
            this->arglist[this->arglistSize].argid = i;
            this->arglist[this->arglistSize].pos = 0;
            this->arglist[this->arglistSize].option = NULL;
            this->arglist[this->arglistSize].valueArg = NULL;

            switch(argTypes[i]) {
                case Argument::TYPE_OPTION_LONGNAME:
                    opti = this->options.GetIterator();
                    while (opti.HasNext()) {
                        Option *opt = opti.Next();
                        ASSERT(opt != NULL);
                        if (opt->longName.Equals(&argv[i][2], false)) {
                            this->arglist[this->arglistSize].option = opt;
                            this->arglist[this->arglistSize].type = Argument::TYPE_OPTION_LONGNAME;
                            if (opt->firstArg == NULL) {
                                opt->firstArg = &this->arglist[this->arglistSize];
                            } else {
                                if (opt->unique) { // a secound encounter of a unique option!
                                    this->warnings.Append(Warning(Warning::UNIQUE_OPTION_MORE_THEN_ONCE, i));
                                    this->arglist[this->arglistSize].type = Argument::TYPE_UNKNOWN; // because this is gonna be ignored!
                                    this->arglist[this->arglistSize].selected = true;
                                }
                            }
                            this->arglistSize++;
                            break;
                        }
                    } /* while (opti.HasNext()) */
                    break; /* TYPE_OPTION_LONGNAME */
                case Argument::TYPE_OPTION_SHORTNAMES:
                    for (Char *sn = &argv[i][1]; *sn != 0; sn++) {
                        opti = this->options.GetIterator();
                        while (opti.HasNext()) {
                            Option *opt = opti.Next();
                            ASSERT(opt != NULL);
                            if (opt->shortName == *sn) {
                                this->arglist[this->arglistSize].pos = static_cast<unsigned int>(sn - argv[i]);
                                this->arglist[this->arglistSize].option = opt;
                                this->arglist[this->arglistSize].type = Argument::TYPE_OPTION_SHORTNAMES;
                                if (opt->firstArg == NULL) {
                                    opt->firstArg = &this->arglist[this->arglistSize];
                                } else {
                                    if (opt->unique) {
                                        this->warnings.Append(Warning(Warning::UNIQUE_OPTION_MORE_THEN_ONCE, i));
                                        this->arglist[this->arglistSize].type = Argument::TYPE_UNKNOWN; // because this is gonna be ignored!
                                        this->arglist[this->arglistSize].selected = true;
                                    }
                                }
                                this->arglistSize++;
                                break;
                            }
                        } /* while (opti.HasNext()) */
                    } /* for (Char *sn = &argv[i][2]; *sn != 0; sn++) */
                    break; /* TYPE_OPTION_SHORTNAMES */
                case Argument::TYPE_OPTION_VALUE: {
                    int optIdx = -1;
                    for (int j = this->arglistSize - 1; j >= 0; j--) {
                        if ((this->arglist[j].type == Argument::TYPE_OPTION_LONGNAME)
                                || (this->arglist[j].type == Argument::TYPE_OPTION_SHORTNAMES)) {
                            optIdx = j;
                        }
                        if (this->arglist[j].type != Argument::TYPE_OPTION_VALUE) break;
                    }

                    if ((optIdx < 0) || (this->arglistSize > optIdx + this->arglist[optIdx].option->GetValueCount())) {
                        // wrong classified option value
                        // sounds like an internal error
                        // may happen to later values if one value of an option with multiple values is invalid
                        this->arglist[this->arglistSize].type = Argument::TYPE_UNKNOWN;
                        this->arglist[this->arglistSize].selected = true;

                    } else {
                        this->arglist[this->arglistSize].type = Argument::TYPE_OPTION_VALUE;
                        this->arglist[this->arglistSize].option = this->arglist[optIdx].option;
                        this->arglist[this->arglistSize].valueArg = argv[i];

                        switch (this->arglist[this->arglistSize].option->GetValueType(this->arglistSize - (optIdx + 1))) {
                            default: // implementation error!
                            case Option::NO_VALUE: // some very uneasy internal error
                                this->arglist[this->arglistSize].valueType = Option::NO_VALUE;
                                break;
                            case Option::STRING_VALUE: // cannot fail
                                this->arglist[this->arglistSize].valueType = Option::STRING_VALUE;
                                break;
                            case Option::INT_VALUE:
                                try {
                                    T::ParseInt(argv[i]);
                                    this->arglist[this->arglistSize].valueType = Option::INT_VALUE;
                                } catch(...) {
                                    this->arglist[this->arglistSize].valueType = Option::NO_VALUE;
                                }
                                break;
                            case Option::DOUBLE_VALUE:
                                try {
                                    T::ParseDouble(argv[i]);
                                    this->arglist[this->arglistSize].valueType = Option::DOUBLE_VALUE;
                                } catch(...) {
                                    this->arglist[this->arglistSize].valueType = Option::NO_VALUE;
                                }
                                break;
                            case Option::BOOL_VALUE:
                                try {
                                    T::ParseBool(argv[i]);
                                    this->arglist[this->arglistSize].valueType = Option::BOOL_VALUE;
                                } catch(...) {
                                    this->arglist[this->arglistSize].valueType = Option::NO_VALUE;
                                }
                                break;
                            case Option::INT_OR_STRING_VALUE:
                                try {
                                    T::ParseInt(argv[i]);
                                    this->arglist[this->arglistSize].valueType = Option::INT_VALUE;
                                } catch(...) {
                                    this->arglist[this->arglistSize].valueType = Option::STRING_VALUE;
                                }
                                break;
                            case Option::DOUBLE_OR_STRING_VALUE:
                                try {
                                    T::ParseDouble(argv[i]);
                                    this->arglist[this->arglistSize].valueType = Option::DOUBLE_VALUE;
                                } catch(...) {
                                    this->arglist[this->arglistSize].valueType = Option::STRING_VALUE;
                                }
                                break;
                            case Option::BOOL_OR_STRING_VALUE:
                                try {
                                    T::ParseBool(argv[i]);
                                    this->arglist[this->arglistSize].valueType = Option::BOOL_VALUE;
                                } catch(...) {
                                    this->arglist[this->arglistSize].valueType = Option::STRING_VALUE;
                                }
                                break;
                        }

                        if (this->arglist[this->arglistSize].valueType == Option::NO_VALUE) {
                            // errorhandling if the value could not be parsed to it's requestes type.
                            this->errors.Add(Error(Error::INVALID_VALUE, i));
                            this->arglist[this->arglistSize].type = Argument::TYPE_UNKNOWN;
                            this->arglist[optIdx].type = Argument::TYPE_UNKNOWN;
                            if ((this->arglist[optIdx].option) && (this->arglist[optIdx].option->firstArg == &this->arglist[optIdx])) {
                                this->arglist[optIdx].option->firstArg = NULL;
                            }
                        }

                        if (static_cast<unsigned int>(optIdx + 1) == this->arglistSize) {
                            // some sort of ugly backward compatibility
                            this->arglist[optIdx].valueArg = this->arglist[this->arglistSize].valueArg;
                            this->arglist[optIdx].valueType = this->arglist[this->arglistSize].valueType;
                        }
                    }
                    
                    this->arglistSize++;
                } break; /* TYPE_OPTION_VALUE */
                case Argument::TYPE_UNKNOWN:
                default:
                    this->arglist[this->arglistSize].type = Argument::TYPE_UNKNOWN;
                    this->arglist[this->arglistSize].selected = true;
                    this->arglistSize++;
                    break; /* TYPE_UNKNOWN / default */
            }

        } /* for (int i = 0; i < argc; i++) */

        // free temporary variables
        delete[] argTypes;

        // checking if all required options are present
        opti = this->options.GetIterator();
        while (opti.HasNext()) {
            Option *opt = opti.Next();
            if (opt->required && (opt->firstArg == NULL)) {
                this->errors.Add(Error(Error::MISSING_REQUIRED_OPTIONS, 0));
                break;
            }
        }

        /** returning the right value */
        if (this->errors.Count() > 0) {
            return -1;
        }
        if (this->warnings.Count() > 0) {
            return 1;
        }
        return 0;
    }


    /*
     * CmdLineParser<T>::ExtractSelectedArguments
     */
    template<class T>
    void CmdLineParser<T>::ExtractSelectedArguments(CmdLineProvider<T> &outCmdLine) {
        ASSERT((this->arglist == NULL) || (outCmdLine.ArgC() == 0) || (this->arglist[0].arg != outCmdLine.ArgV()[0]));

        // outCmdLine must not be the same as used for Parse!
        outCmdLine.clearArgumentList();
    
        if (this->arglist != NULL) {
            // count selected arguments
            outCmdLine.storeCount = 0;
            for (unsigned int i = 0; i < this->arglistSize; i++) {
                if (this->arglist[i].selected) outCmdLine.storeCount++;
            }

            // allocating memory
            outCmdLine.argCount = outCmdLine.storeCount;
            outCmdLine.memoryAnchor[0] = new Char*[outCmdLine.storeCount];
            outCmdLine.memoryAnchor[1] = new Char*[outCmdLine.storeCount];
            outCmdLine.arguments = outCmdLine.memoryAnchor[1];

            // copying selected arguments
            outCmdLine.storeCount = 0;
            for (unsigned int i = 0; i < this->arglistSize; i++) {
                if (this->arglist[i].selected) {
                    Char *end = this->arglist[i].arg;
                    while ((*end != 0) && (*(end + 1) != 0)) end++;

                    outCmdLine.arguments[outCmdLine.storeCount] = outCmdLine.memoryAnchor[0][outCmdLine.storeCount] = 
                        outCmdLine.createArgument(this->arglist[i].arg, end);

                    outCmdLine.storeCount++;
                }
            }

            ASSERT(outCmdLine.argCount == outCmdLine.storeCount);
        }
    }


    /*
     * CmdLineParser<T>::ListMissingRequiredOptions
     */
    template<class T> typename CmdLineParser<T>::OptionPtrList 
            CmdLineParser<T>::ListMissingRequiredOptions(void) {
        OptionPtrList newList;
        OptionPtrIterator opti = this->options.GetIterator();
        while (opti.HasNext()) {
            Option *opt = opti.Next();
            if (opt->required && (opt->firstArg == NULL)) {
                newList.Append(opt);
            }
        }

        return newList;
    }



    /** Template instantiation for ANSI strings. */
    typedef CmdLineParser<CharTraitsA> CmdLineParserA;

    /** Template instantiation for wide strings. */
    typedef CmdLineParser<CharTraitsW> CmdLineParserW;

    /** Template instantiation for TCHARs. */
    typedef CmdLineParser<TCharTraits> TCmdLineParser;

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CMDLINEPARSER_H_INCLUDED */
