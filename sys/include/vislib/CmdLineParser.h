/*
 * CmdLineParser.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CMDLINEPARSER_H_INCLUDED
#define VISLIB_CMDLINEPARSER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/types.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/IllegalParamException.h"
#include "vislib/FormatException.h"
#include "vislib/assert.h"
#include "vislib/CharTraits.h"
#include "vislib/String.h"
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

            /** possible variable types */
            enum ValueType {
                NO_VALUE,
                STRING_VALUE,
                INT_VALUE,
                DOUBLE_VALUE
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
             * @param valueType The variable to be expected to follow this
             *                  option directly in the command line.
             */
            Option(const Char shortName, const Char *longName = NULL, 
                const Char *description = NULL, ValueType valueType = NO_VALUE);

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
             * @param valueType The variable to be expected to follow this
             *                  option directly in the command line.
             */
            Option(const Char shortName, const String<T>& longName, 
                const Char *description = NULL, ValueType valueType = NO_VALUE);

            /** 
             * ctor.
             * 
             * @param shortName The single character name of the option, or 0
             *                  if no short name should be used.
             * @param longName The long name string of the option. The string 
             *                 will be copied into an internal buffer.
             * @param description The description string of the option. The 
             *                    string will be copied into an internal 
             *                    buffer.
             * @param valueType The variable to be expected to follow this
             *                  option directly in the command line.
             */
            Option(const Char shortName, const Char *longName, 
                const String<T>& description, ValueType valueType = NO_VALUE);

            /** 
             * ctor.
             * 
             * @param shortName The single character name of the option, or 0
             *                  if no short name should be used.
             * @param longName The long name string of the option. Can be NULL
             *                 if no long name should be used. Otherwise the
             *                 string will be copied into an internal buffer.
             * @param description The description string of the option. The 
             *                    string will be copied into an internal 
             *                    buffer.
             * @param valueType The variable to be expected to follow this
             *                  option directly in the command line.
             */
            Option(const Char shortName, const String<T>& longName, 
                const String<T>& description, ValueType valueType = NO_VALUE);

            /** dtor. */
            ~Option(void);

            /**
             * Answer the value type of this option.
             *
             * @return The value type of this option.
             */
            inline ValueType GetValueType(void) const {
                return this->valueType;
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
            inline Argument *GetFirstOccurrence(void) {
                return this->firstArg;
            }

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
             * Compare operator.
             *
             * @param rhs The right hand side operand
             *
             * @return if the objects are equal
             */
            inline bool operator==(const Option& rhs) const {
                return (this->shortName == rhs.shortName)
                    && (this->longName == rhs.longName)
                    && (this->valueType == rhs.valueType);
            }
        private:

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

            /** the value type of the option */
            ValueType valueType;

            /** first occurance of this option */
            Argument *firstArg;

        };

    private:
            
        /**
         * private iterator for the list of registered options. Neccessary
         * to avoid a gcc compiler bug not recognizing the type.
         */
        typedef typename vislib::SingleLinkedList<Option*>::Iterator OptionPtrIterator;

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
            inline Type GetType(void) {
                return this->type;
            }

            /**
             * If the type of this argument is not TYPE_UNKNOWN, this method
             * returns a pointer to the option object this argument matches.
             */
            inline Option* GetOption(void) {
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
                return this->option ? this->option->GetValueType() : Option::NO_VALUE;
            }

            /**
             * Answer whether this argument is selected.
             *
             * @return true if this argument is selected, false otherwise.
             */
            inline bool IsSelected(void) {
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
                NEGATIVE_ARGC
            };

            /**
             * Returns a human readable ansi string for the specified error code.
             *
             * @param errorcode The error code
             *
             * @return A human readable ansi string for the specified error code.
             */
            static char * GetErrorString(ErrorCode errorcode) {
                char *retval;
                switch(errorcode) {
                    case NONE: retval = "No Error"; break;
                    case NEGATIVE_ARGC: retval = "Parameter argc must not be negative"; break;
                    case UNKNOWN: // no break
                    default: retval = "Unknown Error"; break;
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
                MISSING_VALUE,
                UNKNOWN_SHORT_NAMES,
                UNKNOWN
            };

            /**
             * Returns a human readable ansi string for the specified warning code.
             *
             * @param warncode The warning code
             *
             * @return A human readable ansi string for the specified warning code.
             */
            static char * GetWarningString(WarnCode warncode) {
                char *retval;
                switch(warncode) {
                    case NONE: retval = "No Warning"; break;
                    case MISSING_VALUE: retval = "Option value is missing"; break;
                    case UNKNOWN_SHORT_NAMES: retval = "At least one short name not recognized. Whole argument ignored."; break;
                    case UNKNOWN: // no break
                    default: retval = "Unknown Warning"; break;
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
            OptionDescIterator(vislib::SingleLinkedList<Option*> &opts);

            /** pointer to the list list of the options of the parser */
            vislib::SingleLinkedList<Option*> *options;

            /** iterator before next option */
            OptionPtrIterator option;

            /** formatter object */
            vislib::ColumnFormatter<T> formatter;

            /** output string */
            Char *output;

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
         *
         * @return 0 on success, 1 if warnings are present, or -1 if errors 
         *         are present. Information about the errors and warnings can
         *         be received using "GetErrors()", and "GetWarnings()".
         */
        int Parse(int argc, Char **argv);

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
        inline unsigned int ArgumentCount(void) {
            return this->arglistSize;
        }

        /**
         * Returns a pointer to the i-th Argument object of the command line 
         * list.
         *
         * @return A pointer to the i-th argument, or NULL if i is out of 
         *         range.
         */
        inline Argument * GetArgument(unsigned int i) {
            return (i < this->arglistSize) ? &this->arglist[i] : NULL;
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
         * @return The option description iterator.
         */
        inline OptionDescIterator OptionDescriptions(void) {
            return OptionDescIterator(this->options);
        }

    private:

        /** forbidden copy Ctor. */
        CmdLineParser(const CmdLineParser& rhs);

        /** forbidden assignment operator. */
        CmdLineParser& operator=(const CmdLineParser& rhs);

        /** list of options */
        vislib::SingleLinkedList<Option*> options;

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
     * CmdLineParser<T>::Option::Option
     */
    template<class T> 
    CmdLineParser<T>::Option::Option(const Char shortName, const Char *longName, 
            const Char *description, ValueType valueType) 
            : parser(NULL), firstArg(NULL) {
        this->shortName = shortName;
        this->longName = longName;
        this->description = description;
        this->valueType = valueType;
    }


    /*
     * CmdLineParser<T>::Option::Option
     */
    template<class T> 
    CmdLineParser<T>::Option::Option(const Char shortName, const String<T>& longName, 
            const Char *description, ValueType valueType) 
            : parser(NULL), firstArg(NULL) {
        this->shortName = shortName;
        this->longName = longName;
        this->description = description;
        this->valueType = valueType;
    }


    /*
     * CmdLineParser<T>::Option::Option
     */
    template<class T> 
    CmdLineParser<T>::Option::Option(const Char shortName, const Char *longName, 
            const String<T>& description, ValueType valueType) 
            : parser(NULL), firstArg(NULL) {
        this->shortName = shortName;
        this->longName = longName;
        this->description = description;
        this->valueType = valueType;
    }


    /*
     * CmdLineParser<T>::Option::Option
     */
    template<class T> 
    CmdLineParser<T>::Option::Option(const Char shortName, const String<T>& longName, 
            const String<T>& description, ValueType valueType) 
            : parser(NULL), firstArg(NULL) {
        this->shortName = shortName;
        this->longName = longName;
        this->description = description;
        this->valueType = valueType;
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
            selected(false), valueArg(NULL) {
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
        this->formatter[1].SetText(opt->GetDescription());
        this->formatter >> str;

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
    CmdLineParser<T>::OptionDescIterator::OptionDescIterator(vislib::SingleLinkedList<Option*> &opts) : formatter(2), output(NULL) {
        this->options = &opts;
        this->formatter.SetSeparator(vislib::String<T>(static_cast<Char>(' '), 2));
        this->formatter.SetMaxWidth(vislib::sys::Console::GetWidth() - 1);
        this->formatter[1].SetWidth(0);

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

        this->options.Remove(const_cast<Option*>(opt));
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
    const typename CmdLineParser<T>::Option* CmdLineParser<T>::FindOption(const Char *longname) const {
        OptionPtrIterator it = const_cast<CmdLineParser<T>*>(this)->options.GetIterator();

        while(it.HasNext()) {
            Option *o = it.Next();
            if (o->longName.CompareInsensitive(longname)) return o;
        }

        return NULL;
    }


    /*
     * CmdLineParser<T>::FindOption
     */
    template<class T>
    const typename CmdLineParser<T>::Option* CmdLineParser<T>::FindOption(const String<T>& longname) const {
        OptionPtrIterator it = const_cast<CmdLineParser<T>*>(this)->options.GetIterator();

        while(it.HasNext()) {
            Option *o = it.Next();
            if (o->longName.CompareInsensitive(longname)) return o;
        }

        return NULL;
    }


    /*
     * CmdLineParser<T>::Parse
     */
    template<class T>
    int CmdLineParser<T>::Parse(int argc, Char **argv) {
        int retval = 0;
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
        for (int i = 0; i < argc; i++) {
            unsigned int multi;

            if (argv[i][0] == static_cast<Char>('-')) { // option
                if (argv[i][1] == static_cast<Char>('-')) { // long name option
                    Option *o = NULL;

                    opti = this->options.GetIterator();
                    while (opti.HasNext()) {
                        Option *opt = opti.Next();
                        ASSERT(opt != NULL);
                        if (opt->longName.CompareInsensitive(&argv[i][2])) {
                            o = opt; 
                            break;
                        }
                    }

                    if (o != NULL) { // known option
                        argTypes[i] = Argument::TYPE_OPTION_LONGNAME;
                        multi = 1;

                        if (o->GetValueType() != Option::NO_VALUE) { // known option with value
                            if (i + 1 >= argc) { 
                                // this is last arg, so there is no value!
                                this->warnings.Append(Warning(Warning::MISSING_VALUE, i));
                                retval = 1;

                            } else {
                                i++;
                                argTypes[i] = Argument::TYPE_OPTION_VALUE;
                                multi++;

                            }
                        }

                    } else { // unknown option
                        argTypes[i] = Argument::TYPE_UNKNOWN; // parameter or value to unknown option
                        multi = 1;

                    }

                } else { // short name option

                    // check if all short names are known!
                    bool cleanList = true;
                    bool someKnown = false;

                    multi = 0;

                    for (Char *sn = &argv[i][1]; *sn != 0; sn++) {
                        bool found = false;

                        multi++; // count short names

                        opti = this->options.GetIterator();
                        while (opti.HasNext()) {
                            Option *opt = opti.Next();
                            ASSERT(opt != NULL);
                            if (opt->shortName == *sn) {
                                found = true;
                                someKnown = true;
                                break;
                            }
                        }

                        if (!found) cleanList = false;
                    }

                    if (cleanList) { // all short names are known
                        argTypes[i] = Argument::TYPE_OPTION_SHORTNAMES;

                        bool warnmissingvalues = false;

                        for (unsigned int j = 1; j <= multi; j++) {
                            opti = this->options.GetIterator();
                            while (opti.HasNext()) {
                                Option *opt = opti.Next();
                                ASSERT(opt != NULL);
                                if (opt->shortName == argv[i][j]) {
                                    
                                    if (opt->GetValueType() != Option::NO_VALUE) {
                                        if (j == multi) { // last of the short names
                                            if (i + 1 >= argc) { 
                                                // this is last arg, so there is no value!
                                                this->warnings.Append(Warning(Warning::MISSING_VALUE, i));
                                                retval = 1;

                                            } else {
                                                i++;
                                                argTypes[i] = Argument::TYPE_OPTION_VALUE;
                                                multi++;

                                            }
                                        } else {
                                            warnmissingvalues = true;
                                        }
                                    }

                                    break;
                                }
                            } /* while (opti.HasNext()) */
                        } /* for (unsigned int j = 1; j <= multi; j++) */

                        if (warnmissingvalues) {
                            this->warnings.Append(Warning(Warning::MISSING_VALUE, i));
                        }

                    } else if (someKnown) { // some but not all short names are known
                        this->warnings.Append(Warning(Warning::UNKNOWN_SHORT_NAMES, i));

                        argTypes[i] = Argument::TYPE_UNKNOWN; // parameter or value to unknown option
                        multi = 1;

                    } else { // no short names are known
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

        // fill arguemnt list
        for (int i = 0; i < argc; i++) {
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
                        if (opt->longName.CompareInsensitive(&argv[i][2])) {
                            this->arglist[this->arglistSize].option = opt;
                            this->arglist[this->arglistSize].type = Argument::TYPE_OPTION_LONGNAME;
                            if (opt->firstArg == NULL) {
                                opt->firstArg = &this->arglist[this->arglistSize];
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
                                }
                                this->arglistSize++;
                                break;
                            }
                        } /* while (opti.HasNext()) */
                    } /* for (Char *sn = &argv[i][2]; *sn != 0; sn++) */
                    break; /* TYPE_OPTION_SHORTNAMES */
                case Argument::TYPE_OPTION_VALUE:
                    this->arglist[this->arglistSize].type = Argument::TYPE_OPTION_VALUE;
                    ASSERT(this->arglistSize > 0);
                    ASSERT((this->arglist[this->arglistSize - 1].type == Argument::TYPE_OPTION_LONGNAME)
                        || (this->arglist[this->arglistSize - 1].type == Argument::TYPE_OPTION_SHORTNAMES));
                    ASSERT(this->arglist[this->arglistSize - 1].option != NULL);
                    ASSERT(this->arglist[this->arglistSize - 1].option->GetValueType() != Option::NO_VALUE);
                    this->arglist[this->arglistSize].option = this->arglist[this->arglistSize - 1].option;
                    this->arglist[this->arglistSize - 1].valueArg = argv[i];
                    this->arglist[this->arglistSize].valueArg = argv[i];
                    this->arglistSize++;
                    break; /* TYPE_OPTION_VALUE */
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

        return retval;
    }


    /*
     * CmdLineParser<T>::ExtractSelectedArguments
     */
    template<class T>
    void CmdLineParser<T>::ExtractSelectedArguments(CmdLineProvider<T> &outCmdLine) {
        ASSERT((this->arglist == NULL) || (outCmdLine.ArgC() == 0) || (this->arglist[0].arg != outCmdLine.ArgV()[0]));

        // outCmdLine must not be the same as used for Parse!
        outCmdLine.ClearArgumentList();
    
        if (this->arglist != NULL) {
            // count selected arguments
            outCmdLine.storeCount = 0;
            for (unsigned int i = 0; i < this->arglistSize; i++) {
                if (this->arglist[i].selected) outCmdLine.storeCount++;
            }

            // allocating memory
            outCmdLine.argCount = outCmdLine.storeCount;
            outCmdLine.memoryAnchor = new Char*[outCmdLine.storeCount];
            outCmdLine.arguments = new Char*[outCmdLine.storeCount];

            // copying selected arguments
            outCmdLine.storeCount = 0;
            for (unsigned int i = 0; i < this->arglistSize; i++) {
                if (this->arglist[i].selected) {
                    Char *end = this->arglist[i].arg;
                    while ((*end != 0) && (*(end + 1) != 0)) end++;

                    outCmdLine.arguments[outCmdLine.storeCount] = outCmdLine.memoryAnchor[outCmdLine.storeCount] = 
                        outCmdLine.CreateArgument(this->arglist[i].arg, end);

                    outCmdLine.storeCount++;
                }
            }

            ASSERT(outCmdLine.argCount == outCmdLine.storeCount);
        }
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
