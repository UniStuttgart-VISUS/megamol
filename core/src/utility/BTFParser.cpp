/*
 * BTFParser.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/utility/BTFParser.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "vislib/CharTraits.h"
#include "vislib/assert.h"
//#include "vislib/memutils.h"
#include "vislib/StringTokeniser.h"
#include "vislib/VersionNumber.h"
#include "vislib/sys/Log.h"
#include "vislib/xmlUtils.h"
#include <sstream>
#include <fstream>

using namespace megamol::core;
using namespace megamol::core::utility::xml;


/*
 * utility::BTFParser::XMLSTATE_CONTENT
 */
unsigned int utility::BTFParser::XMLSTATE_CONTENT = XmlReader::STATE_USER + 50;


/*
 * utility::BTFParser::BTFParser
 */
utility::BTFParser::BTFParser(ShaderSourceFactory& factory)
    : utility::xml::ConditionalParser(), factory(factory), root(),
    stack() {
    // intentionally empty
}


/*
 * utility::BTFParser::~BTFParser
 */
utility::BTFParser::~BTFParser(void) {
    // intentionally empty
}


/*
 * utility::BTFParser::SetRootElement
 */
void utility::BTFParser::SetRootElement(
        const vislib::SmartPtr<utility::BTFParser::BTFElement>& root,
        const vislib::SmartPtr<utility::BTFParser::BTFElement>& masterroot) {
    ASSERT(!root.IsNull());
    ASSERT(!root->Name().IsEmpty());
    ASSERT(root.DynamicCast<BTFNamespace>() != NULL);
    ASSERT(root.DynamicCast<BTFNamespace>()->Children().Count() == 0);
    this->root = root;

    this->stack.Clear();
    this->stack.Push(masterroot);
}


/*
 * utility::BTFParser::CharacterData
 */
void utility::BTFParser::CharacterData(unsigned int level, const XML_Char *text,
        int len, utility::xml::XmlReader::ParserState state) {
    if (state != XMLSTATE_CONTENT) return;

    ASSERT(this->stack.Peek() != NULL);
    vislib::SmartPtr<BTFElement> stacktop = *this->stack.Peek();
    BTFSnippet *snippet = stacktop.DynamicCast<BTFSnippet>();
    if (snippet == NULL) {
        this->FatalError("Internal Error: stack corruption");
        return;
    }

    MMXML_STRING newContent(text, len);
    vislib::xml::DecodeEntities(newContent);

    snippet->SetContent(snippet->Content() + vislib::StringA(newContent));
}


/*
 * utility::BTFParser::Comment
 */
void utility::BTFParser::Comment(unsigned int level, const XML_Char *text,
        utility::xml::XmlReader::ParserState state) {
    if (state != XMLSTATE_CONTENT) return;

    ASSERT(this->stack.Peek() != NULL);
    vislib::SmartPtr<BTFElement> stacktop = *this->stack.Peek();
    BTFSnippet *snippet = stacktop.DynamicCast<BTFSnippet>();
    if (snippet == NULL) {
        this->FatalError("Internal Error: stack corruption");
        return;
    }

    snippet->SetContent(snippet->Content() + vislib::StringA(text));
}


/*
 * utility::BTFParser::CheckBaseTag
 */
bool utility::BTFParser::CheckBaseTag(const utility::xml::XmlReader& reader) {
    if (!reader.BaseTag().Equals(MMXML_STRING("btf"))) {
        vislib::sys::Log::DefaultLog.WriteMsg(1, 
            "BTF file does not specify <btf/> as base tag");
        return false;
    }

    bool typeValid = false;
    bool versionValid = false;
    bool nmspcValid = false;

    const vislib::Array<XmlReader::XmlAttribute>& attrib
        = reader.BaseTagAttributes();
    for (unsigned int i = 0; i < attrib.Count(); i++) {
        const XmlReader::XmlAttribute& attr = attrib[i];

        if (attr.Key().Equals(MMXML_STRING("type"))) {
            if (attr.Value().StartsWith(MMXML_STRING("MegaMol"))) {
                typeValid = true;
            }
        } else if (attr.Key().Equals(MMXML_STRING("version"))) {
            vislib::VersionNumber ver;
            if (ver.Parse(attr.Value()) >= 1) {
                if (ver < vislib::VersionNumber(1, 0)) {
                    versionValid = false; // pre 1.0 does not exist!
                } else if (ver < vislib::VersionNumber(1, 1)) {
                    versionValid = true; // 1.0.x.x
                    this->setConditionalParserVersion(0);
                } else if (ver < vislib::VersionNumber(1, 2)) {
                    versionValid = true; // 1.1.x.x
                    this->setConditionalParserVersion(1);
                } else {
                    versionValid = false; // >= 1.2 does not exist yet!
                }
            }
        } else if (attr.Key().Equals(MMXML_STRING("namespace"))) {
            if (!this->root.IsNull()
                    && (attr.Value().Equals(MMXML_STRING(this->root->Name())))) {
                nmspcValid = true;
            }
        } else {
            this->Warning(
                vislib::StringA("Ignoring unexpected base tag attribute ") +
                vislib::StringA(attr.Key()));
        }
    }

    if (!typeValid) {
        vislib::sys::Log::DefaultLog.WriteMsg(1, 
            "base tag attribute \"type\" not present or invalid.");
        return false;
    }
    if (!versionValid) {
        vislib::sys::Log::DefaultLog.WriteMsg(1, 
            "base tag attribute \"version\" not present or invalid.");
        return false;
    }
    if (!nmspcValid) {
        vislib::sys::Log::DefaultLog.WriteMsg(1, 
            "base tag attribute \"namespace\" not present or invalid.");
        return false;
    }

    this->stack.Push(this->root);

    return true;
}


/*
 * utility::BTFParser::StartTag
 */
bool utility::BTFParser::StartTag(unsigned int num, unsigned int level,
        const XML_Char * name, const XML_Char ** attrib,
        utility::xml::XmlReader::ParserState state,
        utility::xml::XmlReader::ParserState& outChildState,
        utility::xml::XmlReader::ParserState& outEndTagState,
        utility::xml::XmlReader::ParserState& outPostEndTagState) {

    if (ConditionalParser::StartTag(num, level, name, attrib, state, 
            outChildState, outEndTagState, outPostEndTagState)) {
        return true; // handled by base class
    }
    if (this->root.IsNull()) {
        this->FatalError("Internal Error: root missing");
        return true;
    }

    ASSERT(this->stack.Peek() != NULL);
    vislib::SmartPtr<BTFElement> stacktop = *this->stack.Peek();
    BTFNamespace *nmspc = stacktop.DynamicCast<BTFNamespace>();
    BTFShader *shader = stacktop.DynamicCast<BTFShader>();
    if (shader != NULL) nmspc = NULL;
    if ((shader == NULL) && (nmspc == NULL)) {
        this->FatalError("Internal Error: stack corruption");
        return true;
    }

    vislib::SmartPtr<BTFElement> newEl;
    if (MMXML_STRING("include").Equals(name)) {
        const XML_Char *filename = NULL;
        for (int i = 0; attrib[i]; i += 2) {
            if (MMXML_STRING("file").Equals(attrib[i])) {
                if (filename == NULL) {
                    filename = attrib[i + 1];
                } else {
                    this->Warning("\"include\" tag must only have one \"file\" attribute.");
                }
            } else {
                this->WarnUnexpectedAttribut(name, attrib[i]);
            }
        }
        if (filename != NULL) {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_INFO + 150,
                "Including BTF \"%s\" ...", vislib::StringA(filename).PeekBuffer());
            if (!this->factory.LoadBTF(vislib::StringA(filename))) {
                vislib::StringA msg;
                msg.Format("Unable to include \"%s\"", vislib::StringA(filename).PeekBuffer());
                this->FatalError(msg);
                return true;
            }
        } else {
            this->Warning("\"include\" tag without \"file\" attribute ignored.");
        }
        outChildState = utility::xml::XmlReader::STATE_IGNORE_SUBTREE;
        outEndTagState = utility::xml::XmlReader::STATE_IGNORE_SUBTREE;
        return true;
    } else
    if (MMXML_STRING("namespace").Equals(name)) {
        if (nmspc != NULL) {
            const XML_Char *nname = NULL;
            for (int i = 0; attrib[i]; i += 2) {
                if (MMXML_STRING("name").Equals(attrib[i])) {
                    if (nname == NULL) {
                        nname = attrib[i + 1];
                    } else {
                        this->Warning("\"namespace\" tag must only have one \"name\" attribute.");
                    }
                } else {
                    this->WarnUnexpectedAttribut(name, attrib[i]);
                }
            }
            if (nname != NULL) {
                newEl = new BTFNamespace(vislib::StringA(nname));
            } else {
                this->Warning("\"namespace\" tag without \"name\" attribute ignored.");
            }
        } else {
            this->Warning("\"namespace\" tag ignored: only valid within root or namespace tag.");
        }
    } else
    if (MMXML_STRING("snippet").Equals(name)) {
        SIZE_T line = this->Reader().XmlLine();
        vislib::StringA file = this->root->Name();
        if (nmspc != NULL) {
            const XML_Char *sname = NULL;
            const XML_Char *stype = NULL;

            for (int i = 0; attrib[i]; i += 2) {
                if (MMXML_STRING("name").Equals(attrib[i])) {
                    if (sname == NULL) {
                        sname = attrib[i + 1];
                    } else {
                        this->Warning("\"snippet\" tag must only have one \"name\" attribute.");
                    }
                } else if (MMXML_STRING("type").Equals(attrib[i])) {
                    if (stype == NULL) {
                        stype = attrib[i + 1];
                    } else {
                        this->Warning("\"snippet\" tag must only have one \"type\" attribute.");
                    }
                } else {
                    this->WarnUnexpectedAttribut(name, attrib[i]);
                }
            }
            if (sname == NULL) {
                this->Error("\"snippet\" tags within an \"namespace\" must have a \"name\" atribute.");
            } else {
                vislib::StringA snameA(sname);
                if (snameA.Find("::") != vislib::StringA::INVALID_POS) {
                    this->Error("\"snippet\" name is invalid");
                } else if (nmspc->FindChild(snameA).IsNull()) {
                    newEl = new BTFSnippet(snameA);
                    newEl.DynamicCast<BTFSnippet>()->SetLine(line);
                    newEl.DynamicCast<BTFSnippet>()->SetFile(file);
                    if (stype == NULL) {
                        this->Warning("\"snippet\" without type: assuming \"string\".");
                        newEl.DynamicCast<BTFSnippet>()->SetType(BTFSnippet::STRING_SNIPPET);
                    } else if (MMXML_STRING("version").Equals(stype)) {
                        newEl.DynamicCast<BTFSnippet>()->SetType(BTFSnippet::VERSION_SNIPPET);
                    } else if (MMXML_STRING("file").Equals(stype)) {
                        newEl.DynamicCast<BTFSnippet>()->SetType(BTFSnippet::FILE_SNIPPET);
                    } else {
                        if (!MMXML_STRING("string").Equals(stype)) {
                            this->Warning("\"snippet\" with unknown type: using \"string\".");
                        }
                        newEl.DynamicCast<BTFSnippet>()->SetType(BTFSnippet::STRING_SNIPPET);
                    }
                    outChildState = XMLSTATE_CONTENT;
                } else {
                    vislib::StringA msg;
                    msg.Format("The namespace already contains a child named \"%s\"",
                        snameA.PeekBuffer());
                    this->Error(msg);
                }
            }
        } else if (shader != NULL) {
            const XML_Char *sname = NULL;
            const XML_Char *stype = NULL;
            const XML_Char *sid = NULL;

            for (int i = 0; attrib[i]; i += 2) {
                if (MMXML_STRING("name").Equals(attrib[i])) {
                    if (sname == NULL) {
                        sname = attrib[i + 1];
                    } else {
                        this->Warning("\"snippet\" tag must only have one \"name\" attribute.");
                    }
                } else if (MMXML_STRING("id").Equals(attrib[i])) {
                    if (sid == NULL) {
                        sid = attrib[i + 1];
                    } else {
                        this->Warning("\"snippet\" tag must only have one \"id\" attribute.");
                    }
                } else if (MMXML_STRING("type").Equals(attrib[i])) {
                    if (stype == NULL) {
                        stype = attrib[i + 1];
                    } else {
                        this->Warning("\"snippet\" tag must only have one \"type\" attribute.");
                    }
                } else {
                    this->WarnUnexpectedAttribut(name, attrib[i]);
                }
            }
            newEl = new BTFSnippet(vislib::StringA(sname));
            newEl.DynamicCast<BTFSnippet>()->SetLine(line);
            newEl.DynamicCast<BTFSnippet>()->SetFile(file);
            if (stype == NULL) {
                newEl.DynamicCast<BTFSnippet>()->SetType(BTFSnippet::EMPTY_SNIPPET);
            } else if (MMXML_STRING("version").Equals(stype)) {
                newEl.DynamicCast<BTFSnippet>()->SetType(BTFSnippet::VERSION_SNIPPET);
                outChildState = XMLSTATE_CONTENT;
            } else if (MMXML_STRING("file").Equals(stype)) {
                newEl.DynamicCast<BTFSnippet>()->SetType(BTFSnippet::FILE_SNIPPET);
                outChildState = XMLSTATE_CONTENT;
            } else {
                if (!MMXML_STRING("string").Equals(stype)) {
                    this->Warning("\"snippet\" with unknown type: using \"string\".");
                }
                newEl.DynamicCast<BTFSnippet>()->SetType(BTFSnippet::STRING_SNIPPET);
                outChildState = XMLSTATE_CONTENT;
            }
            if (sid != NULL) {
                vislib::StringA sidA(sid);
                if (shader->NameIDs().Contains(sidA)) {
                    vislib::StringA msg;
                    msg.Format("\"id\" attribute of \"snippet\" tag ignored: parent shader already has a name id \"%s\"",
                        sidA.PeekBuffer());
                    this->Warning(msg);
                } else {
                    shader->NameIDs()[sidA] = static_cast<unsigned int>(
                        shader->Children().Count());
                }
            }
        } else {
            this->Warning("\"snippet\" tag ignored: only valid within root, namespace, or shader tag.");
        }
    } else
    if (MMXML_STRING("shader").Equals(name)) {
        const XML_Char *sname = NULL;
        for (int i = 0; attrib[i]; i += 2) {
            if (MMXML_STRING("name").Equals(attrib[i])) {
                if (sname == NULL) {
                    sname = attrib[i + 1];
                } else {
                    this->Warning("\"shader\" tag must only have one \"name\" attribute.");
                }
            } else {
                this->WarnUnexpectedAttribut(name, attrib[i]);
            }
        }
        if (sname == NULL) {
            this->Error("\"shader\" tags must have a \"name\" atribute.");
        } else if (nmspc != NULL) {
            vislib::StringA snameA(sname);
            if (!nmspc->FindChild(snameA).IsNull()) {
                vislib::StringA msg;
                msg.Format("The namespace already contains a child named \"%s\"",
                    snameA.PeekBuffer());
                this->Error(msg);
            } else {
                newEl = new BTFShader(snameA);
            }
        } else if (shader != NULL) {
            vislib::StringA snameA(sname);
            // this is always a reference, so search for the name!
            vislib::Stack<vislib::SmartPtr<BTFElement> > stck(this->stack);
            vislib::Array<vislib::StringA> namepath = vislib::StringTokeniserA::Split(snameA, "::");
            if (namepath.Count() <= 0) {
                this->Warning("\"shader\" tag ignored: empty or invalid \"name\" attribute");
            } else {
                if (namepath[0].IsEmpty()) {
                    // global namespace operator
                    while (stck.Count() > 1) stck.RemoveTop();
                    namepath.RemoveFirst();
                }
                if (namepath.Count() <= 0) {
                    this->Warning("\"shader\" tag ignored: empty or invalid \"name\" attribute");
                } else while ((stck.Count() > 0) && (newEl == NULL)) {
                    vislib::SmartPtr<BTFElement> rn = *stck.Peek();
                    ASSERT(!rn.IsNull());

                    for (SIZE_T i = 0; i < namepath.Count(); i++) {
                        if (rn.DynamicCast<BTFShader>() != NULL) {
                            rn = rn.DynamicCast<BTFShader>()->FindChild(namepath[i]);
                        } else if (rn.DynamicCast<BTFNamespace>() != NULL) {
                            rn = rn.DynamicCast<BTFNamespace>()->FindChild(namepath[i]);
                        } else break;
                        if (rn.IsNull()) break;
                    }
                    if ((!rn.IsNull()) && (rn.DynamicCast<BTFShader>() != NULL)) {
                        newEl = rn;
                    }

                    if (newEl.IsNull()) {
                        stck.RemoveTop();
                    }
                }
                if (newEl.IsNull()) {
                    vislib::StringA msg;
                    msg.Format("\"shader\" tag ignored: unable to find referred shader \"%s\"",
                        snameA.PeekBuffer());
                    this->Warning(msg);
                } else {
                    // test for cycle
                    stck = this->stack;
                    while (!stck.IsEmpty()) {
                        if (stck.Pop() == newEl) {
                            this->Error("\"shader\" tag ignored: reference would create a cycle");
                            newEl = NULL;
                        }
                    }
                }
            }
        } else {
            this->Warning("\"shader\" tag ignored: only valid within root or namespace tag.");
        }
    }
    if (!newEl.IsNull()) {
        this->stack.Push(newEl);
        if (nmspc != NULL) nmspc->Children().Append(newEl);
        else if (shader != NULL) shader->Children().Append(newEl);
        else ASSERT(false);
        return true;
    } else {
        outChildState = utility::xml::XmlReader::STATE_IGNORE_SUBTREE;
        outEndTagState = utility::xml::XmlReader::STATE_IGNORE_SUBTREE;
        outPostEndTagState = state;
    }

    return false;
}


/*
 * utility::BTFParser::EndTag
 */
bool utility::BTFParser::EndTag(unsigned int num, unsigned int level,
        const XML_Char * name, utility::xml::XmlReader::ParserState state,
        utility::xml::XmlReader::ParserState& outPostEndTagState) {
    if (ConditionalParser::EndTag(num, level, name, state, 
            outPostEndTagState)) {
        return true; // handled by base class
    }
    if (this->root.IsNull()) {
        return false; // die silently
    }

    if (MMXML_STRING("include").Equals(name)) {
        return true;
    }
    if (MMXML_STRING("snippet").Equals(name)) {
        vislib::SmartPtr<BTFElement> e = this->stack.Pop();
        ASSERT(!e.IsNull());
        BTFSnippet *s = e.DynamicCast<BTFSnippet>();
        ASSERT(s != NULL);
        ASSERT(this->stack.Peek() != NULL);
        vislib::SmartPtr<BTFElement> p = *this->stack.Peek();

        if (s->Type() == BTFSnippet::VERSION_SNIPPET) {
            vislib::StringA str = s->Content();
            str.TrimSpaces();
            try {
                /*int vn = */vislib::CharTraitsA::ParseInt(str);
                s->SetContent(str);
            } catch(...) {
                this->Warning("\"snippet\" tag ignored: unable to parse version number content.");
                BTFNamespace *pn = p.DynamicCast<BTFNamespace>();
                ASSERT(pn != NULL);
                pn->Children().RemoveAll(e);
            }
            return true;
        } else if (s->Type() == BTFSnippet::FILE_SNIPPET) {
            vislib::StringA str = s->Content();

            s->SetType(BTFSnippet::STRING_SNIPPET);
            s->SetFile(str);
            s->SetLine(0);
            try {
                if (!this->Reader().GetPath().IsEmpty()) {
                    auto path = this->Reader().GetPath();
                    std::ifstream infile(path + "/" + str);
                    if (infile.is_open()) {
                        std::stringstream cont;
                        cont << infile.rdbuf();
                        s->SetContent(cont.str().c_str());
                    } else {
                        std::stringstream msg;
                        msg << "cannot open include '" << str << "' for snippet '" << s->Name() << "'";
                        this->Error(msg.str().c_str());
                    }
                } else {
                    std::stringstream msg;
                    msg << "cannot find include '" << str << "' for snippet '" << s->Name() << "' : path is unknown";
                    this->Error(msg.str().c_str());
                }
            } catch (...) {
                std::stringstream msg;
                msg << "cannot load referenced snippet file " << str;
                this->Error(msg.str().c_str());
                s->SetContent("");
            }
        }

        BTFShader *ps = p.DynamicCast<BTFShader>();
        if (ps != NULL) {
            if (s->Name().IsEmpty()) {
                // cannot be a reference. We are done!
                return true;
            }
            if (s->Type() != BTFSnippet::EMPTY_SNIPPET) {
                // type was present, so this is no reference!
                // check if name already present
                if (ps->FindChild(s->Name()) != e) {
                    vislib::StringA msg;
                    msg.Format("The shader already contains a child named \"%s\"",
                        s->Name().PeekBuffer());
                    this->Error(msg.PeekBuffer());
                    ps->Children().RemoveAll(e);
                }
                return true;
            }
            // This is a reference.
            {
                vislib::StringA c = s->Content();
                c.TrimSpaces();
                if (!c.IsEmpty()) {
                    this->Warning("Inner text of referencing \"snippet\" tag ignored");
                }
            }

            vislib::SmartPtr<BTFElement> newEl;
            vislib::Stack<vislib::SmartPtr<BTFElement> > stck(this->stack);
            vislib::Array<vislib::StringA> namepath = vislib::StringTokeniserA::Split(s->Name(), "::");
            if (namepath.Count() <= 0) {
                this->Warning("\"snippet\" tag ignored: empty or invalid \"name\" attribute");
                ps->Children().RemoveAll(e);
            } else {
                if (namepath[0].IsEmpty()) {
                    // global namespace operator
                    while (stck.Count() > 1) stck.RemoveTop();
                    namepath.RemoveFirst();
                }
                if (namepath.Count() <= 0) {
                    this->Warning("\"snippet\" tag ignored: empty or invalid \"name\" attribute");
                    ps->Children().RemoveAll(e);
                } else while ((stck.Count() > 0) && (newEl == NULL)) {
                    vislib::SmartPtr<BTFElement> rn = *stck.Peek();
                    ASSERT(!rn.IsNull());

                    for (SIZE_T i = 0; i < namepath.Count(); i++) {
                        if (rn.DynamicCast<BTFShader>() != NULL) {
                            rn = rn.DynamicCast<BTFShader>()->FindChild(namepath[i]);
                        } else if (rn.DynamicCast<BTFNamespace>() != NULL) {
                            rn = rn.DynamicCast<BTFNamespace>()->FindChild(namepath[i]);
                        } else break;
                        if (rn == e) {
                            rn = NULL;
                            break;
                        }
                        if (rn.IsNull()) break;
                    }
                    if ((!rn.IsNull()) && (rn.DynamicCast<BTFSnippet>() != NULL)) {
                        newEl = rn;
                    }

                    if (newEl.IsNull()) {
                        stck.RemoveTop();
                    }
                }
                if (newEl.IsNull()) {
                    vislib::StringA msg;
                    msg.Format("\"snippet\" tag ignored: unable to find referred snippet \"%s\"",
                        s->Name().PeekBuffer());
                    ps->Children().RemoveAll(e);
                    this->Warning(msg);
                } else {
                    (*ps->Children().Find(e)) = newEl;
                }
            }
        }
        return true;
    }
    if (MMXML_STRING("namespace").Equals(name)) {
        this->stack.Pop();
        return true;
    }
    if (MMXML_STRING("shader").Equals(name)) {
        this->stack.Pop();
        return true;
    }

    return false; // unhandled.
}
