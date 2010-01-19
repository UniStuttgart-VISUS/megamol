/*
 * ProjectParser.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ProjectParser.h"
#include "ModuleDescriptionManager.h"
#include "CallDescriptionManager.h"
#include "vislib/assert.h"
#include "vislib/memutils.h"
#include "vislib/VersionNumber.h"


using namespace megamol::core;
using namespace megamol::core::utility::xml;


/*
 * utility::ProjectParser::ProjectParser
 */
utility::ProjectParser::ProjectParser(void)
    : xml::ConditionalParser(), vd(NULL), viewDescs() {
    // intentionally empty
}


/*
 * utility::ProjectParser::~ProjectParser
 */
utility::ProjectParser::~ProjectParser(void) {
    if (vd != NULL) {
        SAFE_DELETE(vd);
    }
    vislib::SingleLinkedList<ViewDescription*>::Iterator
        iter = this->viewDescs.GetIterator();
    while (iter.HasNext()) {
        delete iter.Next();
    }
    this->viewDescs.Clear();
}


/*
 * utility::ProjectParser::CheckBaseTag
 */
bool utility::ProjectParser::CheckBaseTag(const utility::xml::XmlReader& reader) {
    if (!reader.BaseTag().Equals(MMXML_STRING("MegaMol"))) {
        vislib::sys::Log::DefaultLog.WriteMsg(1, 
            "Project file does not specify <MegaMol> as base tag");
        return false;
    }

    bool typeValid = false;
    bool versionValid = false;

    const vislib::Array<XmlReader::XmlAttribute>& attrib
        = reader.BaseTagAttributes();
    for (unsigned int i = 0; i < attrib.Count(); i++) {
        const XmlReader::XmlAttribute& attr = attrib[i];

        if (attr.Key().Equals(MMXML_STRING("type"))) {
            if (attr.Value().Equals(MMXML_STRING("project"))) {
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

    return typeValid && versionValid;
}


/*
 * utility::ProjectParser::StartTag
 */
bool utility::ProjectParser::StartTag(unsigned int num, unsigned int level,
        const MMXML_CHAR * name, const MMXML_CHAR ** attrib,
        utility::xml::XmlReader::ParserState state,
        utility::xml::XmlReader::ParserState& outChildState,
        utility::xml::XmlReader::ParserState& outEndTagState,
        utility::xml::XmlReader::ParserState& outPostEndTagState) {

    if (ConditionalParser::StartTag(num, level, name, attrib, state, 
            outChildState, outEndTagState, outPostEndTagState)) {
        return true; // handled by base class
    }

    if (MMXML_STRING("view").Equals(name)) {
        if (this->vd != NULL) {
            this->Error("\"view\" tag nested within another \"view\" tag ignored");
            return true;
        }
        const MMXML_CHAR *vdname = NULL;
        const MMXML_CHAR *viewmodname = NULL;
        for (int i = 0; attrib[i]; i += 2) {
            if (MMXML_STRING("name").Equals(attrib[i])) {
                vdname = attrib[i + 1];
            } else
            if (MMXML_STRING("viewmod").Equals(attrib[i])) {
                viewmodname = attrib[i + 1];
            } else {
                this->WarnUnexpectedAttribut(name, attrib[i]);
            }
        }
        if (vdname == NULL) {
            this->Error("\"view\" tag without a name ignored");
            return true;
        }

        this->vd = new ViewDescription(vislib::StringA(vdname));

        if (viewmodname == NULL) {
            this->Warning("\"view\" tag without \"viewmod\" might result in unexpected behaviour");
        } else {
            vd->SetViewModuleID(vislib::StringA(viewmodname));
        }

        return true;
    }

    if (MMXML_STRING("module").Equals(name)) {
        if (this->vd == NULL) {
            this->Error("\"module\" tag outside a \"view\" tag ignored");
            return true;
        }
        const MMXML_CHAR *className = NULL;
        const MMXML_CHAR *instName = NULL;
        for (int i = 0; attrib[i]; i += 2) {
            if (MMXML_STRING("class").Equals(attrib[i])) {
                className = attrib[i + 1];
            } else
            if (MMXML_STRING("name").Equals(attrib[i])) {
                instName = attrib[i + 1];
            } else {
                this->WarnUnexpectedAttribut(name, attrib[i]);
            }
        }
        if (instName == NULL) {
            this->Error("\"module\" tag without a name ignored");
            return true;
        }
        if (className == NULL) {
            this->Error("\"module\" tag without a class ignored");
            return true;
        }
        ModuleDescription *md = ModuleDescriptionManager::Instance()
            ->Find(vislib::StringA(className));
        if (md == NULL) {
            vislib::StringA msg;
            msg.Format("\"module\" class \"%s\" not found",
                vislib::StringA(className).PeekBuffer());
            this->Error(msg.PeekBuffer());
            return true;
        }
        this->vd->AddModule(md, vislib::StringA(instName));
        return true;
    }

    if (MMXML_STRING("call").Equals(name)) {
        if (this->vd == NULL) {
            this->Error("\"call\" tag outside a \"view\" tag ignored");
            return true;
        }
        const MMXML_CHAR *className = NULL;
        const MMXML_CHAR *fromName = NULL;
        const MMXML_CHAR *toName = NULL;
        for (int i = 0; attrib[i]; i += 2) {
            if (MMXML_STRING("class").Equals(attrib[i])) {
                className = attrib[i + 1];
            } else
            if (MMXML_STRING("from").Equals(attrib[i])) {
                fromName = attrib[i + 1];
            } else
            if (MMXML_STRING("to").Equals(attrib[i])) {
                toName = attrib[i + 1];
            } else {
                this->WarnUnexpectedAttribut(name, attrib[i]);
            }
        }
        if (className == NULL) {
            this->Error("\"call\" tag without a class ignored");
            return true;
        }
        if (fromName == NULL) {
            this->Error("\"call\" tag without a from attribute ignored");
            return true;
        }
        if (toName == NULL) {
            this->Error("\"call\" tag without a to attribute ignored");
            return true;
        }
        CallDescription *cd = CallDescriptionManager::Instance()
            ->Find(vislib::StringA(className));
        if (cd == NULL) {
            this->Error("\"call\" class not found");
            return true;
        }
        this->vd->AddCall(cd, vislib::StringA(fromName), vislib::StringA(toName));
        return true;
    }

    // TODO: Implement param tag
    // TODO: Implement instance tag
    // TODO: Implement job tag
    //       etc.

    return false;
}


/*
 * utility::ProjectParser::EndTag
 */
bool utility::ProjectParser::EndTag(unsigned int num, unsigned int level,
        const MMXML_CHAR * name, utility::xml::XmlReader::ParserState state,
        utility::xml::XmlReader::ParserState& outPostEndTagState) {
    if (ConditionalParser::EndTag(num, level, name, state, 
            outPostEndTagState)) {
        return true; // handled by base class
    }

    if (MMXML_STRING("view").Equals(name)) {
        ASSERT(this->vd != NULL);
        this->viewDescs.Add(this->vd);
        this->vd = NULL;
        return true;
    }
    if (MMXML_STRING("module").Equals(name)
            || MMXML_STRING("call").Equals(name)
            /*|| MMXML_STRING("param").Equals(name)
            || MMXML_STRING("instance").Equals(name)*/) {
        return true;
    }

    return false; // unhandled.
}
