using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows.Forms;

namespace MegaMolConf.Io {

    /// <summary>
    /// Io class for project files version 1.0
    /// </summary>
    class ProjectFileLua : ProjectFile {

        /// <summary>
        /// Escapes paths for Lua
        /// </summary>
        private string safeString(string p) {
            return p.Replace(@"\", @"\\").Replace("\"", "\\\"");
        }

        public override void Save(string filename) {
            using (StreamWriter w = new StreamWriter(filename, false, Encoding.ASCII)) {
                if (!string.IsNullOrEmpty(GeneratorComment)) {
                    w.WriteLine("--[[ " + GeneratorComment + " --]]");
                    w.WriteLine();
                }
                if (!string.IsNullOrEmpty(StartComment)) {
                    w.WriteLine("--[[ " + StartComment + " --]]");
                    w.WriteLine();
                }

                if (Views != null) {
                    foreach (View v in Views) {
                        if (v.ViewModule == null) {
                            MessageBox.Show("Cannot save project without view module");
                            return;
                        }
                        w.WriteLine("mmCreateView(\"" + v.Name + "\", \"" + v.ViewModule.Class + "\", \"" + v.ViewModule.Name + "\")");

                        if (v.Modules != null) {
                            foreach (Module m in v.Modules) {
                                string modFullName = "::" + v.Name + "::" + m.Name;
                                if (m.Name != v.ViewModule.Name) {
                                    w.WriteLine("mmCreateModule(\"" + m.Class + "\", \"" + modFullName + "\")");
                                }
                                //if (!m.ConfPos.IsEmpty) w.Write(" confpos=\"" + m.ConfPos.ToString() + "\"");
                                if (m.Params != null) {
                                    foreach (Param p in m.Params) {
                                        w.WriteLine("mmSetParamValue(\"" + modFullName + "::" + p.Name + "\", \"" + safeString(p.Value) + "\")");
                                    }
                                }
                            }
                        }
                        if (v.Calls != null) {
                            foreach (Call c in v.Calls) {
                                w.WriteLine("mmCreateCall(\"" + c.Class + "\", \"" + "::" + v.Name + "::" + c.FromModule.Name + "::" + c.FromSlot 
                                            + "\", \"" + "::" + v.Name + "::" + c.ToModule.Name + "::" + c.ToSlot + "\")");
                            }
                        }
                    }
                }
            }
        }

        internal void LoadFromXml(System.Xml.XmlDocument doc) {
            //List<View> vs = new List<View>();
            //foreach (System.Xml.XmlNode n in doc.DocumentElement.ChildNodes) {
            //    if (n.NodeType != System.Xml.XmlNodeType.Element) continue;
            //    if (n.Name == "view") {
            //        View v = new View();
            //        loadViewFromXml(ref v, (System.Xml.XmlElement)n);
            //        vs.Add(v);
            //    }
            //}
            //Views = vs.ToArray();
            //GeneratorComment = null;
            //StartComment = null;

        }

        private void loadViewFromXml(ref View view, System.Xml.XmlElement n) {
        //    if (!n.HasAttribute("name")) throw new Exception("View without name encountered");

        //    view.Name = n.Attributes["name"].Value;
        //    string viewmodinstname = n.HasAttribute("viewmod") ? n.Attributes["viewmod"].Value : null;

        //    List<Module> modules = new List<Module>();
        //    List<Call> calls = new List<Call>();

        //    foreach (System.Xml.XmlNode c in n.ChildNodes) {
        //        if (c.NodeType != System.Xml.XmlNodeType.Element) continue;
        //        System.Xml.XmlElement oe = (System.Xml.XmlElement)c;
        //        if (oe.Name == "module") {
        //            Module m = new Module();
        //            if (!oe.HasAttribute("class")) throw new Exception("Module without class encountered");
        //            if (!oe.HasAttribute("name")) throw new Exception("Module without name encountered");
        //            m.Class = oe.Attributes["class"].Value;
        //            m.Name = oe.Attributes["name"].Value;
        //            if (oe.HasAttribute("confpos")) {
        //                try {
        //                    Match pt = Regex.Match(oe.Attributes["confpos"].Value, @"\{\s*X\s*=\s*([-.0-9]+)\s*,\s*Y\s*=\s*([-.0-9]+)\s*\}");
        //                    if (!pt.Success) throw new Exception();
        //                    m.ConfPos = new Point(
        //                        int.Parse(pt.Groups[1].Value),
        //                        int.Parse(pt.Groups[2].Value));
        //                } catch { }
        //            }

        //            List<Param> prms = new List<Param>();

        //            foreach (System.Xml.XmlNode oc in oe.ChildNodes) {
        //                if (oc.NodeType != System.Xml.XmlNodeType.Element) continue;
        //                System.Xml.XmlElement oce = (System.Xml.XmlElement)oc;
        //                if (oce.Name != "param") continue;
        //                if (!oce.HasAttribute("name")) throw new Exception("Param without name encountered");

        //                Param p = new Param();
        //                p.Name = oce.Attributes["name"].Value;
        //                p.Value = (oce.HasAttribute("value")) ? oce.Attributes["value"].Value : string.Empty;
        //                prms.Add(p);
        //            }

        //            m.Params = (prms.Count == 0) ? null : prms.ToArray();

        //            modules.Add(m);
        //            if (m.Name == viewmodinstname) {
        //                view.ViewModule = m;
        //            }
        //        }
        //        if (oe.Name == "call") {
        //            Call cl = new Call();
        //            if (!oe.HasAttribute("class")) throw new Exception("Call without class encountered");
        //            cl.Class = oe.Attributes["class"].Value;
        //            if (!oe.HasAttribute("from")) throw new Exception("Call without source encountered");
        //            string callfromname = oe.Attributes["from"].Value;
        //            if (!oe.HasAttribute("to")) throw new Exception("Call without destination encountered");
        //            string calltoname = oe.Attributes["to"].Value;

        //            foreach (Module m in modules) {
        //                if (callfromname.StartsWith(m.Name)) {
        //                    cl.FromModule = m;
        //                    Debug.Assert(callfromname[m.Name.Length] == ':');
        //                    Debug.Assert(callfromname[m.Name.Length + 1] == ':');
        //                    cl.FromSlot = callfromname.Substring(m.Name.Length + 2);
        //                }
        //                if (calltoname.StartsWith(m.Name)) {
        //                    cl.ToModule = m;
        //                    Debug.Assert(calltoname[m.Name.Length] == ':');
        //                    Debug.Assert(calltoname[m.Name.Length + 1] == ':');
        //                    cl.ToSlot = calltoname.Substring(m.Name.Length + 2);
        //                }
        //            }
        //            calls.Add(cl);
        //        }
        //    }

        //    view.Modules = (modules.Count == 0) ? null : modules.ToArray();
        //    view.Calls = (calls.Count == 0) ? null : calls.ToArray();
        //    view.Params = null; // HAZARD: currently not supported
        }

    }

}
