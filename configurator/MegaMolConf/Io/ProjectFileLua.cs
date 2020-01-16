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
        internal static string SafeString(string p) {
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
                        w.WriteLine("mmCreateView(\"" + v.Name + "\", \"" + v.ViewModule.Class + "\", \"" + v.ViewModule.Name + "\")" + " --confPos=" + v.ViewModule.ConfPos.ToString());

                        if (v.Modules != null) {
                            foreach (Module m in v.Modules) {
                                string modFullName = "::" + v.Name + "::" + m.Name;
                                if (m.Name != v.ViewModule.Name) {
                                    if (!m.ConfPos.IsEmpty)
                                    {
                                        w.WriteLine("mmCreateModule(\"" + m.Class + "\", \"" + modFullName + "\")" + " --confPos=" + m.ConfPos.ToString());
                                    } else
                                    {
                                        w.WriteLine("mmCreateModule(\"" + m.Class + "\", \"" + modFullName + "\")");
                                    }
                                }
                                //if (!m.ConfPos.IsEmpty) w.Write(" confpos=\"" + m.ConfPos.ToString() + "\"");
                                if (m.Params != null) {
                                    foreach (Param p in m.Params) {
                                        w.WriteLine("mmSetParamValue(\"" + modFullName + "::" + p.Name + "\", [=[" + p.Value.ToString() + "]=])");
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

        internal void LoadFromLua(string filename) {
            string[] lines = System.IO.File.ReadAllLines(filename);
            List<View> vs = new List<View>();

            foreach(string line in lines)
            {
                if(line.Contains("mmCreateView"))
                {
                    View v = new View();
                    loadViewFromLua(ref v, lines);
                    vs.Add(v);
                }
            }

            Views = vs.ToArray();
            GeneratorComment = null;
            StartComment = null;
        }

        private void loadViewFromLua(ref View view, string[] lines) {
            // TODO: exception checks needed, currently no checks such as "throw new Exception("Module without class encountered")" are made
            List<Module> modules = new List<Module>();
            List<Call> calls = new List<Call>();
            string vName = "";

            // first, search and find all modules in the .lua
            // now SetParamValue and CreateCall can be called before CreateModule without causing problems
            for (int i = 0; i < lines.Length; ++i)
            {
                string line = lines[i];

                // trimstart removes all leading whitespaces
                if (line.TrimStart().StartsWith("mmCreateView"))
                {
                    Module m = new Module();
                    string[] elements = line.Split('"');
                    vName = elements[1]; view.Name = vName;
                    string mModuleClass = elements[3]; m.Class = mModuleClass;
                    //string[] mModuleFullName = elements[5].Split(':');
                    //m.Name = mModuleFullName.Length == 3 ? mModuleFullName[2] : mModuleFullName[0];
                    string fullname = elements[5];
                    fullname = fullname.TrimStart(':');
                    if (fullname.StartsWith(vName))
                    {
                        fullname = fullname.Substring(vName.Length);
                        fullname = fullname.TrimStart(':');
                    }
                    m.Name = fullname;

                    if (line.Contains("--confPos"))
                    {
                        string[] posLine = line.Split(new string[] { "--" }, StringSplitOptions.None);
                        try
                        {
                            Match pt = Regex.Match(posLine[1], @"\{\s*X\s*=\s*([-.0-9]+)\s*,\s*Y\s*=\s*([-.0-9]+)\s*\}");
                            if (!pt.Success) throw new Exception();
                            m.ConfPos = new Point(
                                int.Parse(pt.Groups[1].Value),
                                int.Parse(pt.Groups[2].Value));
                        }
                        catch { }
                    }

                    modules.Add(m);

                    continue;
                }

                if (line.TrimStart().StartsWith("mmCreateModule"))
                {
                    Module m = new Module();
                    string[] elements = line.Split('"');
                    string mModuleClass = elements[1]; m.Class = mModuleClass;
                    // possibly do same fix here as for mmCreateView
                    // would need to search for mmCreateView first in while file, otherwise namecheck cannot be done
                    string[] mModuleFullName = elements[3].Split(':');
                    m.Name = mModuleFullName.Length == 3 ? mModuleFullName[2] : mModuleFullName[4];

                    if (line.Contains("--confPos"))
                    {
                        string[] posLine = line.Split(new string[] { "--" }, StringSplitOptions.None);
                        try
                        {
                            Match pt = Regex.Match(posLine[1], @"\{\s*X\s*=\s*([-.0-9]+)\s*,\s*Y\s*=\s*([-.0-9]+)\s*\}");
                            if (!pt.Success) throw new Exception();
                            m.ConfPos = new Point(
                                int.Parse(pt.Groups[1].Value),
                                int.Parse(pt.Groups[2].Value));
                        }
                        catch { }
                    }

                    modules.Add(m);

                    continue;
                }
            }

            // second, search for all SetParamValue and CreateCall
            // add the found params to the according module and create the correct calls
            for (int i = 0; i < lines.Length; ++i)
            {
                string line = lines[i];

                if (line.TrimStart().StartsWith("mmSetParamValue"))
                {

                    List<Param> prms = new List<Param>();
                        
                    Param p = new Param();


                    // check next lines if it contains mmSetParam, mmCreateCall, mmCreateModule or mmCreateView 
                    // if so, go on as usual
                    // if not, concatenate consecutive lines to get 1 big line
                    // this allows paramValues to be split accross multiple line and still get interpreted as one
                    if (i < (lines.Length - 1))
                    {
                        while (i < (lines.Length - 1) && !Regex.IsMatch(lines[i + 1],
                            @"mmSetParamValue|mmCreateCall|mmCreateModule|mmCreateView"))
                        {
                            line += lines[i + 1];
                            ++i;
                        }
                    }


                    // in case of .lua files that encloses its value with [=[ and ]=]
                    string pValue = "";
                    Match vm = Regex.Match(line, @"\[\=\[(.*?)\]\=\]");
                    if (vm.Success)
                    {
                        pValue = Regex.Replace(vm.Groups[1].Value, @"\\+\""", "\"");
                    }


                    // if .lua does not use [=[ ]=] syntax, split at '"' and put it back together
                    string[] paramElements = line.Split('"');
                    string[] paramFullName = paramElements[1].Split(':');
                    if(!vm.Success)
                    {
                        pValue += paramElements[3];
                        for (int j = 4; j < paramElements.Length - 1; ++j)
                        {
                            pValue += "\"" + paramElements[j];
                        }
                    }


                    // we need to put together the full parameter name that was previously split up
                    // paramName will always start at the 7th position and each consecutive name element 
                    // has an offset of 2, so the paramName is [6]::[8]:: ... ::[6 + 2i]
                    string pName = paramFullName[6];
                    for (int j = 8; j < paramFullName.Length; j += 2)
                    {
                        pName += "::" + paramFullName[j];
                    }
                    p.Name = pName;


                    // replace every sequence of "\...\" with "\\" so the correct filepathes get saved
                    if (pName.Equals("filename", StringComparison.OrdinalIgnoreCase))
                    {
                        pValue = Regex.Replace(pValue, @"\\+", @"\\");
                    }
                    
                    p.Value = pValue;


                    // TODO?: instead of [4] do [x] to account for different paramFullNameLengths
                    foreach (Module mTemp in modules)
                    {
                        if (paramFullName[4].StartsWith(mTemp.Name))
                        {
                            List<Param> prmsTemp = new List<Param>();
                            if (mTemp.Params != null)
                            {
                                foreach (Param pTemp in mTemp.Params)
                                {
                                    prmsTemp.Add(pTemp);
                                }
                                prmsTemp.Add(p);
                                mTemp.Params = (prmsTemp.Count == 0) ? null : prmsTemp.ToArray();
                            }
                            else
                            {
                                prmsTemp.Add(p);
                                mTemp.Params = (prmsTemp.Count == 0) ? null : prmsTemp.ToArray();
                            }
                        }
                    }

                    continue;
                }


                if (line.TrimStart().StartsWith("mmCreateCall"))
                {
                    Call c = new Call();
                    string[] elements = line.Split('"');
                    string cClass = elements[1];
                    string[] cFrom = elements[3].Split(':');
                    string cFromModuleName = cFrom.Length == 5 ? cFrom[2] : cFrom[4];
                    string cFromSlot = cFrom.Length == 5 ? cFrom[4] : cFrom[6];
                    string[] cTo = elements[5].Split(':');
                    string cToModuleName = cTo.Length == 5 ? cTo[2] : cTo[4];
                    string cToSlot = cTo.Length == 5 ? cTo[4] : cTo[6];

                    c.Class = cClass;

                    foreach (Module m in modules)
                    {
                        if (cFromModuleName.StartsWith(m.Name)) {
                            c.FromModule = m;
                            c.FromSlot = cFromSlot;
                        }
                        if (cToModuleName.StartsWith(m.Name)) {
                            c.ToModule = m;
                            c.ToSlot = cToSlot;
                        }
                    }
                    calls.Add(c);

                    continue;
                }
            }

            view.Modules = (modules.Count == 0) ? null : modules.ToArray();
            view.Calls = (calls.Count == 0) ? null : calls.ToArray();
            view.Params = null; // HAZARD: currently not supported
        }

    }

}
