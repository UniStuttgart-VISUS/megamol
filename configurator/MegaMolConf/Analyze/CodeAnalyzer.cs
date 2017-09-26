using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace MegaMolConf.Analyze {

    class CodeAnalyzer {

        //static string ctags = @"T:\home\reina\src\ctags58\Win32\Debug\ctags58.exe";
        static string ctags = @"..\MegaMolConf\ctags\Win32\Debug\ctags58.exe";
        static string ctagsArgs = @"-N --c++-kinds=cm --fields=iKszt -f - ";

        public CodeAnalyzer() {
        }

        public void Analyze(List<string> paths) {
            string output = RunCtagsOnFiles(paths, ctagsArgs);
            ParseOutput(output);
        }

        public void Analyze(string path) {
            string output = RunCtagsOnFile(path, ctagsArgs);
            ParseOutput(output);
        }

        private static string RunCtagsOnFiles(List<string> paths, string args) {
            string tmpFile = Path.GetTempFileName();
            StreamWriter w = new StreamWriter(tmpFile);
            w.Write(string.Join("\n", paths.ToArray()));
            w.Close();
            Process p = new Process();
            p.StartInfo.WindowStyle = ProcessWindowStyle.Hidden;
            p.StartInfo.UseShellExecute = false;
            p.StartInfo.CreateNoWindow = true;
            p.StartInfo.RedirectStandardOutput = true;
            p.StartInfo.FileName = ctags;
            p.StartInfo.Arguments = args + "-L " + tmpFile;
            p.Start();
            string output = p.StandardOutput.ReadToEnd();
            p.WaitForExit();
            File.Delete(tmpFile);
            return output;
        }

        private static string RunCtagsOnFile(string path, string args) {
            Process p = new Process();
            p.StartInfo.WindowStyle = ProcessWindowStyle.Hidden;
            p.StartInfo.UseShellExecute = false;
            p.StartInfo.CreateNoWindow = true;
            p.StartInfo.RedirectStandardOutput = true;
            p.StartInfo.FileName = ctags;
            p.StartInfo.Arguments = args + path;
            p.Start();
            string output = p.StandardOutput.ReadToEnd();
            p.WaitForExit();
            return output;
        }

        private void ParseOutput(string output) {
            //Regex r = new Regex(@"^([^\t]+)\t([^\t]+)\t\d+;.\tkind:([^\t]+)\tnamespace:([^\t]+)(?:\tinherits:(\S+))?\r\n$");
            //string ex = @"^(\S+)\t(\S+)\t\d+;.\tkind:(\S+)\tnamespace:(\S+)(?:\tinherits:(\S+))?$";
            string ex = @"^(\S+)\t(\S+)\t/(.*?)/;""\tkind:(\S+)\t(?:namespace|class):(\S+)(?:\tinherits:(\S+))?(?:\ttyperef:none:(\w+)?)?";
            MatchCollection matches = Regex.Matches(output, ex, RegexOptions.Multiline);
            if (matches.Count > 0) {
                foreach (Match m in matches) {
                    if (m.Groups[4].Captures[0].Value.Equals("class")) {
                        Module mod = FindModule(m.Groups[1].Captures[0].Value, m.Groups[2].Captures[0].Value);
                        if (mod == null) {
                            // could be a call
                            Call call = FindCall(m.Groups[1].Captures[0].Value, m.Groups[2].Captures[0].Value);
                            if (call == null) {
                                // could be an abstract Module that sets call compatibility anyway!
                                mod = InvestigateModule(m.Groups[1].Captures[0].Value, m.Groups[2].Captures[0].Value);
                            } else {
                                call.ClassName = m.Groups[5].Captures[0].Value + "::" + m.Groups[1].Captures[0].Value;
                                call.SourceFile = m.Groups[2].Captures[0].Value;
                                // calls do not support inheritance
                                //AnalyzerData.AddModule(mod);
                            }
                        } 
                        if (mod != null) {
                            mod.ClassName = m.Groups[5].Captures[0].Value + "::" + m.Groups[1].Captures[0].Value;
                            mod.SourceFile = m.Groups[2].Captures[0].Value;
                            if (m.Groups[6].Captures.Count > 0) {
                                string[] inherits = m.Groups[6].Captures[0].ToString().Split(new char[] { ',' });
                                foreach (string inh in inherits) {
                                    Inheritance i = new Inheritance();
                                    i.name = inh;
                                    mod.Inherited.Add(i);
                                }
                            }
                            //AnalyzerData.AddModule(mod);
                        }
                    }
                }

                foreach (Match m in matches) {
                    if (m.Groups[4].Captures[0].Value.Equals("member")) {
                        string container = m.Groups[5].Captures[0].Value;
                        Module parent = AnalyzerData.GetModuleByClassName(container);
                        if (parent == null) {
                            //Debug.WriteLine("Cannot find parent " + container + " of Member " + m.Groups[1].Captures[0].Value);
                        } else {
                            string mname = m.Groups[1].Captures[0].Value;
                            //Match localm = Regex.Match(m.Groups[3].Captures[0].Value, @"(.*?)\s+(?:\*)*"+mname+@"\s*(?:\[.*?\]\s*)?(?:\s*=.*?)?;");
                            if (m.Groups[7].Captures.Count > 0) {
                                string mtype = m.Groups[7].Captures[0].Value;
                                if (Module.ClassMatches("megamol::core::CallerSlot", mtype)) {
                                    UnderstandCallerSlot(parent, mname, m.Groups[2].Captures[0].Value);
                                }
                                if (Module.ClassMatches("megamol::core::CalleeSlot", mtype)) {
                                    UnderstandCalleeSlot(parent, mname, m.Groups[2].Captures[0].Value);
                                }
                                if (Module.ClassMatches("megamol::core::param::ParamSlot", mtype)) {
                                    UnderstandParam(parent, mname, m.Groups[2].Captures[0].Value);
                                }
                            }
                        }
                    }
                }
            }
        }

        private void UpdateSourceTagCache(string header) {
            if (!allSource.ContainsKey(header)) {
                allSource[header] = ReadAllSource(header);
            }
            if (!allTags.ContainsKey(header)) {
                string tags;
                string args = @"-N --c++-kinds=+mvl --fields=iKszt --language-force=c++ -f - ";
                tags = RunCtagsOnFile(header, args);
                string src = FindSourceFile(header);
                if (!string.IsNullOrEmpty(src)) {
                    tags += "\r\n" + RunCtagsOnFile(src, args);
                }
                allTags[header] = tags;
            }
        }

        private Module InvestigateModule(string name, string header) {
            UpdateSourceTagCache(header);
            string source = allSource[header];
            string tags = allTags[header];
            
            // find the constructor
            Match m = Regex.Match(source, name + "::" + name + @".*?{(.*)}", RegexOptions.Singleline);
            if (m.Success) {
                string constrCode = m.Groups[1].Captures[0].Value;
                if (constrCode.Contains("SetCallback") || constrCode.Contains("SetCompatibleCall")) {
                    // that's some kind of module!
                    Module mod = new Module() {
                        ClassName = name,
                        SourceFile = header,
                        ExportedName = null
                    };
                    AnalyzerData.AddModule(mod);
                    return mod;
                }
            }
            return null;
        } 
        
        private Call FindCall(string callName, string header) {
            UpdateSourceTagCache(header);
            string source = allSource[header];
            string tags = allTags[header];

            // does this class have an exported name?
            bool found = false;
            MatchCollection coll = Regex.Matches(tags, @"^ClassName\t(\S+)\t/(.*?)/;""\tkind:(\S+)\t(?:namespace|class):(\S+)(?:\tinherits:(\S+))?(?:\ttyperef:none:(\w+)?)?", RegexOptions.Multiline);
            foreach (Match m in coll) {
                if (m.Groups[3].Captures[0].Value.Equals("function")
                    && m.Groups[4].Captures[0].Value.EndsWith("::" + callName)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                // that's not a call
                return null;
            }

            Match cs = Regex.Match(source, @"class\s+(?:\w+\s+)?" + callName + @".*?" +
                @"static\s+const\s+char\s*\*\s*ClassName\s*\(\s*(?:void)?\s*\)\s*{.*?\s*return\s*""(.*?)""\s*;", RegexOptions.Singleline);
            if (cs.Success) {
                return AnalyzerData.GetCallByExportedName(cs.Groups[1].Captures[0].Value);
            }
            Debug.WriteLine("Cannot find Call " + callName + "'s ClassName in " + header);
            return null;
        }

        private Module FindModule(string className, string header) {
            UpdateSourceTagCache(header);
            string source = allSource[header];
            string tags = allTags[header];

            // does this class have an exported name?
            bool found = false;
            MatchCollection coll = Regex.Matches(tags, @"^ClassName\t(\S+)\t/(.*?)/;""\tkind:(\S+)\t(?:namespace|class):(\S+)(?:\tinherits:(\S+))?(?:\ttyperef:none:(\w+)?)?", RegexOptions.Multiline);
            foreach (Match m in coll) {
                if (m.Groups[3].Captures[0].Value.Equals("function")
                    && m.Groups[4].Captures[0].Value.EndsWith("::" + className)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                // that's not a module
                return null;
            }

            Match cs = Regex.Match(source, @"class\s+(?:\w+\s+)?" + className + @".*?" +
                @"static\s+const\s+char\s*\*\s*ClassName\s*\(\s*(?:void)?\s*\)\s*{\s*return\s*""(.*?)""\s*;", RegexOptions.Singleline);
            if (cs.Success) {
                return AnalyzerData.GetModuleByExportedName(cs.Groups[1].Captures[0].Value);
            }
            Debug.WriteLine("Cannot find Module " + className + "'s ClassName in " + header);
            return null;
        }

        private void UnderstandCalleeSlot(Module mod, string name, string header) {
            UpdateSourceTagCache(header);
            string source = allSource[header];
            string tags = allTags[header];
            string exported = "";
            string descr = "";

            // find constructor for exported name and description
            Match constr = Regex.Match(source, name + @"\s*\((?:\s*new\s+CalleeSlot\()?\s*""(.*?)""\s*,\s*""(.*?)""\s*\)", RegexOptions.Singleline);
            if (constr.Success) {
                exported = constr.Groups[1].Captures[0].Value;
                descr = constr.Groups[2].Captures[0].Value;
                CalleeSlot cs = new CalleeSlot() {
                    ExportedName = exported,
                    Description = descr
                };
                MatchCollection cbs = Regex.Matches(source, name + @"(?:\.|->)SetCallback\s*\(.*?""(.*?)"".*?\)", RegexOptions.Singleline);
                foreach (Match m in cbs) {
                    string cbName = m.Groups[1].Captures[0].Value;
                    cs.Callbacks.Add(cbName);
                }
                if (cs.Callbacks.Count > 0) {
                    mod.CalleeSlots.Add(cs);
                } else {
                    Debug.WriteLine("CalleeSlot without Functions: " + cs.ExportedName);
                }
            } else {
                // TODO should only happen for: Call
                Debug.WriteLine("Cannot find constructor for " + name);
            }

            // TODO: what about the callback anyway?
        }

        private void UnderstandCallerSlot(Module mod, string name, string header) {
#if false
            UpdateSourceTagCache(header);
            string source = allSource[header];
            string tags = allTags[header];
            string exported = "";
            string descr = "";

            // find constructor for exported name and description
            Match constr = Regex.Match(source, name + @"\s*\((?:\s*new\s+CallerSlot\()?\s*""(.*?)""\s*,\s*""(.*?)""\s*\)", RegexOptions.Singleline);
            if (constr.Success) {
                exported = constr.Groups[1].Captures[0].Value;
                descr = constr.Groups[2].Captures[0].Value;
            } else {
                // TODO should only happen for: Call, MuxRenderer3D
                Debug.WriteLine("Cannot find constructor for " + name);
            }

            // find compatible call / type
            string [] variants = { (name + @"(?:\.|->)SetCompatibleCall\s*<\s*(.*?)\s*>\s*\(\);") };
            foreach (string variant in variants) {
                MatchCollection matches = Regex.Matches(source, variant, RegexOptions.Singleline);
                if (matches.Count > 0) {
                    string type = "";
                    foreach (Match m in matches) {
                        Match m2 = Regex.Match(m.Groups[1].Captures[0].Value, @"<(.*?)>", RegexOptions.Singleline);
                        if (m2.Success) {
                            type = m2.Groups[1].Captures[0].Value;
                        } else {
                            type = m.Groups[1].Captures[0].Value;
                        }
                        // TODO: mehrfach!!
                    }
                    mod.AddCaller(name, type, exported, descr);
                    break;
                } else {
                    // TODO should only happen for: Call, MuxRenderer3D
                    // TODO DataFileSequence.inDataSlot is a problem !
                    Debug.WriteLine("Cannot find compatible call for " + name);
                }
            }
#endif
        }

        private void UnderstandParam(Module mod, string name, string header) {
            UpdateSourceTagCache(header);
            string source = allSource[header];
            string tags = allTags[header];
            string exported = "";
            string descr = "";

            // find constructor for exported name and description
            Match constr = Regex.Match(source, name + @"\s*\(\s*""(.*?)""\s*,\s*""(.*?)""\s*\)", RegexOptions.Singleline);
            if (constr.Success) {
                exported = constr.Groups[1].Captures[0].Value;
                descr = constr.Groups[2].Captures[0].Value;
            } else {
                // TODO: should only happen for MuxRenderer3D rendererActiveSlot
                Debug.WriteLine("Cannot find constructor for " + name);
            }

            // find type
            string [] variants = { (name + @"(?:\.|->)SetParameter\((.*?)\);"),
                (name + @"\s*<<\s*(.*?);") };
            foreach (string variant in variants) {
                MatchCollection matches = Regex.Matches(source, variant, RegexOptions.Multiline);
                foreach (Match m in matches) {
                    Match m2 = Regex.Match(m.Groups[1].Captures[0].Value, @"^\s*new\s+(.*?)$");
                    Match m3 = null;
                    if (m2.Success) {
                        m3 = Regex.Match(m2.Groups[1].Captures[0].Value, @"[^A-Z]([A-Z]\w+Param)\s*\(", RegexOptions.Multiline);
                    } else {
                        // declared elsewhere, hate.
                        //Debug.WriteLine("wtf is that " + m.Groups[1].Captures[0].Value);
                        MatchCollection mc = Regex.Matches(tags, @"^" + m.Groups[1].Captures[0].Value + @"\t\S+\t/(.*?)/;""", RegexOptions.Multiline);
                        foreach (Match m4 in mc) {
                            m3 = Regex.Match(m4.Groups[1].Captures[0].Value, @"[^A-Z]([A-Z]\w+Param)\s*\*?\s*" + m.Groups[1].Captures[0].Value);
                            if (m3.Success) {
                                break;
                            }
                        }
                    }
                    if (m3 != null && m3.Success) {
                        string ParamType = m3.Groups[1].Captures[0].Value;
                        mod.AddParameter(name, ParamType, exported, descr);
                    } else {
                        Debug.WriteLine("unable to understand parameter " + name);
                    }
                }
            }
        }

        private static string ReadAllSource(string header) {
            string source;
            StreamReader sr = new StreamReader(header);
            source = sr.ReadToEnd();
            sr.Close();
            string sf = FindSourceFile(header);
            if (File.Exists(sf)) {
                sr = new StreamReader(sf);
                source += sr.ReadToEnd();
                sr.Close();
                return source;
            }
            return source;
        }

        private static string FindSourceFile(string header) {
            string[] Hexts = { ".h", ".hpp", ".cuh" };
            string[] Cexts = { ".cpp", ".cu" };
            string hext = "";

            foreach (string ext in Hexts) {
                if (header.EndsWith(ext)) {
                    hext = ext;
                    break;
                }
            }

            foreach (string ext in Cexts) {
                string name = header.Replace(hext, ext);
                if (File.Exists(name)) {
                    return name;
                }
            }
            return "";
        }

        //public int Filter(string[] baseClasses) {
        //    for (int x = this.knownModules.Count - 1; x > -1; x--) {
        //        Module mod = this.knownModules[x];
        //        bool found = false;
        //        foreach (string bc in baseClasses) {
        //            if (mod.IsA(bc)) {
        //                found = true;
        //                break;
        //            }
        //        }
        //        if (!found) {
        //            this.knownModules.RemoveAt(x);
        //        }
        //    }
        //    return this.knownModules.Count;
        //}

        private Dictionary<string, string> allSource = new Dictionary<string, string>();
        private Dictionary<string, string> allTags = new Dictionary<string, string>();
    }

}
