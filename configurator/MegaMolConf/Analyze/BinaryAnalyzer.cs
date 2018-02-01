using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using megamol.config.armesau;

namespace MegaMolConf.Analyze {

    class BinaryAnalyzer {
        List<Data.PluginFile> plugins = new List<Data.PluginFile>();

        public List<Data.PluginFile> Analyze(string path) {
            
            string bitness = IntPtr.Size == 4 ? "32" : "64";
            List<MegaMolInterface> mmis = new List<MegaMolInterface>();
            mmis.Add(new MegaMolCoreInterface());
            List<String> dlls = FileFinder.FindFiles(path, "*.mmplg");
            foreach (string dll in dlls) {
                if (dll.Contains("win" + bitness) && 
// very large cinema indeed
#if !DEBUG
                    !
#endif
                    dll.Contains(bitness + "d")) {
                    mmis.Add(new MegaMolPluginInterface(dll));
                }
            }

            foreach (MegaMolInterface mmi in mmis) {

                Data.PluginFile pf = new Data.PluginFile();
                pf.Filename = mmi.Filename;
                pf.IsCore = mmi is MegaMolCoreInterface;
                List<Data.Call> calls = new List<Data.Call>();
                Debug.WriteLine(mmi.Filename + " has " + mmi.ModuleCount + " modules and " + mmi.CallCount + " calls.");
                for (int x = 0; x < mmi.CallCount; x++) {
                    Data.Call c = new Data.Call();
                    c.Name = mmi.CallName(x);
                    c.Description = mmi.CallDescription(x);
                    List<string> functions = new List<string>();
                    for (int y = 0; y < mmi.FunctionCount(x); y++) {
                        string s = mmi.FunctionName(x, y);
                        functions.Add(s);
                    }
                    c.FunctionName = functions.ToArray();
                    calls.Add(c);
                }
                pf.Calls = calls.ToArray();

                plugins.Add(pf);
            }

            foreach(MegaMolInterface mmi in mmis) {
                Data.PluginFile pf = FindPluginByFileName(mmi.Filename);
                if (pf == null) {
                    continue;
                }
                List<Data.Module> modules = new List<Data.Module>();
                for (int x = 0; x < mmi.ModuleCount; x++) {
                    Data.Module m = new Data.Module();
                    m.Name = mmi.ModuleName(x);
                    m.Description = mmi.ModuleDescription(x);
                    ModuleDescriptionObj mdo = mmi.ModuleDescriptionObj(x);

                    List<Data.ParamSlot> parms = new List<Data.ParamSlot>();
                    for (int y = 0; y < mdo.ParamSlotCount; y++) {
                        Data.ParamSlot p = new Data.ParamSlot();
                        p.Name = mdo.ParameterSlot(y).Name;
                        p.Description = mdo.ParameterSlot(y).Description;
                        Byte[] typeInfo = mdo.ParameterSlot(y).TypeInfo;
                        p.Type = this.TypeFromTypeInfo(typeInfo);
                        if (p.Type is Data.ParamTypeValueBase) {
                            ((Data.ParamTypeValueBase)p.Type).ParseDefaultValue(mdo.ParameterSlot(y).DefaultValue);
                        }
                        parms.Add(p);
                    }
                    if (parms.Count > 0) {
                        m.ParamSlots = parms.ToArray();
                    } else {
                        m.ParamSlots = null;
                    }

                    List<Data.CalleeSlot> calleeSlots = new List<Data.CalleeSlot>();
                    for (int y = 0; y < mdo.CalleeSlotCount; y++) {
                        Dictionary<string, List<string>> callTypes = new Dictionary<string, List<string>>();
                        CalleeSlotDescription csd = mdo.CalleeSlot(y);
                        for (int z = 0; z < csd.CallbackCount; z++) {
                            if (!callTypes.ContainsKey(csd.CallType(z))) {
                                callTypes.Add(csd.CallType(z), new List<string>());
                            }
                            callTypes[csd.CallType(z)].Add(csd.FunctionName(z));
                        }
                        // now find out if these fit any calls
                        Data.CalleeSlot cs = new Data.CalleeSlot();
                        // TODO this is bullshit! why?
                        cs.Name = csd.Name;
                        cs.Description = csd.Description;
                        List<string> compCalls = new List<string>();
                        foreach (string calltype in callTypes.Keys) {
                            Data.Call c = FindCallByName(calltype);
                            if (c != null) {
                                bool ok = true;
                                foreach (string fun in c.FunctionName) {
                                    ok &= callTypes[calltype].Contains(fun);
                                }
                                if (ok) {
                                    compCalls.Add(calltype);
                                }
                            }
                        }
                        if (compCalls.Count > 0) {
                            cs.CompatibleCalls = compCalls.ToArray();
                        } else {
                            cs.CompatibleCalls = null;
                        }
                        calleeSlots.Add(cs);
                    }
                    if (calleeSlots.Count > 0) {
                        m.CalleeSlots = calleeSlots.ToArray();
                    } else {
                        m.CalleeSlots = null;
                    }
                   
                    List<Data.CallerSlot> callerSlots = new List<Data.CallerSlot>();
                    for (int y = 0; y < mdo.CallerSlotCount; y++) {
                        Data.CallerSlot callerSlot = new Data.CallerSlot();
                        CallerSlotDescription csd = mdo.CallerSlot(y);
                        callerSlot.Name = csd.Name;
                        callerSlot.Description = csd.Description;
                        List<string> compCalls = new List<string>();

                        for (int z = 0; z < csd.CompatibleCallCount; z++) {
                            string callName = csd.CompatibleCall(z);
                            if (FindCallByName(callName) != null) {
                                compCalls.Add(callName);
                            }
                        }
                        callerSlot.CompatibleCalls = compCalls.ToArray();
                        callerSlots.Add(callerSlot);
                    }
                    if (callerSlots.Count > 0) {
                        m.CallerSlots = callerSlots.ToArray();
                    } else {
                        m.CallerSlots = null;
                    }

                    modules.Add(m);
                }
                pf.Modules = modules.ToArray();

            }
            return plugins;
        }

        private Data.PluginFile FindPluginByFileName(string p) {
            foreach (Data.PluginFile pf in plugins) {
                if (pf.Filename.Equals(p)) {
                    return pf;
                }
            }
            return null;
        }

        private Data.Call FindCallByName(string name) {
            foreach (Data.PluginFile pf in plugins) {
                foreach (Data.Call c in pf.Calls) {
                    if (c.Name.Equals(name)) {
                        return c;
                    }
                }
            }
            return null;
        }

        /// <summary>
        /// Creates a type object based on a type info description
        /// </summary>
        /// <param name="typeInfo">The type info description</param>
        /// <returns>The type object</returns>
        private Data.ParamTypeBase TypeFromTypeInfo(byte[] typeInfo) {
            string typeName = string.Empty;
            if (typeInfo.Length >= 6) {
                typeName = Encoding.ASCII.GetString(typeInfo, 0, 6);
            }

            Data.ParamTypeBase ptb = null;

            if (typeName == "MMBUTN") {
                ptb = new Data.ParamType.Button();
            } else if (typeName == "MMBOOL") {
                ptb = new Data.ParamType.Bool();
            } else if (typeName == "MMENUM") {
                ptb = new Data.ParamType.Enum();
                int pos = 6;
                int cnt = (int)BitConverter.ToUInt32(typeInfo, pos);
                ((Data.ParamType.Enum)ptb).ValueNames = new string[cnt];
                ((Data.ParamType.Enum)ptb).Values = new int[cnt];
                pos += 4;
                for (int i = 0; i < cnt; i++) {
                    ((Data.ParamType.Enum)ptb).Values[i] = (int)BitConverter.ToUInt32(typeInfo, pos);
                    pos += 4;
                    int ePos = pos;
                    while ((ePos < typeInfo.Length) && (typeInfo[ePos] != 0)) ePos++;
                    ((Data.ParamType.Enum)ptb).ValueNames[i] = Encoding.ASCII.GetString(typeInfo, pos, ePos - pos);
                    pos = ePos + 1;
                }
            } else if (typeName == "MMFLOT") {
                ptb = new Data.ParamType.Float();
                ((Data.ParamType.Float)ptb).MinValue = BitConverter.ToSingle(typeInfo, 6);
                ((Data.ParamType.Float)ptb).MaxValue = BitConverter.ToSingle(typeInfo, 10);
            } else if (typeName == "MMINTR") {
                ptb = new Data.ParamType.Int();
                ((Data.ParamType.Int)ptb).MinValue = BitConverter.ToInt32(typeInfo, 6);
                ((Data.ParamType.Int)ptb).MaxValue = BitConverter.ToInt32(typeInfo, 10);
            } else {
                ptb = new Data.ParamType.String();
            }

            if (ptb != null) {
                ptb.TypeName = typeName;
            }
            return ptb;
        }

    }
}
