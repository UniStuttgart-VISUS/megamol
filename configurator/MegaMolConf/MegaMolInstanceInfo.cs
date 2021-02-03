using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;
using MegaMolConf.Analyze;
using MegaMolConf.Communication;
using ZeroMQ;

namespace MegaMolConf {
    class MegaMolInstanceInfo {
        public System.Diagnostics.Process Process { get; set; }
        public Communication.Connection Connection { get; set; }
        public TabPage TabPage { get; set; }
        public int Port { get; set; }
        public string Host { get; set; }
        public System.Threading.Thread Thread { get; set; }
        public Form1 ParentForm { get; set; }
        public bool ReadGraphFromInstance { get; set; }

        private bool stopQueued;
        //private string[] knownParams;

        private Object myLock = new Object();
        private string connectionString = "";

        private List<GraphicalModule> moduleCreations = new List<GraphicalModule>();
        private List<GraphicalModule> moduleDeletions = new List<GraphicalModule>();
        private List<GraphicalConnection> connectionCreations = new List<GraphicalConnection>();
        private List<GraphicalConnection> connectionDeletions = new List<GraphicalConnection>();

        public MegaMolInstanceInfo() {
            Host = "localhost";
            ReadGraphFromInstance = false;
        }

        public void QueueModuleCreation(GraphicalModule gm) {
            lock (moduleCreations) {
                moduleCreations.Add(gm);
            }
        }

        public void QueueModuleDeletion(GraphicalModule gm) {
            lock (moduleDeletions) {
                moduleDeletions.Add(gm);
            }
        }

        public void QueueConnectionCreation(GraphicalConnection gc) {
            lock (connectionCreations) {
                connectionCreations.Add(gc);
            }
        }

        public void QueueConnectionDeletion(GraphicalConnection gc) {
            lock (connectionDeletions) {
                connectionDeletions.Add(gc);
            }
        }

        public enum MegaMolProcessState {
            MMPS_NONE = 1,
            MMPS_CONNECTION_GOOD = 2,
            MMPS_CONNECTION_BROKEN = 3
        }

        private static bool IsRunningOnMono() {
            return Type.GetType("Mono.Runtime") != null;
        }

        private void SetProcessState(MegaMolProcessState state) {
            ParentForm.SetTabPageIcon(TabPage, (int)state);
        }

        private static byte[] StringToByteArray(string hex) {
            byte[] theBytes;

            if (hex.Length % 2 == 0) {
                theBytes = new byte[hex.Length / 2];
                for (int x = 0; x < hex.Length; x += 2) {
                    theBytes[x / 2] = Convert.ToByte(hex.Substring(x, 2), 16);
                }
            } else {
                theBytes = null;
            }
            return theBytes;
        }

        public void StopObserving() {
            stopQueued = true;
        }

        public void Observe() {
            Connection = null;
            //Process.Exited += new EventHandler(delegate (Object o, EventArgs a) {
            //    SetProcessState(MegaMolProcessState.MMPS_NONE);
            //    StopObserving();
            //    ParentForm.SetTabPageTag(TabPage, null);
            //    ParentForm.listBoxLog.Log(Util.Level.Info, string.Format("Tab '{0}' disconnected", TabPage.Text));
            //});
            connectionString = "tcp://" + Host + ":" + Port;

            TryConnecting(connectionString);
            string err = "", ans = "";
            string res;

            if (!stopQueued) {
                // the following code does not work under Mono --
                // detached process gets a different ID
                if (!IsRunningOnMono()) {

                    if (Connection.Valid) {

                        Request("return mmGetProcessID()");
                        err = GetAnswer(ref ans);
                        if (String.IsNullOrWhiteSpace(err)) {
                            uint id = 0;
                            try {
                                id = uint.Parse((string)ans);
                            } catch {
                                id = 0;
                            }

                            foreach (var proc in System.Diagnostics.Process.GetProcesses()) {
                                if (proc.Id == id) {
                                    Process = proc;
                                    break;
                                }
                            }
                            if (Process != null) {
                                try {
                                    Process.EnableRaisingEvents = true;
                                    Process.Exited += new EventHandler(delegate(Object o, EventArgs a) {
                                        SetProcessState(MegaMolProcessState.MMPS_NONE);
                                        StopObserving();
                                        ParentForm.SetTabPageTag(TabPage, null);
                                        ParentForm.listBoxLog.Log(Util.Level.Info,
                                            $"Tab '{TabPage.Text}' disconnected");
                                        ParentForm.SetTabPageIcon(TabPage, 1);
                                        ParentForm.listBoxLog.Log(Util.Level.Info, $"Tab '{TabPage.Text}' exited");
                                        Process = null;
                                    });
                                } catch {
                                    ParentForm.listBoxLog.Log(Util.Level.Warning, $"could not install exit handler for Tab '{TabPage.Text}'");
                                    // we can only do this if we started the process ourselves
                                }
                                SetProcessState(MegaMolProcessState.MMPS_CONNECTION_GOOD);
                            } else {
                                SetProcessState(MegaMolProcessState.MMPS_CONNECTION_BROKEN); // wrong instance
                                Connection = null;
                            }
                            //if (id == Process.Id) {
                            //    SetProcessState(MegaMolProcessState.MMPS_CONNECTION_GOOD);
                            //} else {
                            //    SetProcessState(MegaMolProcessState.MMPS_CONNECTION_BROKEN); // wrong instance
                            //    Connection = null;
                            //}
                        } else {
                            SetProcessState(MegaMolProcessState.MMPS_CONNECTION_BROKEN); // broken
                            Connection = null;
                        }
                    } else {
                        // could not connect
                        //mmii.Connection = null;
                        SetProcessState(MegaMolProcessState.MMPS_CONNECTION_BROKEN); //broken
                    }
                } else {
                    if (!Connection.Valid) {
                        // could not connect
                        //mmii.Connection = null;
                        SetProcessState(MegaMolProcessState.MMPS_CONNECTION_BROKEN); //broken
                    }
                }
            }

            System.Threading.Thread.Sleep(100);

            if (Connection != null && Connection.Valid) {
                List<string> foundInstances = new List<string>();
                do {
                    Request("return mmListInstantiations()");
                    ans = null;
                    do {
                        res = GetAnswer(ref ans);
                    } while (ans == null);
                    if (String.IsNullOrWhiteSpace(res)) {
                        var insts = ((string) ans).Split(new char[] {'\n'}, StringSplitOptions.None);
                        foreach (var i in insts) {
                            if (!String.IsNullOrEmpty(i)) {
                                foundInstances.Add(i);
                            }
                        }

                        if (foundInstances.Count == 0) {
                            ParentForm.SetTabInstantiation(TabPage, "");
                        } else if (foundInstances.Count == 1 && !ReadGraphFromInstance) {
                            ParentForm.SetTabInstantiation(TabPage, foundInstances[0]);
                        } 
                        else {
                            ParentForm.ChooseInstantiation(TabPage, foundInstances.ToArray());
                        }
                    }

                    System.Threading.Thread.Sleep(1000);
                } while (foundInstances.Count == 0);

                if (ReadGraphFromInstance) {
                    if (string.IsNullOrEmpty(ParentForm.TabInstantiation(TabPage))) {
                        Request("return mmListModules()");
                        res = GetAnswer(ref ans);
                    } else {
                        Request($"return mmListModules(\"{ParentForm.TabInstantiation(TabPage)}\")");
                        res = GetAnswer(ref ans);
                    }
                    if (string.IsNullOrWhiteSpace(res)) {
                        var modules = ((string)ans).Split(new char[] { '\n' }, StringSplitOptions.None);
                        foreach (var mod in modules) {
                            var stuff = mod.Split(new char[] {';'}, StringSplitOptions.None);
                            if (stuff.Length == 2) {
                                ParentForm.AddModule(TabPage, stuff[0], stuff[1]);
                            }
                        }
                    }

                    if (string.IsNullOrEmpty(ParentForm.TabInstantiation(TabPage))) {
                        Request("return mmListCalls()");
                        res = GetAnswer(ref ans);
                    } else {
                        Request($"return mmListCalls(\"{ParentForm.TabInstantiation(TabPage)}\")");
                        res = GetAnswer(ref ans);
                    }

                    if (String.IsNullOrWhiteSpace(res)) {
                        var calls = ((string)ans).Split(new char[] { '\n' }, StringSplitOptions.None);
                        foreach (var c in calls) {
                            var stuff = c.Split(new char[] { ';' }, StringSplitOptions.None);
                            if (stuff.Length == 3) {
                                var fromTo = stuff[1].Split(new char[] { ',' }, StringSplitOptions.None);
                                var srcDest = stuff[2].Split(new char[] { ',' }, StringSplitOptions.None);
                                if (fromTo.Length == 2 && srcDest.Length == 2) {
                                    ParentForm.AddConnection(TabPage, stuff[0], fromTo[0], srcDest[0], fromTo[1], srcDest[1]);
                                }
                            }
                        }
                    }
                    ParentForm.ForceDirectedLayout(ParentForm.SelectedTab);
                    ParentForm.RefreshCurrent();
                    ParentForm.listBoxLog.Log(Util.Level.Info, "Done reading attached project info");
                }
            } else {
                ParentForm.FreePort(Port);
                return;
            }
            while (!stopQueued) {
                System.Threading.Thread.Sleep(1000);
                // check current tab (is the correct instance controlled)
                if (stopQueued) {
                    break;
                }
                TabPage tp = ParentForm.SelectedTab;
                if (tp == TabPage) {

                    lock (moduleCreations) {
                        foreach (GraphicalModule gmc in moduleCreations) {
                            string command =
                                $@"mmCreateModule(""{gmc.Module.Name}"", ""{ParentForm.TabInstantiation(TabPage)}::{gmc.Name}"")";
                            Request("return " + command);
                            res = GetAnswer(ref ans);
                            if (String.IsNullOrWhiteSpace(res) && !stopQueued) {
                                // huh.
                            } else {
                                ParentForm.listBoxLog.Log(Util.Level.Error, $@"Error on {command}: {res}");
                            }
                        }
                        moduleCreations.Clear();
                    }

                    lock (moduleDeletions) {
                        foreach (GraphicalModule gmc in moduleDeletions) {
                            string command = $@"mmDeleteModule(""{ParentForm.TabInstantiation(TabPage)}::{gmc.Name}"")";
                            Request("return " + command);
                            res = GetAnswer(ref ans);
                            if (String.IsNullOrWhiteSpace(res) && !stopQueued) {
                                // huh.
                            } else {
                                ParentForm.listBoxLog.Log(Util.Level.Error, $@"Error on {command}: {res}");
                            }
                        }
                        moduleDeletions.Clear();
                    }

                    // TODO: connections. beware that connections will probably be cleaned if the modules go away, so failure is okayish? :/

                    lock (connectionDeletions) {
                        foreach (GraphicalConnection gcc in connectionDeletions) {
                            string command =
                                $@"mmDeleteCall(""{ParentForm.TabInstantiation(TabPage)}::{gcc.src.Name}::{
                                    gcc.srcSlot.Name
                                }"", ""{ParentForm.TabInstantiation(TabPage)}::{gcc.dest.Name}::{gcc.destSlot.Name}"")";
                            Request($"return {command}");
                            res = GetAnswer(ref ans);
                            if (String.IsNullOrWhiteSpace(res) && !stopQueued) {
                                // huh.
                            } else {
                                ParentForm.listBoxLog.Log(Util.Level.Error, $@"Error on {command}: {res}");
                            }
                        }
                        connectionDeletions.Clear();
                    }

                    lock (connectionCreations) {
                        foreach (GraphicalConnection gcc in connectionCreations) {
                            string command =
                                $@"mmCreateCall(""{gcc.Call.Name}"",""{ParentForm.TabInstantiation(TabPage)}::{gcc.src.Name}::{
                                    gcc.srcSlot.Name
                                }"", ""{ParentForm.TabInstantiation(TabPage)}::{gcc.dest.Name}::{gcc.destSlot.Name}"")";
                            Request("return " + command);
                            res = GetAnswer(ref ans);
                            if (String.IsNullOrWhiteSpace(res) && !stopQueued) {
                                // huh.
                            } else {
                                ParentForm.listBoxLog.Log(Util.Level.Error, $@"Error on {command}: {res}");
                            }
                        }
                        connectionCreations.Clear();
                    }

                    // what the hell?
                    GraphicalModule gm = Form1.selectedModule;
                    if (gm != null) {
#if true
                        if (stopQueued)
                            break;

                        res = UpdateModuleParams(gm);
                    }
#else
                    string prefix = "inst::" + gm.Name + "::";
                    var keys = gm.ParameterValues.Keys.ToList();
                    foreach (Data.ParamSlot p in keys) {
                        if (stopQueued) break;
                        res = this.Request(new MegaMol.SimpleParamRemote.MMSPR1.GETVALUE() { ParameterName = prefix + p.Name }, ref ans);
                        if (String.IsNullOrWhiteSpace(res) && !stopQueued) {
                            if (!p.Type.ValuesEqual(gm.ParameterValues[p], (string)ans)) {
                                gm.ParameterValues[p] = (string)ans;
                                this.ParentForm.ParamChangeDetected();
                            }
                        }
                    }
#endif
                }
            }
            if (Connection != null) {
                Connection.Close();
            }
            ParentForm.FreePort(Port);
        }

        public void UpdateModules(IEnumerable<GraphicalModule> modules) {
            foreach (GraphicalModule gm in modules) {
                UpdateModuleParams(gm);
            }
        }

        private string UpdateModuleParams(GraphicalModule gm) {
            string ans = null;
            string command = $"mmGetModuleParams(\"{ParentForm.TabInstantiation(TabPage)}::{gm.Name}\")";
            string res;
            Request("return " + command);
            res = GetAnswer(ref ans);
            if (String.IsNullOrWhiteSpace(res) && !stopQueued) {
                string[] stuff = (ans).Split(new char[] { '\u0001' }, StringSplitOptions.None);
                int len = stuff.Count();
                // don't forget the trailing element since we conclude with one last separator!
                if (len > 1 && (len - 1) % 4 == 1) {
                    string module = stuff[0];
                    for (int x = 1; x < len - 1; x += 4) {
                        string slotname = stuff[x];
                        // +1 description
                        // +2 typedescription
                        //MegaMol.SimpleParamRemote.ParameterTypeDescription ptd = new MegaMol.SimpleParamRemote.ParameterTypeDescription();
                        //ptd.SetFromHexStringDescription(stuff[x + 2]);
                        string slotvalue = stuff[x + 3];
                        var keys = gm.ParameterValues.Keys.ToList();
                        foreach (Data.ParamSlot p in keys) {
                            if (p.Name.Equals(slotname)) {
                                if (!p.Type.ValuesEqual(gm.ParameterValues[p], slotvalue)) {
                                    gm.ParameterValues[p] = slotvalue;
                                    ParentForm.ParamChangeDetected();
                                }
                                //if (ptd.Type == MegaMol.SimpleParamRemote.ParameterType.FlexEnumParam) {
                                if (p.Type.TypeName == "MMFENU") {
                                    //int numVals = ptd.ExtraSettings.Keys.Count() / 2;
                                    p.Type = p.TypeFromTypeInfo(StringToByteArray(stuff[x + 2]));
                                }
                            }
                        }
                    }
                } else {
                    ParentForm.listBoxLog.Log(Util.Level.Error, $"invalid response to {command}");
                }
            } else {
                ParentForm.listBoxLog.Log(Util.Level.Error, $"Error in {command}: {res}");
            }

            return res;
        }

        private void TryConnecting(string conn) {
            while (!stopQueued && Connection == null) {
                try {
                    Connection = Communication.Connection.Connect(conn);
                } catch {
                    // nothing
                    System.Threading.Thread.Sleep(500);
                }
                ParentForm.listBoxLog.Log(Util.Level.Info, string.Format("Tab '{0}' opened connection to {1}", TabPage.Text, conn));
            }
        }

        internal void SendUpdate(GraphicalModule gm, string p, string v) {
            if (gm != null) {
                string prefix = $"{ParentForm.TabInstantiation(TabPage)}::{gm.Name}::";
                string ans = null;
                string val = v.Replace("\\", "\\\\");
                Request($"return mmSetParamValue(\"{prefix}{p}\",\"{val}\")");
                string res = GetAnswer(ref ans);
            }
        }

        bool TryGetAnswer(ref string answer, ref string err) {
            if (Connection == null)
            {
                return false;
            }
            Communication.Response r = new Response();
            bool good = Connection.TryReceive(ref r, ref err);
            if (r != null) {
                answer = r.Answer;
            } else {
                answer = null;
            }
            return good;
        }

        string GetAnswer(ref string answer) {
            string err = "";
            bool good = false;
            int numTries = 500;
            while (!(good = TryGetAnswer(ref answer, ref err)) && numTries > 0) {
                numTries--;
                System.Threading.Thread.Sleep(10);
            }

            if (!good)
                err = "timeout";
            return err;
        }

        internal string Request(string req) {
            string ret = "";
            lock (myLock) {
                if (Connection != null && Connection.Valid && !stopQueued) {
                    try {
                        Communication.GenericRequest request = new Communication.GenericRequest { Command = req };
                        Connection.Send(request);
                        //Communication.Response res = Connection.Send(request);
                        //if (!string.IsNullOrWhiteSpace(res.Error)) {
                        //    ret = res.Error;
                        //    ParentForm.listBoxLog.Log(Util.Level.Error, string.Format("MegaMol did not accept {0}: {1}", req.ToString(), res.Error));
                        //    SetProcessState(MegaMolProcessState.MMPS_CONNECTION_BROKEN); // broken
                        //    //if (!res.Error.StartsWith("NOTFOUND")) {
                        //    //    MessageBox.Show(string.Format("MegaMol did not accept {0}", req.ToString()), "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        //    //    SetProcessState(MegaMolProcessState.MMPS_CONNECTION_BROKEN); // broken
                        //    //} else {
                        //    //    SetProcessState(MegaMolProcessState.MMPS_CONNECTION_BROKEN); // broken
                        //    //}
                        //} else {
                        //    // problem: slow megamol might re-order answers. so if we continue listening no matter what,
                        //    // we will get answers from other requests which blows us up
                        //    answer = res.Answer;
                        //    SetProcessState(MegaMolProcessState.MMPS_CONNECTION_GOOD); // good
                        //}
                    } catch (Exception ex) {
                        if (!stopQueued) {
                            // todo set this to false when message ordering really works (see above)
#if false
                            MessageBox.Show("Failed to send: " + ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                            ret = ex.Message;
                            this.StopObserving();
                            this.Connection.Close();
                            this.Connection = null;
#endif
                            ParentForm.listBoxLog.Log(Util.Level.Error, string.Format("Exception with request {0} in flight: {1}", req.ToString(), ex.Message));
                            SetProcessState(MegaMolProcessState.MMPS_CONNECTION_BROKEN); // broken
                            ret = ex.Message;
                            Connection = null;
                            TryConnecting(connectionString);
                        }
                    }
                }
            }
            return ret;
        }
    }
}
