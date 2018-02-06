using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MegaMolConf {
    class MegaMolInstanceInfo {
        public System.Diagnostics.Process Process { get; set; }
        public Communication.Connection Connection { get; set; }
        public System.Windows.Forms.TabPage TabPage { get; set; }
        public int Port { get; set; }
        public System.Threading.Thread Thread { get; set; }
        public Form1 ParentForm { get; set; }

        private bool stopQueued = false;
        //private string[] knownParams;

        private Object myLock = new Object();
        private string connectionString = "";

        private List<GraphicalModule> moduleCreations = new List<GraphicalModule>();
        private List<GraphicalModule> moduleDeletions = new List<GraphicalModule>();
        private List<GraphicalConnection> connectionCreations = new List<GraphicalConnection>();
        private List<GraphicalConnection> connectionDeletions = new List<GraphicalConnection>();

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
            this.ParentForm.SetTabPageIcon(this.TabPage, (int)state);
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
            this.Connection = null;
            this.Process.Exited += new EventHandler(delegate (Object o, EventArgs a) {
                SetProcessState(MegaMolProcessState.MMPS_NONE);
                this.StopObserving();
                this.ParentForm.SetTabPageTag(this.TabPage, null);
                ParentForm.listBoxLog.Log(Util.Level.Info, string.Format("Tab '{0}' disconnected", this.TabPage.Text));
            });
            this.connectionString = "tcp://localhost:" + this.Port;

            TryConnecting(this.connectionString);
            object ans = null;
            string res;

            if (!stopQueued) {
                // the following code does not work under Mono --
                // detached process gets a different ID
                if (!IsRunningOnMono()) {

                    if (this.Connection.Valid) {
                        string ret = this.Request("return mmGetProcessID()", ref ans);
                        if (String.IsNullOrWhiteSpace(ret)) {
                            uint id = 0;
                            try {
                                id = uint.Parse((string)ans);
                            } catch {
                                id = 0;
                            }
                            if (id == this.Process.Id) {
                                SetProcessState(MegaMolProcessState.MMPS_CONNECTION_GOOD);
                            } else {
                                SetProcessState(MegaMolProcessState.MMPS_CONNECTION_BROKEN); // wrong instance
                                this.Connection = null;
                            }
                        } else {
                            SetProcessState(MegaMolProcessState.MMPS_CONNECTION_BROKEN); // broken
                            this.Connection = null;
                        }
                    } else {
                        // could not connect
                        //mmii.Connection = null;
                        SetProcessState(MegaMolProcessState.MMPS_CONNECTION_BROKEN); //broken
                    }
                } else {
                    if (!this.Connection.Valid) {
                        // could not connect
                        //mmii.Connection = null;
                        SetProcessState(MegaMolProcessState.MMPS_CONNECTION_BROKEN); //broken
                    }
                }
            }

            if (Connection != null && Connection.Valid) {
                //res = this.Request(new MegaMol.SimpleParamRemote.MMSPR1.QUERYPARAMS(), ref ans);
                //if (String.IsNullOrWhiteSpace(res)) {
                //    knownParams = (string[])ans;
                //    Array.Sort(knownParams);
                //}
            } else {
                ParentForm.FreePort(Port);
                return;
            }
            while (!stopQueued) {
                System.Threading.Thread.Sleep(1000);
                GraphicalModule gm = Form1.selectedModule;
                // check current tab (is the correct instance controlled)
                TabPage tp = ParentForm.selectedTab;
                if (tp == this.TabPage) {

                    lock (moduleCreations) {
                        foreach (GraphicalModule gmc in moduleCreations) {
                            string command = @"mmCreateModule(""" + gmc.Module.Name + @""", ""::inst::" + gmc.Name + @""")";
                            res = this.Request("return " + command, ref ans);
                            if (String.IsNullOrWhiteSpace(res) && !stopQueued) {
                                // huh.
                            } else {
                                ParentForm.listBoxLog.Log(Util.Level.Error, @"Error on " + command + ": " + res);
                            }
                        }
                        moduleCreations.Clear();
                    }

                    lock (moduleDeletions) {
                        foreach (GraphicalModule gmc in moduleDeletions) {
                            string command = @"mmDeleteModule(""::inst::" + gmc.Name + @""")";
                            res = this.Request("return " + command, ref ans);
                            if (String.IsNullOrWhiteSpace(res) && !stopQueued) {
                                // huh.
                            } else {
                                ParentForm.listBoxLog.Log(Util.Level.Error, @"Error on " + command + ": " + res);
                            }
                        }
                        moduleDeletions.Clear();
                    }

                    // TODO: connections. beware that connections will probably be cleaned if the modules go away, so failure is okayish? :/

                    lock (connectionCreations) {
                        foreach (GraphicalConnection gcc in connectionCreations) {
                            string command = @"mmCreateCall(""" + gcc.Call.Name + @""",""::inst::" + gcc.src.Name + "::"
                                + gcc.srcSlot.Name + @""", ""::inst::" + gcc.dest.Name + "::" + gcc.destSlot.Name + @""")";
                            res = this.Request("return " + command, ref ans);
                            if (String.IsNullOrWhiteSpace(res) && !stopQueued) {
                                // huh.
                            } else {
                                ParentForm.listBoxLog.Log(Util.Level.Error, @"Error on " + command + ": " + res);
                            }
                        }
                        connectionCreations.Clear();
                    }

                    lock (connectionDeletions) {
                        foreach (GraphicalConnection gcc in connectionDeletions) {
                            string command = @"mmDeleteCall(""::inst::" + gcc.src.Name + "::" 
                                + gcc.srcSlot.Name + @""", ""::inst::" + gcc.dest.Name + "::" + gcc.destSlot.Name + @""")";
                            res = this.Request("return " + command, ref ans);
                            if (String.IsNullOrWhiteSpace(res) && !stopQueued) {
                                // huh.
                            } else {
                                ParentForm.listBoxLog.Log(Util.Level.Error, @"Error on " + command + ": " + res);
                            }
                        }
                        connectionDeletions.Clear();
                    }

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
            object ans = null;
            string res = this.Request("return mmGetModuleParams(\"" + "inst::" + gm.Name + "\")", ref ans);
            if (String.IsNullOrWhiteSpace(res) && !stopQueued) {
                string[] stuff = ((string)ans).Split(new char[] { '\u0001' }, StringSplitOptions.None);
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
                                    this.ParentForm.ParamChangeDetected();
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
                    ParentForm.listBoxLog.Log(Util.Level.Error, "invalid response to mmcGetModuleParams(\"inst::" + gm.Name + "\")");
                }
            } else {
                ParentForm.listBoxLog.Log(Util.Level.Error, "Error in mmcGetModuleParams(\"inst::" + gm.Name + "\"): " + res);
            }

            return res;
        }

        private void TryConnecting(string conn) {
            while (!stopQueued && this.Connection == null) {
                try {
                    this.Connection = Communication.Connection.Connect(conn);
                } catch {
                    // nothing
                    System.Threading.Thread.Sleep(500);
                }
                ParentForm.listBoxLog.Log(Util.Level.Info, string.Format("Tab '{0}' opened connection to {1}", this.TabPage.Text, conn));
            }
        }

        internal void SendUpdate(string p, string v) {
            GraphicalModule gm = Form1.selectedModule;
            if (gm != null) {
                string prefix = "inst::" + gm.Name + "::";
                object ans = null;
                string ret = Request("return mmSetParamValue(\""  + prefix + p + "\",\"" + v +"\")", ref ans);
            }
        }

        internal string Request(string req, ref object answer) {
            string ret = "";
            lock (myLock) {
                if (this.Connection != null && this.Connection.Valid && !this.stopQueued) {
                    try {
                        Communication.GenericRequest request = new Communication.GenericRequest { Command = req };
                        Communication.Response res = Connection.Send(request);
                        if (!string.IsNullOrWhiteSpace(res.Error)) {
                            ret = res.Error;
                            ParentForm.listBoxLog.Log(Util.Level.Error, string.Format("MegaMol did not accept {0}: {1}", req.ToString(), res.Error));
                            SetProcessState(MegaMolProcessState.MMPS_CONNECTION_BROKEN); // broken
                            //if (!res.Error.StartsWith("NOTFOUND")) {
                            //    MessageBox.Show(string.Format("MegaMol did not accept {0}", req.ToString()), "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                            //    SetProcessState(MegaMolProcessState.MMPS_CONNECTION_BROKEN); // broken
                            //} else {
                            //    SetProcessState(MegaMolProcessState.MMPS_CONNECTION_BROKEN); // broken
                            //}
                        } else {
                            // problem: slow megamol might re-order answers. so if we continue listening no matter what,
                            // we will get answers from other requests which blows us up
                            answer = res.Answer;
                            SetProcessState(MegaMolProcessState.MMPS_CONNECTION_GOOD); // good
                        }
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
                            this.Connection = null;
                            TryConnecting(this.connectionString);
                        }
                    }
                }
            }
            return ret;
        }
    }
}
