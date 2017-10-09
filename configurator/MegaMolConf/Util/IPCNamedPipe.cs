using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO.Pipes;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MegaMolConf.Util {

    /// <summary>
    /// Migrated from HiddenConsole™ (c) SGrottel 2015
    /// Back-Ported to DotNet v4
    /// </summary>
    internal class IPCNamedPipe {
        private NamedPipeServerStream pipe = null;
        private byte[] buffer;
        private int bytesReceived;
        private int bytesExpected;
        private int receiveMode;
        private string name;
        public bool ServerPipeOpen { get { return pipe != null; } }
        public delegate void StringReceivedDelegate(object sender, string line);
        public event StringReceivedDelegate StringReceived;
        public IPCNamedPipe(string name) {
            try {
                this.name = name;
                pipe = new NamedPipeServerStream(name, PipeDirection.In, 1, PipeTransmissionMode.Byte, PipeOptions.Asynchronous);
                buffer = new byte[1024];
                pipe.BeginWaitForConnection(asyncConnected, null);
                //pipe.WaitForConnectionAsync().ContinueWith(connected);
            } catch { }
        }
        public void Close() {
            pipe.Dispose();
        }
        public void SendStrings(string[] args) {
            using (NamedPipeClientStream npcs = new NamedPipeClientStream(".", name, PipeDirection.Out)) {
                npcs.Connect();
                foreach (string arg in args) {
                    byte[] buf = System.Text.Encoding.UTF8.GetBytes(arg);
                    byte[] buf2 = BitConverter.GetBytes((int)buf.Length);
                    npcs.Write(buf2, 0, 4);
                    npcs.Write(buf, 0, buf.Length);
                }
            }
        }
        private void asyncConnected(IAsyncResult ar) {
            pipe.EndWaitForConnection(ar);
            if (ar.IsCompleted) {
                bytesReceived = 0;
                bytesExpected = 4; // length of incoming string
                receiveMode = 0; // string length
                try {
                    if (buffer.Length < bytesExpected) buffer = new byte[bytesExpected];
                    //pipe.ReadAsync(buffer, bytesReceived, bytesExpected - bytesReceived).ContinueWith(received);
                    pipe.BeginRead(buffer, bytesReceived, bytesExpected - bytesReceived, asyncReceived, null);
                } catch { }
            }
        }
        //private void connected(Task t) {
        //    if (!t.IsFaulted) {
        //        bytesReceived = 0;
        //        bytesExpected = 4; // length of incoming string
        //        receiveMode = 0; // string length
        //        try {
        //            if (buffer.Length < bytesExpected) buffer = new byte[bytesExpected];
        //            pipe.ReadAsync(buffer, bytesReceived, bytesExpected - bytesReceived).ContinueWith(received);
        //        } catch { }
        //    }
        //}
        private void asyncReceived(IAsyncResult ar) {
            int bytesReceivedAsync = pipe.EndRead(ar);
            if (ar.IsCompleted) {
                bytesReceived += bytesReceivedAsync;
                if (bytesReceived > bytesExpected) {
                    // Does this happen? I hope not
                    Debug.WriteLine("Ack");
                } else if (bytesReceived == bytesExpected) {
                    // handle message
                    switch (receiveMode) {
                        case 0:
                            int strLen = BitConverter.ToInt32(buffer, 0);
                            if (strLen > 0) {
                                receiveMode = 1; // string data
                                bytesExpected = strLen;
                                bytesReceived = 0;
                            }
                            break;
                        case 1:
                            string s = System.Text.Encoding.UTF8.GetString(buffer, 0, bytesReceived);
                            Debug.WriteLine(">>>" + s);
                            StringReceivedDelegate eh = StringReceived;
                            if (eh != null) {
                                try {
                                    eh(this, s);
                                } catch { }
                            }
                            bytesReceived = 0;
                            bytesExpected = 4; // length of incoming string
                            receiveMode = 0; // string length
                            break;
                    }
                }
            }

            if (pipe.IsConnected) {
                // request more data
                try {
                    //pipe.ReadAsync(buffer, bytesReceived, bytesExpected - bytesReceived).ContinueWith(received);
                    pipe.BeginRead(buffer, bytesReceived, bytesExpected - bytesReceived, asyncReceived, null);
                } catch { }

            } else {
                // pipe broken
                try {
                    pipe.Disconnect();
                    pipe.BeginWaitForConnection(asyncConnected, null);
                    //pipe.WaitForConnectionAsync().ContinueWith(connected);
                } catch { }
            }
        }
        //private void received(Task<int> dataSize) {
        //    if (!dataSize.IsFaulted) {
        //        bytesReceived += dataSize.Result;
        //        if (bytesReceived > bytesExpected) {
        //            // Does this happen? I hope not
        //            Debug.WriteLine("Ack");
        //        } else if (bytesReceived == bytesExpected) {
        //            // handle message
        //            switch (receiveMode) {
        //                case 0:
        //                    int strLen = BitConverter.ToInt32(buffer, 0);
        //                    if (strLen > 0) {
        //                        receiveMode = 1; // string data
        //                        bytesExpected = strLen;
        //                        bytesReceived = 0;
        //                    }
        //                    break;
        //                case 1:
        //                    string s = System.Text.Encoding.UTF8.GetString(buffer, 0, bytesReceived);
        //                    Debug.WriteLine(">>>" + s);
        //                    EventHandler<string> eh = StringReceived;
        //                    if (eh != null) {
        //                        try {
        //                            eh(this, s);
        //                        } catch { }
        //                    }
        //                    bytesReceived = 0;
        //                    bytesExpected = 4; // length of incoming string
        //                    receiveMode = 0; // string length
        //                    break;
        //            }
        //        }
        //    }

        //    if (pipe.IsConnected) {
        //        // request more data
        //        try {
        //            pipe.ReadAsync(buffer, bytesReceived, bytesExpected - bytesReceived).ContinueWith(received);
        //        } catch { }

        //    } else {
        //        // pipe broken
        //        try {
        //            pipe.Disconnect();
        //            pipe.BeginWaitForConnection(asyncConnected, null);
        //            //pipe.WaitForConnectionAsync().ContinueWith(connected);
        //        } catch { }
        //    }
        //}
    }
}
