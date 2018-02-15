/*
C# Network Programming 
by Richard Blum

Publisher: Sybex 
ISBN: 0782141765
*/
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Diagnostics;

using System.Collections.Generic;

public class UdpSrvrSample
{
    public static void Main()
    {
        byte[] data = new byte[1024];
        IPEndPoint ipep = new IPEndPoint(IPAddress.Any, 9050);
        UdpClient newsock = new UdpClient(ipep);

        Console.WriteLine("ArchVisMSM dummy server");
        Console.WriteLine("Waiting for MegaMol connection...");

        IPEndPoint sender = new IPEndPoint(IPAddress.Any, 0);

        data = newsock.Receive(ref sender);

        Console.WriteLine("Message received from {0}:", sender.ToString());
        Console.WriteLine(Encoding.ASCII.GetString(data, 0, data.Length));

        string welcome = "Welcome to my test server";
        data = Encoding.ASCII.GetBytes(welcome);
        //newsock.Send(data, data.Length, sender);

        float[] node_displacement = new float[72];
        byte[] byte_buffer = new byte[72*4];

        Stopwatch stopwatch = new Stopwatch();
        stopwatch.Reset();
        stopwatch.Start();

        while (true)
        {
            
            double t = stopwatch.ElapsedMilliseconds / 1000.0;

            for (int i = 0; i < node_displacement.Length; ++i)
            {
                double phase_shift = Math.Floor((double)i / 12.0);

                if (i % 3 == 0)
                {
                    double displacement = Math.Sin(t + phase_shift);
                    node_displacement[i] = (float)displacement * 0.1f;
                }
                else if(i%3 == 1)
                {
                    //double displacement = Math.Cos(t + phase_shift);
                    //node_displacement[i] = (float)displacement * 0.1f;
                }
                else
                {
                    node_displacement[i] = 0.0f;
                }
            }

            Buffer.BlockCopy(node_displacement, 0, byte_buffer, 0, byte_buffer.Length);

            newsock.Send(byte_buffer, 72 * 4, sender);
        }
    }
}