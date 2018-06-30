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
using System.Text.RegularExpressions;
using System.Diagnostics;
using System.Globalization;
using System.Collections.Generic;

public struct DataPackage
{
    public DataPackage(int node_cnt, int forces_cnt)
    {
        raw_storage = new byte[4 + node_cnt * 3 * 4 + forces_cnt * 4];
    }

    public byte[] raw_storage;
}

public class Util
{
    static public float[] parseFloatValues(string file_path)
    {
        string file_string = System.IO.File.ReadAllText(file_path);
        file_string = Regex.Replace(file_string, @"\t|\n|\r", ",");
        string[] value_strings = file_string.Split(',');

        float[] values = new float[value_strings.Length];
        for (int i = 0; i < value_strings.Length; ++i)
        {
            if (value_strings[i].Length > 0)
                values[i] = float.Parse(value_strings[i], CultureInfo.InvariantCulture.NumberFormat);
        }

        return values;
    }
}


public class TcpSrvrSample
{
    public static void Run()
    {
        //float[] y_q_t = parseFloatValues("y_q_t_uncon.dat");
        //float[] y_f = parseFloatValues("y_f_uncon.dat");
        float[] y_q_t = Util.parseFloatValues("2018_06_27_2/y_q_t.dat");
        float[] y_f = Util.parseFloatValues("2018_06_27_2/y_f.dat");
        float[] y_q_t_uncon = Util.parseFloatValues("2018_06_27_2/y_q_t_uncon.dat");
        float[] y_f_uncon = Util.parseFloatValues("2018_06_27_2/y_funcon.dat");
        float[] timesteps = Util.parseFloatValues("2018_06_27_2/t.dat");

        Console.WriteLine("Number of displacement values: " + y_q_t.Length.ToString());
        Console.WriteLine("Number of force values: " + y_f.Length.ToString());
        
        byte[] data = new byte[1024];
        IPEndPoint ipep = new IPEndPoint(IPAddress.Parse("141.58.24.247"), 11000);
        TcpListener server = new TcpListener(ipep);

        server.Start();

        Console.WriteLine("ArchVisMSM dummy server");

        Console.Write("Waiting for a connection... ");

        // Perform a blocking call to accept requests.
        // You could also user server.AcceptSocket() here.
        TcpClient client = server.AcceptTcpClient();
        Console.WriteLine("Connected!");

        data = null;

        // Get a stream object for reading and writing
        NetworkStream stream = client.GetStream();

        //float[] node_displacement = new float[72];
        //byte[] byte_buffer = new byte[72*4];

        Stopwatch stopwatch = new Stopwatch();
        stopwatch.Reset();
        stopwatch.Start();

        double t_last = stopwatch.ElapsedMilliseconds / 1000.0; ;

        int simulation_frame_cnt = timesteps.Length;
        int simulation_frame = 0;

        DataPackage simulation_data = new DataPackage(24, 60);
        DataPackage simulation_data_uncon = new DataPackage(24, 60);

        while (true)
        {
            double t = stopwatch.ElapsedMilliseconds / 1000.0;
            

            // identifiy current simulation frame
            if ((t - t_last) > timesteps[simulation_frame])
            {
                simulation_frame = (simulation_frame + 1);

                if (simulation_frame >= timesteps.Length)
                {
                    t_last = t;
                    simulation_frame = 0;
                }
            }

            /////////////////
            // Simulation con
            Buffer.BlockCopy(timesteps, simulation_frame * 4, simulation_data.raw_storage, 0, 4);

            Buffer.BlockCopy(y_f, simulation_frame * 60 * 4, simulation_data.raw_storage, 4 + 24 * 3 * 4, 60 * 4);

            for (int i = 0; i < 60; ++i)
            {
                Buffer.BlockCopy(y_f, simulation_frame * 4 + i * simulation_frame_cnt * 4, simulation_data.raw_storage, 4 + 24 * 3 * 4 + i * 4, 4);
            }

            for (int i = 0; i < 24; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    Buffer.BlockCopy(y_q_t, simulation_frame * 4 + j * simulation_frame_cnt * 4 + i * 3 * 4 * simulation_frame_cnt, simulation_data.raw_storage, 4 + j * 4 + i * 3 * 4, 4);
                }
            }

            ///////////////////
            // Simulation uncon

            Buffer.BlockCopy(timesteps, simulation_frame * 4, simulation_data_uncon.raw_storage, 0, 4);

            Buffer.BlockCopy(y_f_uncon, simulation_frame * 60 * 4, simulation_data_uncon.raw_storage, 4 + 24 * 3 * 4, 60 * 4);

            for (int i = 0; i < 60; ++i)
            {
                Buffer.BlockCopy(y_f_uncon, simulation_frame * 4 + i * simulation_frame_cnt * 4, simulation_data_uncon.raw_storage, 4 + 24 * 3 * 4 + i * 4, 4);
            }

            for (int i = 0; i < 24; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    Buffer.BlockCopy(y_q_t_uncon, simulation_frame * 4 + j * simulation_frame_cnt * 4 + i * 3 * 4 * simulation_frame_cnt, simulation_data_uncon.raw_storage, 4 + j * 4 + i * 3 * 4, 4);
                }
            }


            Console.WriteLine(timesteps[simulation_frame]);
            //for (int i = 0; i < 24; ++i)
            //{
            //    double phase_shift = Math.Floor((double)i / 4.0);
            //
            //    float displacement = (float)Math.Sin(t + phase_shift) * 0.1f;
            //
            //    Buffer.BlockCopy(BitConverter.GetBytes(displacement), 0, simulation_data.raw_storage, 4 + i*3*4, 4);
            //}

            //newsock.Send(simulation_data.raw_storage, simulation_data.raw_storage.Length, sender);
            stream.Write(simulation_data.raw_storage, 0, simulation_data.raw_storage.Length);

            stream.Write(simulation_data_uncon.raw_storage, 0, simulation_data_uncon.raw_storage.Length);
        }
    }
}


public class UdpSrvrSample
{
   public static void Run()
   {
      //float[] y_q_t = parseFloatValues("y_q_t_uncon.dat");
      //float[] y_f = parseFloatValues("y_f_uncon.dat");
      float[] y_q_t = Util.parseFloatValues("y_q_t.dat");
      float[] y_f = Util.parseFloatValues("y_f.dat");
      float[] timesteps = Util.parseFloatValues("t.dat");

      Console.WriteLine("Number of displacement values: " + y_q_t.Length.ToString());
      Console.WriteLine("Number of force values: " + y_f.Length.ToString());
      

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



      //float[] node_displacement = new float[72];
      //byte[] byte_buffer = new byte[72*4];

      Stopwatch stopwatch = new Stopwatch();
      stopwatch.Reset();
      stopwatch.Start();

      double t_last = stopwatch.ElapsedMilliseconds / 1000.0; ;

      int simulation_frame_cnt = timesteps.Length;
      int simulation_frame = 0;

      DataPackage simulation_data = new DataPackage(24, 60);

      while (true)
      {
          double t = stopwatch.ElapsedMilliseconds / 1000.0;

          // identifiy current simulation frame
          if ( (t-t_last) > timesteps[simulation_frame])
          {
              simulation_frame = (simulation_frame + 1);

              if(simulation_frame >= timesteps.Length)
              {
                  t_last = t;
                  simulation_frame = 0;
              }
          }


          Buffer.BlockCopy(timesteps, simulation_frame*4, simulation_data.raw_storage, 0, 4);
          
          Buffer.BlockCopy(y_f, simulation_frame*60*4, simulation_data.raw_storage, 4 + 24 * 3 * 4, 60 * 4);
          
          for(int i=0; i<60; ++i)
          {
              Buffer.BlockCopy(y_f, simulation_frame*4 + i * simulation_frame_cnt * 4, simulation_data.raw_storage, 4 + 24 * 3 * 4 + i*4, 4);
          }

          for(int i=0; i<24; ++i)
          {
              for(int j=0; j<3; ++j)
              {
                  Buffer.BlockCopy(y_q_t, simulation_frame*4 + j*simulation_frame_cnt*4 + i*3*4*simulation_frame_cnt, simulation_data.raw_storage, 4 + j*4 + i*3*4, 4);
              }
          }

          Console.WriteLine(timesteps[simulation_frame]);


          //for (int i = 0; i < 24; ++i)
          //{
          //    double phase_shift = Math.Floor((double)i / 4.0);
          //
          //    float displacement = (float)Math.Sin(t + phase_shift) * 0.1f;
          //
          //    Buffer.BlockCopy(BitConverter.GetBytes(displacement), 0, simulation_data.raw_storage, 4 + i*3*4, 4);
          //}


          newsock.Send(simulation_data.raw_storage, simulation_data.raw_storage.Length, sender);
      }
   }
}

public class EntryPoint
{
    public static void Main()
    {
        TcpSrvrSample.Run();
    }
}