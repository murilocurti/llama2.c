using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
public struct GeneratorParams
{
    public string checkpoint;
    public float temperature;
    public int steps;
}

[StructLayout(LayoutKind.Sequential)]
public struct GeneratorContext
{
    public Config config;
    public TransformerWeights weights;
    public RunState state;
    public IntPtr data; // float *data;
    public int fd;
    public IntPtr vocab; // char **vocab;
    public float temperature;
    public int steps;
    public int pos;
    public int token;
    public long start;
    public UIntPtr file_size; // size_t file_size;
}

[StructLayout(LayoutKind.Sequential)]
public struct Config
{
    public int dim;
    public int hidden_dim;
    public int n_layers;
    public int n_heads;
    public int n_kv_heads;
    public int vocab_size;
    public int seq_len;
}

[StructLayout(LayoutKind.Sequential)]
public struct TransformerWeights
{
    public IntPtr token_embedding_table;
    public IntPtr rms_att_weight;
    public IntPtr rms_ffn_weight;
    public IntPtr wq;
    public IntPtr wk;
    public IntPtr wv;
    public IntPtr wo;
    public IntPtr w1;
    public IntPtr w2;
    public IntPtr w3;
    public IntPtr rms_final_weight;
    public IntPtr freq_cis_real;
    public IntPtr freq_cis_imag;
    public IntPtr wcls;
}

[StructLayout(LayoutKind.Sequential)]
public struct RunState
{
    public IntPtr x;
    public IntPtr xb;
    public IntPtr xb2;
    public IntPtr hb;
    public IntPtr hb2;
    public IntPtr q;
    public IntPtr k;
    public IntPtr v;
    public IntPtr att;
    public IntPtr logits;
    public IntPtr key_cache;
    public IntPtr value_cache;
}


const string DLL = "llama";

[DllImport(DLL)]
public static extern IntPtr initialize(ref GeneratorParams param);

[DllImport(DLL)]
public static extern IntPtr next_token(IntPtr context);

public static string next_token_string(IntPtr context)
{
    IntPtr ptr = next_token(context);
    return Marshal.PtrToStringAnsi(ptr);
}

[DllImport(DLL)]
public static extern void report_stats(IntPtr context);

[DllImport(DLL)]
public static extern void cleanup(IntPtr context);

void print(string s = "")
{
    // write with timestamp
    if (!string.IsNullOrEmpty(s))
    {
        Console.WriteLine($"{DateTime.Now.ToString("HH:mm:ss.fff")} {s}");
    }
    else
    {
        Console.WriteLine();
    }

    Console.Out.Flush();
}

void print(string t, string s = "")
{
    print($"{t}: {s}");
}

void println(string s = "")
{
    print(s);
    Console.WriteLine("-------------------------\n");
}

void println(string t, string s)
{
    println($"{t}: {s}");
}

public class LLama : IDisposable
{
    private bool disposed = false;

    private GeneratorParams generatorParams;
    private GeneratorContext context;
    private IntPtr contextPtr;

    public LLama()
    {
    }

    public LLama(GeneratorParams generatorParams)
    {
        this.generatorParams = generatorParams;
    }

    public string next_token_string(IntPtr contextPtr)
    {
        //IntPtr ptr = next_token(this.contextPtr);
        //Marshal.StructureToPtr(this.context, this.contextPtr, false);
        IntPtr ptr = next_token(this.contextPtr);
        this.context = Marshal.PtrToStructure<GeneratorContext>(this.contextPtr);

        return Marshal.PtrToStringAnsi(ptr);
    }

    public string Generate()
    {
        return this.Generate(generatorParams);
    }

    public string Generate(GeneratorParams generatorParams)
    {
        IntPtr contextPtr = initialize(ref generatorParams);
        GeneratorContext context = Marshal.PtrToStructure<GeneratorContext>(this.contextPtr);

        var sb = new StringBuilder();

        foreach (var token in this.GenerateAsync())
        {
            sb.Append(token);
        }

        return sb.ToString();
    }



    public IEnumerable<string> GenerateAsync()
    {
        if (this.generatorParams == null)
        {
            throw new Exception("GeneratorParams not provided");
        }

        this.contextPtr = initialize(ref generatorParams);
        this.context = Marshal.PtrToStructure<GeneratorContext>(this.contextPtr);

        while (this.context.pos < this.context.steps + 0)
        {
            yield return next_token_string();
        }
    }

    public IEnumerable<string> GenerateAsync(GeneratorParams generatorParams)
    {
        GeneratorContext context = this.GetNewContext();

        while (context.pos < context.steps + 0)
        {
            yield return next_token_string();
        }
    }

    private GeneratorContext GetNewContext()
    {
        if (this.generatorParams == null)
        {
            throw new Exception("GeneratorParams not provided");
        }

        IntPtr contextPtr = initialize(ref generatorParams);
        return Marshal.PtrToStructure<GeneratorContext>(this.contextPtr);
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!disposed)
        {
            if (disposing)
            {
                // Dispose of the managed resources (if they implement IDisposable).

            }

            // Release unmanaged resources 
            cleanup(this.contextPtr);

            disposed = true;
        }
    }
}

// Create a GeneratorParams object
var generatorParams = new GeneratorParams
{
    checkpoint = @"C:\Users\MuriloCurti\source\repos\.murilocurti\stories15M.bin",
    steps = 256,
    temperature = 0.0f
};

using (var llama = new LLama(generatorParams))
{
    foreach (var token in llama.GenerateAsync())
    {
        Console.Write(token);
    }
}