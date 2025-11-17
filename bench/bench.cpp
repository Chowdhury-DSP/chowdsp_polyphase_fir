#include <chowdsp_filters/chowdsp_filters.h>
#include <chowdsp_polyphase_fir.h>

#include <benchmark/benchmark.h>

static constexpr int n_channels = 2;
static constexpr int n_samples = 512;
chowdsp::Buffer<float> buffer { n_channels, n_samples };
chowdsp::Buffer<float> buffer_x2 { n_channels, n_samples * 2 };
chowdsp::Buffer<float> buffer_x3 { n_channels, n_samples * 3 };

static constexpr int n_taps = 57;
static constexpr float coeffs[n_taps] {
    -0.000011466433343440f,
    -0.000048159680438716f,
    -0.000070951498745996f,
    0.000000000000000000f,
    0.000226394896924650f,
    0.000570593070542985f,
    0.000843882357908127f,
    0.000742644459879189f,
    -0.000000000000000001f,
    -0.001387330856962112f,
    -0.002974017320060804f,
    -0.003876072294410999f,
    -0.003078546062261788f,
    0.000000000000000002f,
    0.004911281747189231f,
    0.009897689392343489f,
    0.012239432700612913f,
    0.009302643950572093f,
    -0.000000000000000005f,
    -0.013947257358995015f,
    -0.027649479941983943f,
    -0.034045985830080311f,
    -0.026173578588735643f,
    0.000000000000000007f,
    0.043288592274982135f,
    0.096612134128292462f,
    0.148460098539443669f,
    0.186178664190551207f,
    0.199977588313552862f,
    0.186178664190551235f,
    0.148460098539443669f,
    0.096612134128292462f,
    0.043288592274982128f,
    0.000000000000000007f,
    -0.026173578588735653f,
    -0.034045985830080325f,
    -0.027649479941983936f,
    -0.013947257358995019f,
    -0.000000000000000005f,
    0.009302643950572094f,
    0.012239432700612920f,
    0.009897689392343489f,
    0.004911281747189235f,
    0.000000000000000002f,
    -0.003078546062261790f,
    -0.003876072294411004f,
    -0.002974017320060808f,
    -0.001387330856962112f,
    -0.000000000000000001f,
    0.000742644459879189f,
    0.000843882357908127f,
    0.000570593070542984f,
    0.000226394896924650f,
    0.000000000000000000f,
    -0.000070951498745996f,
    -0.000048159680438716f,
    -0.000011466433343440f,
};

static void ref_interp2 (benchmark::State& state)
{
    chowdsp::ArenaAllocator<> arena { 1 << 14 };
    chowdsp::FIRPolyphaseInterpolator<float, 2, n_taps> ref_filter;
    ref_filter.prepare (n_channels, n_samples, coeffs, arena);
    for (auto _ : state)
    {
        ref_filter.processBlock (buffer, buffer_x2);
    }
}

static void ref_interp3 (benchmark::State& state)
{
    chowdsp::ArenaAllocator<> arena { 1 << 14 };
    chowdsp::FIRPolyphaseInterpolator<float, 3, n_taps> ref_filter;
    ref_filter.prepare (n_channels, n_samples, coeffs, arena);
    for (auto _ : state)
    {
        ref_filter.processBlock (buffer, buffer_x3);
    }
}

static void ref_decim2 (benchmark::State& state)
{
    chowdsp::ArenaAllocator<> arena { 1 << 14 };
    chowdsp::FIRPolyphaseDecimator<float, 2, n_taps> ref_filter;
    ref_filter.prepare (n_channels, n_samples * 2, coeffs, arena);
    for (auto _ : state)
    {
        ref_filter.processBlock (buffer_x2, buffer);
    }
}

static void ref_decim3 (benchmark::State& state)
{
    chowdsp::ArenaAllocator<> arena { 1 << 14 };
    chowdsp::FIRPolyphaseDecimator<float, 3, n_taps> ref_filter;
    ref_filter.prepare (n_channels, n_samples * 3, coeffs, arena);
    for (auto _ : state)
    {
        ref_filter.processBlock (buffer_x3, buffer);
    }
}

static void bench_interp (benchmark::State& s, chowdsp::Buffer<float>& buffer_out, int factor, bool use_avx)
{
    namespace pfir = chowdsp::polyphase_fir;
    const auto alignment = use_avx ? 32 : 16;
    const auto persistent_bytes = pfir::persistent_bytes_required (n_channels, n_taps, factor, n_samples, alignment);
    const auto scratch_bytes = pfir::scratch_bytes_required (n_taps, factor, n_samples, alignment);
    chowdsp::ArenaAllocator<> arena { persistent_bytes + scratch_bytes + alignment };

    auto state = pfir::init (n_channels,
                             n_taps,
                             factor,
                             n_samples,
                             arena.allocate_bytes (persistent_bytes, alignment),
                             alignment);
    pfir::load_coeffs (state, coeffs, n_taps);

    auto* scratch_data = arena.allocate_bytes (scratch_bytes, alignment);
    for (auto _ : s)
    {
        pfir::process_interpolate (state,
                                   buffer.getArrayOfReadPointers(),
                                   buffer_out.getArrayOfWritePointers(),
                                   n_channels,
                                   n_samples,
                                   scratch_data,
                                   use_avx);
    }
}

static void bench_decim (benchmark::State& s, chowdsp::Buffer<float>& buffer_in, int factor, bool use_avx)
{
    namespace pfir = chowdsp::polyphase_fir;
    const auto alignment = use_avx ? 32 : 16;
    const auto persistent_bytes = pfir::persistent_bytes_required (n_channels, n_taps, factor, n_samples * factor, alignment);
    const auto scratch_bytes = pfir::scratch_bytes_required (n_taps, factor, n_samples * factor, alignment);
    chowdsp::ArenaAllocator<> arena { persistent_bytes + scratch_bytes + alignment };

    auto state = pfir::init (n_channels,
                             n_taps,
                             factor,
                             n_samples * factor,
                             arena.allocate_bytes (persistent_bytes, alignment),
                             alignment);
    pfir::load_coeffs (state, coeffs, n_taps);

    auto* scratch_data = arena.allocate_bytes (scratch_bytes, alignment);
    for (auto _ : s)
    {
        pfir::process_decimate (state,
                                   buffer_in.getArrayOfReadPointers(),
                                   buffer.getArrayOfWritePointers(),
                                   n_channels,
                                   n_samples * factor,
                                   scratch_data,
                                   use_avx);
    }
}

static void interp2 (benchmark::State& state)
{
    bench_interp (state, buffer_x2, 2, false);
}

static void interp3 (benchmark::State& state)
{
    bench_interp (state, buffer_x3, 3, false);
}

static void interp2_avx (benchmark::State& state)
{
    bench_interp (state, buffer_x2, 2, true);
}

static void interp3_avx (benchmark::State& state)
{
    bench_interp (state, buffer_x3, 3, true);
}

static void decim2 (benchmark::State& state)
{
    bench_decim (state, buffer_x2, 2, false);
}

static void decim3 (benchmark::State& state)
{
    bench_decim (state, buffer_x3, 3, false);
}

static void decim2_avx (benchmark::State& state)
{
    bench_decim (state, buffer_x2, 2, true);
}

static void decim3_avx (benchmark::State& state)
{
    bench_decim (state, buffer_x3, 3, true);
}

BENCHMARK (ref_interp2)->MinTime (1);
BENCHMARK (ref_interp3)->MinTime (1);
BENCHMARK (interp2)->MinTime (1);
BENCHMARK (interp3)->MinTime (1);
#if defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)
BENCHMARK (interp2_avx)->MinTime (1);
BENCHMARK (interp3_avx)->MinTime (1);
#endif

BENCHMARK (ref_decim2)->MinTime (1);
BENCHMARK (ref_decim3)->MinTime (1);
BENCHMARK (decim2)->MinTime (1);
BENCHMARK (decim3)->MinTime (1);
#if defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)
BENCHMARK (decim2_avx)->MinTime (1);
BENCHMARK (decim3_avx)->MinTime (1);
#endif

int main(int argc, char** argv)
{
   for (auto [ch, data] : chowdsp::buffer_iters::channels (buffer))
   {
       for (auto [n, x] : chowdsp::enumerate (data))
           x = static_cast<float> (n);
   }
   for (auto [ch, data] : chowdsp::buffer_iters::channels (buffer_x2))
   {
       for (auto [n, x] : chowdsp::enumerate (data))
           x = static_cast<float> (n);
   }
   for (auto [ch, data] : chowdsp::buffer_iters::channels (buffer_x3))
   {
       for (auto [n, x] : chowdsp::enumerate (data))
           x = static_cast<float> (n);
   }

   ::benchmark::Initialize(&argc, argv);
   ::benchmark::RunSpecifiedBenchmarks();
}
