// inferzero driver: feeds token ids, prints top logits and a greedy continuation; or benchmarks prefill.
// usage (every option: runz --help):
//   runz model.gguf (PROMPT | --text-file F | --tokens 1,2,3 | --tokens-file ids.txt) [--n 16] [--prof] [--chunk N] [--ctx N]
//   runz model.gguf --bench-pp 2048 [--prof]
#include "../src/full_speed.h"
#include "engine.h"  // inferzero (Level Zero)
#include "../src/tokenizer.h"
#include <cstdio>
#include <cwctype>
#include <fstream>
#include <sstream>
#ifdef _WIN32
#include <tlhelp32.h>
#endif

static std::vector<int> parse_ids(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) if (!tok.empty() && tok != "\r" && tok != "\n") v.push_back(std::stoi(tok));
  return v;
}

// --kl-ref's reference: either every position's log-probs over the vocabulary (f32, tools/ref_forward REF_ALL), or the
// compact form (2026-11-19, bench/prompts/refs): "IZKL", version 1, V, positions, K (uint32 each), then per position K
// token ids (int32), their log-probs (f32) and the log of the probability left to the other tokens (f32). The compact
// form is read into full rows with the tokens outside the top K at a sentinel (-1e37, which the KL loops skip as they
// skip a log-prob of -inf), and the rest's mass kept per row: kl_tail adds it as one bucket against the model's own
// probability outside those K tokens. The top token and the margins are exact; the KL is the full one's with the tail
// taken as one bucket (a lower bound: the tail's own spread is not seen; 2 % low at K = 512). ~1 MB a prompt instead of ~250 MB.
// Empty on a read error.
static std::vector<float> g_ref_tail;  // (the compact reference's log rest mass per position; empty for a full one)
static const float* g_ref_base = nullptr;
static size_t g_ref_v = 0;
static double kl_tail(const float* l, const float* rr) {
  if (g_ref_tail.empty() || !g_ref_base) return 0.0;
  const size_t r = (size_t)(rr - g_ref_base) / g_ref_v;
  if (r >= g_ref_tail.size() || !(g_ref_tail[r] > -1e30f)) return 0.0;
  double mx = l[0];
  for (size_t i = 1; i < g_ref_v; ++i) mx = std::max(mx, (double)l[i]);
  double z = 0, qtop = 0;
  for (size_t i = 0; i < g_ref_v; ++i) {
    const double e = std::exp((double)l[i] - mx);
    z += e;
    if (rr[i] > -1e36f) qtop += e;
  }
  const double pt = std::exp((double)g_ref_tail[r]), qt = std::max(1.0 - qtop / z, 1e-30);
  return pt * ((double)g_ref_tail[r] - std::log(qt));
}
static std::vector<float> read_kl_ref(const std::string& path, size_t V) {
  std::vector<float> ref;
  FILE* fi = fopen(path.c_str(), "rb");
  if (!fi) return ref;
  char magic[4] = {};
  uint32_t h[4] = {};
  if (fread(magic, 1, 4, fi) == 4 && memcmp(magic, "IZKL", 4) == 0) {
    if (fread(h, 4, 4, fi) != 4 || h[0] != 1 || h[1] != V || h[3] == 0 || h[3] >= V) { fclose(fi); return ref; }
    g_ref_tail.clear();
    const size_t npos = h[2], K = h[3];
    ref.resize(npos * V);
    std::vector<int32_t> ids(K);
    std::vector<float> lp(K);
    for (size_t r = 0; r < npos; ++r) {
      float tail = 0;
      if (fread(ids.data(), 4, K, fi) != K || fread(lp.data(), 4, K, fi) != K || fread(&tail, 4, 1, fi) != 1) {
        ref.clear();
        break;
      }
      float* row = ref.data() + r * V;
      std::fill(row, row + V, -1e37f);
      g_ref_tail.push_back(tail);
      for (size_t k = 0; k < K; ++k)
        if (ids[k] >= 0 && (size_t)ids[k] < V) row[ids[k]] = lp[k];
    }
    fclose(fi);
    if (ref.empty()) g_ref_tail.clear();
    g_ref_base = ref.data(), g_ref_v = V;  // (kl_tail's rows: the vector's buffer moves with it, not copied)
    return ref;
  }
  plat::fseek64(fi, 0, SEEK_END);  // (64-bit: references of more than 2 GB)
  ref.resize((size_t)plat::ftell64(fi) / 4);
  plat::fseek64(fi, 0, SEEK_SET);
  if (fread(ref.data(), 4, ref.size(), fi) != ref.size()) ref.clear();
  fclose(fi);
  return ref;
}
// (ARC_KL_FLIPS=1, --kl-ref: a position whose top token is not the reference's, with the reference's top two, their
//  margin in log-prob, and where our pick stands in the reference: a near-tie (margin ~0) or a real miss. 2026-11-19,
//  for Flash Next's same-top-token rule on discrete cards)
static void kl_flip(size_t pos, const float* l, const float* rr, size_t V) {
  static const bool on = getenv("ARC_KL_FLIPS") && atoi(getenv("ARC_KL_FLIPS")) != 0;
  if (!on) return;
  const size_t te = std::max_element(l, l + V) - l, t1 = std::max_element(rr, rr + V) - rr;
  if (te == t1) return;
  size_t t2 = t1 == 0 ? 1 : 0;
  for (size_t i = 0; i < V; ++i)
    if (i != t1 && rr[i] > rr[t2]) t2 = i;
  printf("flip pos %zu: ref top %zu (p %.4f), 2nd %zu (p %.4f), margin %.4f | inferzero top %zu (ref p %.4f)\n", pos, t1,
         std::exp((double)rr[t1]), t2, std::exp((double)rr[t2]), (double)rr[t1] - rr[t2], te, std::exp((double)rr[te]));
}
static void top_k(const std::vector<float>& logits, int k, std::vector<std::pair<int, float>>& out) {
  std::vector<int> idx(logits.size());
  for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
  float mx = logits[idx[0]];
  double z = 0;
  for (float l : logits) z += std::exp((double)l - mx);
  out.clear();
  for (int i = 0; i < k; ++i) out.push_back({idx[i], (float)(std::exp((double)logits[idx[i]] - mx) / z)});
}

// Other processes on the iGPU (serverz / another runz, llama.cpp) share its time and its memory, so speeds measured while one of
// them runs are not representative: Gemma 4 E4B generated 8 tok/s with --mtp (23 alone) while a 27B server
// processed a prompt, and its load took 10-40 s instead of 2 s with an idle 27B server loaded.
static void warn_gpu_neighbours() {
#ifndef _WIN32
  // (Linux: the processes' names in /proc/<pid>/comm, "serverz", "runz", "llama-...")
  std::string found;
  std::error_code ec;
  const std::string me = std::to_string(plat::process_id());
  // (the error_code forms throughout: a process ending during the walk must not end runz with an exception)
  for (std::filesystem::directory_iterator it("/proc", ec), end; !ec && it != end; it.increment(ec)) {
    const std::filesystem::path& d = it->path();
    const std::string pid = d.filename().string();
    if (pid.empty() || pid.find_first_not_of("0123456789") != std::string::npos || pid == me) continue;
    std::ifstream f(d / "comm");
    std::string n;
    if (!std::getline(f, n)) continue;
    if (n == "serverz" || n == "runz" || n.rfind("llama-", 0) == 0) found += (found.empty() ? "" : ", ") + n + " (pid " + pid + ")";
  }
#else
  HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (snap == INVALID_HANDLE_VALUE) return;
  PROCESSENTRY32W pe{};
  pe.dwSize = sizeof(pe);
  std::string found;
  for (BOOL ok = Process32FirstW(snap, &pe); ok; ok = Process32NextW(snap, &pe)) {
    if (pe.th32ProcessID == GetCurrentProcessId()) continue;
    std::string n;
    for (const wchar_t* c = pe.szExeFile; *c; ++c) n.push_back((char)towlower(*c));
    if (n == "serverz.exe" || n == "runz.exe" || n.rfind("llama-", 0) == 0)
      found += (found.empty() ? "" : ", ") + n + " (pid " + std::to_string(pe.th32ProcessID) + ")";
  }
  CloseHandle(snap);
#endif
  if (!found.empty())
    zlog::eprintf("note: other GPU processes are running: %s. They share the iGPU and its memory, so the speeds "
                    "measured now are not representative.\n", found.c_str());
}

// " (P % of this GPU's measured N GB/s)": the read bandwidth measured once on this machine (Engine::read_gbs)
static std::string peak_note(Engine& eng, double gbs) {
  static double peak = -1;
  if (peak < 0) {
    try { peak = eng.read_gbs(); } catch (const std::exception&) { peak = 0; }  // (no room left for its buffer)
  }
  // (a pass that read its weights faster than the probe's figure: the probe timed something else, so no figure; #45, #80)
  if (peak <= 0 || gbs > peak * 1.1) return "";
  char s[96];
  snprintf(s, sizeof(s), " (%.0f %% of this GPU's measured %.0f GB/s)", gbs / peak * 100, peak);
  return s;
}

// (--prof / --prof-gen on a split model, 2026-11-23: the split paths took no profile flags, so they printed nothing
//  (MaineTim's B60 pair, #44; the dual A770, #45). One table per GPU (a layer split's stage, a tensor split's shard):
//  its kernels' GPU time in the last profiled prompt pass or over the profiled generation steps)
static void print_split_prof(Engine& e, const char* what, int r) {
  double total = 0;
  for (auto& [n, ms] : e.prof_ms) total += ms;
  std::vector<std::pair<double, std::string>> rows;
  for (auto& [n, ms] : e.prof_ms) rows.push_back({ms, n});
  std::sort(rows.rbegin(), rows.rend());
  printf("GPU time by kernel, %s %d: total %.1f ms\n", what, r, total);
  for (auto& [ms, n] : rows) printf("  %-20s %8.2f ms  %5.1f%%  (%d launches)\n", n.c_str(), ms, total > 0 ? 100 * ms / total : 0.0, e.prof_n[n]);
  e.prof_ms.clear();
  e.prof_n.clear();
}
// --gpus N --split layer: the model's layers over N GPUs (zero/mgpu/engine_pp.h): the prompt (or --bench-pp N), then
// greedy generation, as run_tp; --tensor-split a,b,...: each GPU's share of the layers
static int run_pp(const std::string& model, const EngineOptions& eo, int gpus, const std::vector<double>& shares,
                  const std::vector<int>& toks, int n_gen, int bench_pp, const std::string& dump_logits, bool show,
                  bool prof = false, bool prof_gen = false) {
  PpEngine pp(model, eo, gpus, shares);
  Engine& el = pp.last();
  Vocab vocab(el.gguf());
  printf("loaded: %d GPUs by layers, %d layers, n_embd %d, vocab %d | GPU buffers", gpus, el.hp.n_layer, el.hp.n_embd,
         el.hp.n_vocab);
  for (int r = 0; r < gpus; ++r) printf(" %.2f", pp.gpu_bytes(r) / 1073741824.0);
  printf(" GiB | chunk %d | ctx %d\n", el.max_chunk, el.ctx);
  auto ms_since = [](std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
  };
  std::vector<int> prompt = toks;
  if (bench_pp > 0) {
    prompt.resize(bench_pp);
    for (int i = 0; i < bench_pp; ++i) prompt[i] = 1000 + (i * 7919) % 50000;
  }
  if (prompt.empty()) { zlog::eprintf("no tokens\n"); return 1; }
  if (bench_pp > 0) pp.forward(prompt.data(), std::min((int)prompt.size(), 64));  // warm-up (kernels, page-in)
  const int runs = bench_pp > 0 ? 3 : 1;
  std::vector<float> lg;
  for (int r = 0; r < runs; ++r) {
    if (bench_pp > 0) pp.reset();
    for (int s = 0; s < pp.stages(); ++s) pp.stage(s).profile = prof && !prof_gen && r == runs - 1;
    const auto t0 = std::chrono::steady_clock::now();
    lg = pp.forward(prompt.data(), (int)prompt.size());
    const double ms = ms_since(t0);
    printf("pp%zu run %d: %.1f ms  ->  %.1f tokens/s%s\n", prompt.size(), r, ms, prompt.size() / ms * 1000.0,
           prof && !prof_gen && r == runs - 1 ? " (profiled)" : "");
  }
  if (prof && !prof_gen)
    for (int s = 0; s < pp.stages(); ++s) {
      pp.stage(s).profile = false;
      print_split_prof(pp.stage(s), "prompt, stage", s);
    }
  for (int s = 0; s < pp.stages(); ++s) pp.stage(s).profile = prof_gen;  // (--prof-gen: the generation steps below)
  if (bench_pp <= 0) {
    std::vector<std::pair<int, float>> tk;
    top_k(lg, 10, tk);
    printf("top-10 next token:");
    for (auto& [id, p] : tk) printf(" %d(%.4f)", id, p);
    printf("\n");
  }
  FILE* dl = dump_logits.empty() ? nullptr : fopen(dump_logits.c_str(), "wb");
  if (dl) fwrite(lg.data(), sizeof(float), lg.size(), dl);
  int tok = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
  if (bench_pp > 0) tok = 1000;
  std::vector<int> gen;
  std::vector<double> step_ms;
  const auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < n_gen; ++i) {
    gen.push_back(tok);
    if (bench_pp <= 0 && tok == vocab.eos) break;
    const auto ts = std::chrono::steady_clock::now();
    const auto& l2 = pp.forward(&tok, 1);
    step_ms.push_back(ms_since(ts));
    if (dl) fwrite(l2.data(), sizeof(float), l2.size(), dl);
    tok = (int)(std::max_element(l2.begin(), l2.end()) - l2.begin());
  }
  if (dl) fclose(dl);
  const double gms = ms_since(t1);
  if (bench_pp <= 0) {
    printf("greedy:");
    for (int id : gen) printf(" %d", id);
    printf("\n");
  }
  if (!step_ms.empty()) {
    std::vector<double> st = step_ms;
    std::sort(st.begin(), st.end());
    printf("tg%zu: %.1f ms  ->  %.2f tokens/s | steps min %.1f, median %.1f ms%s\n", step_ms.size(), gms,
           step_ms.size() / gms * 1000.0, st.front(), st[st.size() / 2], prof_gen ? " (profiled: slower)" : "");
  }
  if (prof_gen)
    for (int s = 0; s < pp.stages(); ++s) {
      pp.stage(s).profile = false;
      printf("(over %zu generation steps)\n", step_ms.size());
      print_split_prof(pp.stage(s), "generation, stage", s);
    }
  if (show) printf("----- text -----\n%s\n----------------\n", vocab.decode(gen).c_str());
  return 0;
}

// --gpus N: the model split over N GPUs (zero/engine_tp.h): the prompt (or --bench-pp N), then greedy generation
static int run_tp(const std::string& model, const EngineOptions& eo, int gpus, const std::vector<int>& toks, int n_gen,
                  int bench_pp, const std::string& dump_logits, bool show, const std::string& kl_ref, int bench_batch,
                  bool prof = false, bool prof_gen = false) {
  TpEngine tp(model, eo, gpus);
  Engine& e0 = tp.shard(0);
  Vocab vocab(e0.gguf());
  printf("loaded: %d GPUs, %d layers, n_embd %d, vocab %d | GPU buffers %.2f GiB per GPU | W8 %s | chunk %d | ctx %d | load %.0f ms (GPU 0)\n",
         gpus, e0.hp.n_layer, e0.hp.n_embd, tp.vocab(), e0.gpu_bytes / 1073741824.0,
         e0.w8_resident ? "resident" : "just-in-time", e0.max_chunk, e0.ctx, e0.load_ms);
  auto ms_since = [](std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
  };
  auto syncs = [&] {  // (GPU 0's time in the shards' sums: its GPU's wait + the barrier)
    return std::make_pair(tp.comm().wait_ms[0], tp.comm().reduces[0]);
  };
  auto read_bytes = [&] {
    uint64_t b = 0;
    for (int r = 0; r < gpus; ++r) b += tp.shard(r).rd_bytes;
    return b;
  };
  std::vector<int> prompt = toks;
  if (bench_pp > 0) {
    prompt.resize(bench_pp);
    for (int i = 0; i < bench_pp; ++i) prompt[i] = 1000 + (i * 7919) % 50000;
  }
  if (prompt.empty()) { zlog::eprintf("no tokens\n"); return 1; }
  if (bench_batch > 1) {  // batched decoding over the split model's slots: bit-identical to each slot alone, then the speed
    const int NS = bench_batch, K = 8;
    if (!tp.batch_ok() || tp.slot_count() < NS) { zlog::eprintf("--bench-batch: no batched decoding for this model\n"); return 1; }
    if ((int)prompt.size() < NS + 8) { zlog::eprintf("--bench-batch %d: the prompt needs at least %d tokens\n", NS, NS + 8); return 1; }
    const size_t V = (size_t)tp.vocab();
    auto fnv = [&](const float* p) {
      uint64_t h = 1469598103934665603ull;
      const uint8_t* b = (const uint8_t*)p;
      for (size_t i = 0; i < V * 4; ++i) h = (h ^ b[i]) * 1099511628211ull;
      return h;
    };
    auto argmax = [&](const float* p) { return (int)(std::max_element(p, p + V) - p); };
    // slot s: the prompt less its last s tokens (each slot its own conversation)
    auto feed = [&](int s) -> const std::vector<float>& {
      tp.bind(s);
      tp.reset();
      return tp.forward(prompt.data(), (int)prompt.size() - s);
    };
    std::vector<std::vector<uint64_t>> alone(NS), batched(NS);
    std::vector<int> sl(NS), tk(NS);
    for (int s = 0; s < NS; ++s) {  // each slot alone, K greedy steps
      int t = argmax(feed(s).data());
      for (int k = 0; k < K; ++k) {
        const std::vector<float>& l = tp.forward(&t, 1);
        alone[s].push_back(fnv(l.data()));
        t = argmax(l.data());
      }
    }
    for (int s = 0; s < NS; ++s) {
      sl[s] = s;
      tk[s] = argmax(feed(s).data());
    }
    for (int k = 0; k < K; ++k) {  // the same K steps, every slot in one pass
      const std::vector<float>& l = tp.forward_batch(sl, tk);
      for (int s = 0; s < NS; ++s) {
        batched[s].push_back(fnv(&l[s * V]));
        tk[s] = argmax(&l[s * V]);
      }
    }
    const bool same = alone == batched;
    printf("batched vs each slot alone, %d slots x %d steps: logits %s\n", NS, K, same ? "bit-identical" : "DIFFER");
    const int reps = 16;
    for (int B = 1; B <= std::min(NS, tp.batch_rows()); ++B) {
      double best = 1e30, one = 1e30, wait = 0;
      std::vector<int> bs(sl.begin(), sl.begin() + B), bt(tk.begin(), tk.begin() + B);
      for (int round = 0; round < 3; ++round) {
        const auto [w0, n0] = syncs();
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) tp.forward_batch(bs, bt);
        const double ms = ms_since(t0) / reps;
        if (ms < best) { best = ms; wait = (syncs().first - w0) / reps; }
        t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) tp.forward(&bt[0], 1);
        one = std::min(one, ms_since(t0) / reps);
      }
      printf("B %d: %.1f ms per batched step (%.1f tok/s together, %.2f ms waiting for the sums) | one slot alone %.1f ms (%.1f tok/s): %.2fx\n",
             B, best, 1000.0 * B / best, wait, one, 1000.0 / one, one * B / best);
    }
    return same ? 0 : 1;
  }
  if (!kl_ref.empty()) {  // (--kl-ref with every position's reference log-probs: the prompt fed token by token, as --kl-decode)
    // (Inkling 2: the head's rows past the vocabulary are -inf logits; the reference holds the vocabulary only)
    const size_t V = e0.hp.inkling ? (size_t)e0.ink_vocab() : (size_t)tp.vocab();
    const std::vector<float> ref = read_kl_ref(kl_ref, V);
    if (ref.empty()) { zlog::eprintf("cannot read %s\n", kl_ref.c_str()); return 1; }
    if (ref.size() != V * prompt.size()) { zlog::eprintf("reference must hold every position\n"); return 1; }
    double sum = 0, mxk = 0;
    int agree = 0;
    // the prompt path first (a prompt of one chunk: every row's logits from the rows the pass left, keep_rows), then
    // the same tokens fed one by one
    if ((int)prompt.size() <= e0.max_chunk) {
      for (int r = 0; r < gpus; ++r) tp.shard(r).keep_rows = true;
      tp.forward(prompt.data(), (int)prompt.size());
      for (size_t r = 0; r < prompt.size(); ++r) {
        const auto& l = tp.logits_at((int)r);
        const double mx = *std::max_element(l.begin(), l.end());
        double z = 0;
        for (float v : l) z += std::exp((double)v - mx);
        const float* rr = ref.data() + r * V;
        double kl = 0;
        for (size_t i = 0; i < V; ++i)
          if (rr[i] > -1e30f) kl += std::exp((double)rr[i]) * (rr[i] - (l[i] - mx - std::log(z)));
        kl += kl_tail(l.data(), rr);  // (a compact reference's rest, read_kl_ref)
        sum += kl;
        mxk = std::max(mxk, kl);
        agree += (std::max_element(l.begin(), l.end()) - l.begin()) == (std::max_element(rr, rr + V) - rr); kl_flip(r, l.data(), rr, V);
      }
      printf("KL prompt path (ref || inferzero) = %.5f (mean over %zu positions, max %.5f), top-1 agreement %d/%zu\n",
             sum / prompt.size(), prompt.size(), mxk, agree, prompt.size());
      for (int r = 0; r < gpus; ++r) tp.shard(r).keep_rows = false;
      tp.reset();
      sum = mxk = 0;
      agree = 0;
    }
    for (size_t r = 0; r < prompt.size(); ++r) {
      const auto& l = tp.forward(&prompt[r], 1);
      const double mx = *std::max_element(l.begin(), l.end());
      double z = 0;
      for (float v : l) z += std::exp((double)v - mx);
      const float* rr = ref.data() + r * V;
      double kl = 0;
      for (size_t i = 0; i < V; ++i)
        if (rr[i] > -1e30f) kl += std::exp((double)rr[i]) * (rr[i] - (l[i] - mx - std::log(z)));
      kl += kl_tail(l.data(), rr);  // (a compact reference's rest, read_kl_ref)
      sum += kl;
      mxk = std::max(mxk, kl);
      agree += (std::max_element(l.begin(), l.end()) - l.begin()) == (std::max_element(rr, rr + V) - rr); kl_flip(r, l.data(), rr, V);
    }
    printf("KL decode path (ref || inferzero) = %.5f (mean over %zu positions, max %.5f), top-1 agreement %d/%zu\n",
           sum / prompt.size(), prompt.size(), mxk, agree, prompt.size());
    return 0;
  }
  tp.forward(prompt.data(), std::min((int)prompt.size(), 64));  // warm-up (kernels, page-in)
  const int runs = bench_pp > 0 ? 3 : 1;
  std::vector<float> lg;
  // (--prof / --prof-gen on the tensor split: each shard's events, collected on its own GPU)
  auto shard_prof = [&](bool on) { for (int r = 0; r < gpus; ++r) tp.shard(r).profile = on; };
  auto shard_tables = [&](const char* what) {
    zrt::Dev* const saved = zrt::cur;
    for (int r = 0; r < gpus; ++r) {
      zrt::cur = zrt::devs[r];
      tp.shard(r).collect_profile_pub();
      tp.shard(r).profile = false;
      print_split_prof(tp.shard(r), what, r);
    }
    zrt::cur = saved;
  };
  for (int r = 0; r < runs; ++r) {
    tp.reset();
    shard_prof(prof && !prof_gen && r == runs - 1);
    const auto [w0, n0] = syncs();
    const auto t0 = std::chrono::steady_clock::now();
    lg = tp.forward(prompt.data(), (int)prompt.size());
    const double ms = ms_since(t0);
    const auto [w1, n1] = syncs();
    printf("pp%zu run %d: %.1f ms  ->  %.1f tokens/s (sums: %lld, %.1f ms waiting)\n", prompt.size(), r, ms,
           prompt.size() / ms * 1000.0, (long long)(n1 - n0), w1 - w0);
  }
  if (prof && !prof_gen) shard_tables("prompt, GPU");
  shard_prof(prof_gen);  // (--prof-gen: the generation steps below)
  if (bench_pp <= 0) {
    std::vector<std::pair<int, float>> tk;
    top_k(lg, 10, tk);
    printf("top-10 next token:");
    for (auto& [id, p] : tk) printf(" %d(%.4f)", id, p);
    printf("\n");
  }
  FILE* dl = dump_logits.empty() ? nullptr : fopen(dump_logits.c_str(), "wb");
  if (dl) fwrite(lg.data(), sizeof(float), lg.size(), dl);
  int tok = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
  if (bench_pp > 0) tok = 1000;
  std::vector<int> gen;
  std::vector<double> step_ms;
  const uint64_t b0 = read_bytes();
  const auto [w0, n0] = syncs();
  const auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < n_gen; ++i) {
    gen.push_back(tok);
    if (bench_pp <= 0 && tok == vocab.eos) break;
    const auto ts = std::chrono::steady_clock::now();
    const auto& l2 = tp.forward(&tok, 1);
    step_ms.push_back(ms_since(ts));
    if (dl) fwrite(l2.data(), sizeof(float), l2.size(), dl);
    tok = (int)(std::max_element(l2.begin(), l2.end()) - l2.begin());
  }
  if (dl) fclose(dl);
  const double gms = ms_since(t1);
  const auto [w1, n1] = syncs();
  if (bench_pp <= 0) {
    printf("greedy:");
    for (int id : gen) printf(" %d", id);
    printf("\n");
  }
  if (!step_ms.empty()) {
    std::vector<double> st = step_ms;
    std::sort(st.begin(), st.end());
    printf("tg%zu: %.1f ms  ->  %.2f tokens/s | steps min %.1f, median %.1f ms | sums per token %.0f, %.2f ms of each token waiting\n",
           step_ms.size(), gms, step_ms.size() / gms * 1000.0, st.front(), st[st.size() / 2],
           (double)(n1 - n0) / step_ms.size(), (w1 - w0) / step_ms.size());
    const double gb = (read_bytes() - b0) / 1e9;
    printf("weights read: %.0f MB per token over %d GPUs, %.1f GB/s together\n", gb * 1000 / step_ms.size(), gpus, gb / gms * 1000);
  }
  if (prof_gen) {
    printf("(over %zu generation steps, profiled: slower)\n", step_ms.size());
    shard_tables("generation, GPU");
  }
  if (show) printf("----- text -----\n%s\n----------------\n", vocab.decode(gen).c_str());
  return 0;
}

static int run_main(int argc, char** argv);
int main(int argc, char** argv) {
  full_speed_process();
  try {
    return run_main(argc, argv);
  } catch (const std::exception& e) {
    zlog::eprintf("error: %s\n", e.what());
    return 1;
  }
}

static int run_main(int argc, char** argv) {
  if (argc >= 2 && std::string(argv[1]) == "--clear-hang") {  // (the failsafe's marker, src/gpu_fail.h; no GPU)
    const std::string p = gpufail::marker_path();
    if (gpufail::clear_marker()) printf("deleted the GPU hang marker %s\n", p.c_str());
    else printf("no GPU hang marker (%s)\n", p.c_str());
    return 0;
  }
  const bool help = argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h");
  if (argc < 2 || help) {
    fprintf(stderr,
            "usage: runz model.gguf PROMPT [options]  feeds a prompt, prints the top next tokens and a greedy continuation\n"
            "       runz model.gguf --bench-pp N [--n N] prompt processing of N tokens, then generation speed\n"
            "       runz --clear-hang     delete the GPU hang marker (after a GPU hang, GPU programs refuse to start until the\n"
            "                             computer restarts; docs/gpu-failsafe.md)\n"
            "prompt:\n"
            "  --tokens 1,2,3        token ids               --tokens-file F   token ids from a file (comma separated)\n"
            "  --text-file F         text, tokenized with the model's vocabulary (special tokens such as <|im_start|> parsed)\n"
            "  --dump-ids F          write the prompt's token ids to F and exit\n"
            "  --must-fit            exit (code 6) when the model and the context exceed the GPU share (a warning without)\n"
            "generation:\n"
            "  --n N                 tokens to generate (default 16)     --show   print the generated text\n"
            "  --dump-logits F       write the prompt logits and every generation step's (f32) to F, to compare runs bitwise\n"
            "  --ring-snap-test      Gemma 4: a snapshot's ring copy restored after the rings moved on (a 7,600+ token\n"
            "                        prompt), the next tokens' logits against a fresh run, bit for bit\n"
            "  --snap-test           a snapshot (device, then host) restored after other tokens, the next tokens' logits\n"
            "                        against a fresh run, bit for bit (DeepSeek-V4.1's rings and pending rows)\n"
            "  --mtp                 speculative decoding (MTP block / Gemma drafter)\n"
            "  --mtp-drafts N        drafts per step, 2..7 (default: 27B 7 on iGPUs, 4 on the A770 / B60 / B65 / B580, 3 on\n"
            "                        the B70; 35B-A3B 5; Qwen 4B and\n"
            "                        smaller, Flash Next 3; 9B 2; with --dflash 7); Gemma 4's drafter always drafts 3\n"
            "  --mtp-model F         the MTP head or drafter in a file of its own (Gemma 4 drafter; Qwen3.8 Flash Next's\n"
            "                        mtp-*.gguf head; for a dense Qwen3.5 / 3.8 fine-tune without an MTP block, the\n"
            "                        base model's .gguf, whose block must match it); default: an mtp-*.gguf next to the model\n"
            "  --dflash DIR          speculative decoding with a DFlash2 block drafter (Qwen3.8 27B, Qwen3.6 35B-A3B, Flash Next)\n"
            "  --draft auto|mtp|dflash  with --dflash: both drafters picked per step (auto), the MTP block only, or\n"
            "                        DFlash2 only (the MTP block is not loaded)\n"
            "  --mtp-min-p P         draft a second token only above this MTP probability (default 0.5)\n"
            "  --copy-draft          with --mtp: where the last tokens appeared earlier in the context, draft what followed\n"
            "                        them there (copy drafting; the same output, verified as any drafts)\n"
            "engine:\n"
            "  --ctx N               context (default: what the prompt needs)     --chunk N   prompt tokens per step\n"
            "  --feed-step N         feed the prompt in calls of N tokens (tests resuming mid-prompt)\n"
            "  --budget GB           keep decoded weights resident when the model and them fit in GB (default 9; Alchemist\n"
            "                        cards (A770, A750, A580) without --budget: decoded just in time, for the Arc 140T\n"
            "                        preset); above it they are decoded just in time, and on discrete cards the layers\n"
            "                        loaded first keep a resident copy in the room the context leaves (iGPUs with 16-lane\n"
            "                        XMX and a GPU share of 64 GiB or more: up to 16 GiB; ARC_W8_PARTIAL_GB=0: off, a\n"
            "                        number: that many GB; on other iGPUs only a number works)\n"
            "  --gpus N              the model split over N GPUs (Qwen3.5 / 3.6 / 3.8, the 35B-A3B, Flash Next,\n"
            "                        DeepSeek-V4.1): the prompt or --bench-pp, then greedy generation (ARC_TP_SIM=1: N\n"
            "                        shards on GPU 0, to test the split)\n"
            "  --split tensor|layer  with --gpus: tensor (the default) splits each layer's matrices over the GPUs, with\n"
            "                        sums every layer; layer gives each GPU whole layers, one hand-off a token (Qwen's dense\n"
            "                        models for now, no --mtp yet): no sums, so slow links cost little, but one request\n"
            "                        runs at about one card's speed. --tensor-split a,b,...: each GPU's share of the layers\n"
            "  --attn-f32           f32 attention (f16 K/V)      --act256    one activation scale per 256 values (not 32)\n"
            "  --npu on|off|force    the NPU computes part of each prompt chunk next to the GPU (27B, Gemma 12B; force: any\n"
            "                        Qwen3.5 / 3.6 / 3.8 or Gemma 4 model; default off, or ARC_NPU)\n"
            "profiling:\n"
            "  --prof                time to first token and per-kernel profile of the prompt\n"
            "  --driver-reserve-gb N GPU memory left to the driver, GiB (default 0.5; ARC_DRIVER_RESERVE_GB too)\n"
            "  --prof-gen            per-kernel profile of generation (ARC_PROF_LAYERS=8: per group of 8 layers); with\n"
            "                        --bench-batch: each kernel's GPU time of a step alone and batched, side by side\n"
            "accuracy:\n"
            "  --kl-ref F            KL of the next-token distributions against an f32 reference (tools\\ref_forward)\n"
            "  --kl-decode           with --kl-ref: through the token-by-token generation path\n"
            "  (ARC_KL_FLIPS=1 with --kl-ref: each position whose top token is not the reference's, with the reference's\n"
            "   top two and their margin)\n"
            "  --slots-check         two conversations (the prompt's halves) alone, then held at once in slots 0 and 1\n"
            "                        with their steps interleaved and batched: every logit must be bit for bit the same\n"
            "  --bench-batch N       time of a batched decoding step over 1..N slots against one slot alone; with --gpus,\n"
            "                        first a check that the batched steps' logits equal each slot's alone, bit for bit\n"
            "  --perplexity F        perplexity of a text file (llama-perplexity method: second half of each chunk scored)\n"
            "  --ppl-ctx N           chunk length (default 512)   --ppl-chunks N   chunks to score (default all)\n"
            "  --ppl-dump F          write the chunks' token ids to F (for ref_forward)\n"
            "  --ppl-kl-save F       save each scored token's top-64 log-probs   --ppl-kl F   KL against such a file\n"
            "  --ppl-score-last N    score only the last N positions of each chunk   --ppl-score-step S  feed those S at a\n"
            "                        time (S <= 8: through the generation / verification kernels)\n"
            "PROMPT is text (special tokens such as <|im_start|> parsed); a long one: --text-file (Windows limits a\n"
            "command line to ~32K characters).\n"
            "environment (tests): ARC_NO_XMX=1 DP4a instead of XMX (Xe2 / Xe3), ARC_XMX_TRY=1 the XMX check on any GPU (=2 / =3: a\n"
            "  right / wrong DP4a variant in the XMX slot, the check must pass / fail),\n"
            "  ARC_NO_SG8=1 the 16-lane kernels, ARC_G4_Q2MM=1 Gemma E4B on two passes, ARC_FN_Q2=1 Flash Next's dense\n"
            "  matmuls on two passes, ARC_ZE_KPROPS=1 register spills per kernel (ARC_NO_KCACHE=1: build them again),\n"
            "  ARC_KLIST=1 each kernel's cache file and build options (for ocloc disasm),\n"
            "  ARC_VIT_Q2=0 / 1 the Qwen vision encoder's int8 matmuls off / on, A/B switches of the 2026-10-27 kernel\n"
            "  changes: ARC_MOE_TM_SMALL=0 (MoE tiles of 16 for 2-8 rows), ARC_MOE_GR_DN_S=1, ARC_RG_IMUL=1; of the stacked\n"
            "  float matrices (BF16 models, 2026-10-31): ARC_NO_QZ_FLT=1, ARC_NO_GU_FLT=1, ARC_NO_QKV_FLT=1, ARC_NO_QZAB16=1;\n"
            "  the discrete cards' defaults of 2026-11-15 to 17 turned off (iGPUs do not use them, but for the first):\n"
            "  ARC_ATTN_FX=0 prompt attention as flash attention on XMX (also on iGPUs with 16-lane XMX), ARC_GEMM_XW_WGX=1\n"
            "  the H8 GEMM's work-groups of 4 column sub-groups (2 / 8: that many), ARC_GEMM_XW_2D=0 its 2D block loads,\n"
            "  ARC_GEMV_DB=0 the GEMV's read-ahead (dense models), ARC_GEMV_WAVE=0 GEMV rows from the GPU's threads,\n"
            "  ARC_H8_GRF=256 / ARC_MOE_GRF=256 the H8 and MoE GEMM kernels' 256 registers (not 128), ARC_MOE_GEMM_PAD=0\n"
            "  padded experts' gate / up per tile (Flash Next), ARC_140T=0 the Alchemist cards' Arc 140T preset\n"
            "  ARC_GEMM_XW_GU=0 / 1: the FFN's gate and up GEMMs in one launch, the same results (on by default on discrete\n"
            "  cards with 16-lane XMX and on Alchemist (8 lanes, since 2026-11-23), opt-in on the 140V / B390 / 140T)\n"
            "  8-lane XMX: ARC_GEMM_XW_SHAPE=0 one tile for every prompt GEMM (default on Alchemist, one GPU: a tile per shape;\n"
            "  ARC_GEMM_XW_NC_E=2 turns it off),\n"
            "  ARC_GEMM_XW_DB=0 / 1 the 32-token tiles' reads one step ahead (default on Alchemist), the same results;\n"
            "                   =2 every tile, the next weights in place and the loop unrolled by 4 (opt-in)\n"
            "  ARC_GEMM_XW_BIG=0 / 1 the 32 x 32 tile through local memory where it fits (default on Alchemist),\n"
            "  ARC_GEMM_XW_EPI2=0 / 1 the epilogue a row at a time (default on Alchemist), ARC_GEMM_XW_SLM=1 / 2 / 8\n"
            "  local memory with that many token sub-groups (=1: 4) on every tile it fits; all the same results\n"
            "  ARC_GEMM_XW_WGL=0 / 1 the big tile's operands in its work-group order (default on Alchemist, the same results)\n"
            "  ARC_GEMM_XW_SLM_WG_GU=X,Y the gate / up GEMMs' big-tile sub-groups, the others ARC_GEMM_XW_SLM_WG's (Alchemist: 4,8)\n"
            "  ARC_GEMM_XW_BG=1 (with WGL) the big tile's B read from global memory, only A in local memory (opt-in, the same)\n"
            "  ARC_GEMM_XW_SPLIT=0 / 1 the big tile's stage barrier split in arrive and wait (default on Alchemist, with\n"
            "  ARC_GEMM_XW_SLM_WG 4,8 for every GEMM; =0: the barrier and the shapes of before; the same results)\n"
            "  ARC_H8_OUT=0 / 1 the DeltaNet / attention outputs rotated for H8 by the kernels that make them (default on\n"
            "  Alchemist, the same results)\n"
            "  MoE partly off the GPU: ARC_XCPU=0 / 1 the experts missing from the GPU cache computed by the CPU from the RAM\n"
            "  tier (KL-checked; on by default on discrete 16-lane cards: the B-series, a Data Center Max; off on Alchemist and\n"
            "  iGPUs; the same results run to run from the same cache state, its .lru file); in Flash Next's precise\n"
            "  mode one int8 pass on the CPU by default (faster), ARC_XCPU_Q2=1 both passes as on the GPU (exact)\n"
            "  ARC_XRAM_GB=N the RAM tier' size per GPU (0: none), ARC_XRAM_FILL=1 / 0 the expert cache's fill at start\n"
            "  into the RAM tier too or not (default: not, where the tier and the cache cannot hold every expert)\n");
    return help ? 0 : 1;
  }
  std::string model = argv[1], tokens_arg, kl_ref, ppl_file, ppl_dump, text_prompt, dump_ids;
  std::string ppl_kl_save, ppl_kl;
  int ppl_ctx = 512, ppl_chunks = 0, ppl_score_last = 0, ppl_score_step = 0;
  int n_gen = 16, chunk = 0, ctx = 0, bench_pp = 0, feed_step = 0, gpus = 1;
  EngineOptions eo;
  bool prof = false, show = false, prof_gen = false, kl_decode = false, slots_check = false, must_fit = false;
  int bench_batch = 0;
  std::string split_mode = "tensor";  // (--split tensor | layer: zero/mgpu/engine_tp.h or engine_pp.h)
  std::vector<double> split_shares;   // (--tensor-split a,b,...: each GPU's share of the layers with --split layer)
  bool ring_snap_test = false;  // Gemma: a snapshot's ring copy restored after the rings moved on, logits bitwise
  bool snap_test = false;       // (--snap-test: a snapshot restored after other tokens, logits bitwise)
  std::string dump_logits;  // every generation step's logits (f32), to compare two builds / settings bitwise
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* {  // (an option's value; missing at the end of the line: an error, not argv[argc])
      if (i + 1 >= argc) { zlog::eprintf("%s: missing value\n", a.c_str()); exit(1); }
      return argv[++i];
    };
    if (a == "--tokens") tokens_arg = next();
    else if (a == "--tokens-file") {
      std::ifstream f(next());
      std::stringstream ss;
      ss << f.rdbuf();
      tokens_arg = ss.str();
    }
    else if (a == "--text-file") {  // prompt text (special tokens parsed), tokenized with the model's vocabulary
      std::ifstream f(next(), std::ios::binary);
      std::stringstream ss;
      ss << f.rdbuf();
      text_prompt = ss.str();
    }
    else if (a == "--dump-ids") dump_ids = next();  // write the prompt's token ids (comma separated) and exit
    else if (a == "--must-fit") must_fit = true;  // exit (code 6) when the model and context exceed the GPU share
    else if (a == "--feed-step") feed_step = atoi(next());
    else if (a == "--n") n_gen = atoi(next());
    else if (a == "--chunk") chunk = atoi(next());
    else if (a == "--ctx") ctx = atoi(next());
    else if (a == "--bench-pp") bench_pp = atoi(next());
    else if (a == "--prof") prof = true;
    else if (a == "--show") show = true;
    else if (a == "--dump-logits") dump_logits = next();
    else if (a == "--prof-gen") prof_gen = true;
    else if (a == "--driver-reserve-gb" && i + 1 < argc) plat::set_env("ARC_DRIVER_RESERVE_GB", argv[++i]);  // (Engine::driver_slack)
    else if (a == "--act256") eo.act32_all = false;
    else if (a == "--attn-f32") eo.attn_int8 = false;
    else if (a == "--npu") {  // the NPU next to the GPU for prompt chunks (zero/npu): on, off, force
      const std::string v = next();
      eo.npu = v == "on" ? 1 : v == "off" ? 0 : v == "force" ? 2 : -2;
      if (eo.npu == -2) { zlog::eprintf("--npu: on, off or force\n"); return 1; }
    }
    else if (a == "--mtp") eo.mtp = true;
    else if (a == "--dflash") { eo.dflash = next(); eo.mtp = true; }  // DFlash2 drafter folder
    else if (a == "--dspark") eo.dspark = next();  // Inkling 2's DSpark drafter (dspark-*.gguf): greedy drafting
    else if (a == "--draft") {  // auto: DFlash2 and the MTP block, picked per step; mtp: the block only; dflash: DFlash2 only
      const std::string d = next();
      if (d == "mtp") eo.dflash.clear();
      else if (d == "dflash") eo.mtp_block = false;
      else if (d != "auto") { zlog::eprintf("--draft: auto, mtp or dflash\n"); return 1; }
    }
    else if (a == "--mtp-drafts") eo.mtp_drafts = atoi(next());
    else if (a == "--mtp-model") { eo.mtp_model = next(); eo.mtp = true; }  // (an MTP head / drafter file of its own)
    else if (a == "--mtp-min-p") eo.mtp_min_p = (float)atof(next());
    else if (a == "--copy-draft") eo.copy_draft = true;  // (R1: drafts copied from the context where it repeats)
    else if (a == "--kl-ref") kl_ref = next();
    else if (a == "--slots-check") slots_check = true;
    else if (a == "--bench-batch") bench_batch = atoi(next());
    else if (a == "--ring-snap-test") ring_snap_test = true;
    else if (a == "--snap-test") snap_test = true;
    else if (a == "--kl-decode") kl_decode = true;  // with --kl-ref of every position: feed the prompt token by token
    else if (a == "--perplexity") ppl_file = next();
    else if (a == "--ppl-dump") ppl_dump = next();
    else if (a == "--ppl-kl-save") ppl_kl_save = next();
    else if (a == "--ppl-score-last") ppl_score_last = atoi(next());
    else if (a == "--ppl-score-step") ppl_score_step = atoi(next());
    else if (a == "--ppl-kl") ppl_kl = next();
    else if (a == "--ppl-ctx") ppl_ctx = atoi(next());
    else if (a == "--ppl-chunks") ppl_chunks = atoi(next());
    else if (a == "--budget") eo.w8_budget_gb = atof(next()), eo.w8_budget_set = true;
    else if (a == "--gpus") gpus = std::max(1, atoi(next()));
    else if (a == "--split") {
      split_mode = next();
      if (split_mode != "tensor" && split_mode != "layer") { zlog::eprintf("--split: tensor or layer\n"); return 1; }
    } else if (a == "--tensor-split") {
      std::stringstream ss(next());
      std::string t;
      while (std::getline(ss, t, ',')) split_shares.push_back(atof(t.c_str()));
    }
    else if (!a.empty() && a[0] != '-' && text_prompt.empty()) text_prompt = a;  // PROMPT (special tokens parsed)
    else { zlog::eprintf("unknown option %s (--help: the options)\n", a.c_str()); return 1; }
  }
  std::vector<int> toks = parse_ids(tokens_arg);
  // (a model that expects BOS, Gemma, gets it first, like llama.cpp's tokenizer: without it the KL against ref_forward
  //  measured an off-distribution prompt, Gemma 26B 0.093 instead of 0.004. Every text prompt since 2026-11-23: it was
  //  added for --dump-ids only, so a run of the same text went without it (mejoudeh's 140T, #84: "prompt: 19529
  //  tokens" for 19,530 dumped ids). Ids given with --tokens stay as given, but --dump-ids, as before)
  auto bos_first = [&]() {
    GgufFile g(model);
    Vocab v(g);
    // (LFM2: the GGUF has no add_bos_token, llama.cpp adds its BOS in code)
    if ((g.get_int("tokenizer.ggml.add_bos_token", 0) != 0 || g.get_str("tokenizer.ggml.pre") == "lfm2") && v.bos >= 0 && (toks.empty() || toks[0] != v.bos))
      toks.insert(toks.begin(), v.bos);
  };
  if (!text_prompt.empty()) {  // the vocabulary is read from the GGUF before the engine loads
    GgufFile g(model);
    toks = Vocab(g).encode(text_prompt);
    bos_first();
  }
  if (!dump_ids.empty()) {
    bos_first();  // (ids from --tokens too, as before)
    FILE* fo = fopen(dump_ids.c_str(), "w");
    for (size_t i = 0; i < toks.size(); ++i) fprintf(fo, i ? ",%d" : "%d", toks[i]);
    fclose(fo);
    printf("%zu tokens -> %s\n", toks.size(), dump_ids.c_str());
    return 0;
  }
  // (audit 2026-11-12 #1: 0 divided the text into chunks by zero, a negative value gave 0 chunks; a chunk needs a
  //  scored position: j from ppl_ctx / 2 up to ppl_ctx - 1, so at least 3; 2 printed nan)
  if (!ppl_file.empty() && ppl_ctx < 3) throw std::runtime_error("--ppl-ctx must be at least 3");
  const int need = std::max((bench_pp > 0 ? bench_pp : (int)toks.size()) + n_gen + 16 + (bench_batch > 1 ? 1024 : 0),
                            ppl_file.empty() ? 0 : ppl_ctx + 16);
  if (!ppl_file.empty() && chunk == 0) chunk = std::min(ppl_ctx, 8192);  // (longer chunks: fed in pieces of 8192)
  eo.ctx = std::max(ctx, need);
  eo.chunk = chunk;
  // (ARC_KL_PACK=OUT [ARC_KL_PACK_K=128] with --kl-ref FULL: the full reference written as the compact one, read_kl_ref,
  //  the vocabulary's size from the model's token_embd; nothing run on the GPU. 512: Flash Next's prose KL 0.00894 against the
  //  full reference's 0.00914 (128: 0.00860), 1 MB a prompt)
  if (const char* pack = getenv("ARC_KL_PACK")) {
    const GgufFile g(model);
    const GgufTensor* te = g.find("token_embd.weight");
    if (!te || kl_ref.empty()) { zlog::eprintf("ARC_KL_PACK: needs --kl-ref and a model with token_embd\n"); return 1; }
    const size_t V = (size_t)te->dims[1], K = getenv("ARC_KL_PACK_K") ? std::clamp(atoi(getenv("ARC_KL_PACK_K")), 2, 4096) : 512;
    const std::vector<float> ref = read_kl_ref(kl_ref, V);
    if (ref.empty() || ref.size() % V) { zlog::eprintf("ARC_KL_PACK: %s is not a reference of %zu-token rows\n", kl_ref.c_str(), V); return 1; }
    FILE* fo = fopen(pack, "wb");
    if (!fo) { zlog::eprintf("cannot write %s\n", pack); return 1; }
    const uint32_t h[4] = {1u, (uint32_t)V, (uint32_t)(ref.size() / V), (uint32_t)K};
    fwrite("IZKL", 1, 4, fo);
    fwrite(h, 4, 4, fo);
    std::vector<int32_t> idx(V);
    double worst_tail = 0;
    for (size_t r = 0; r < ref.size() / V; ++r) {
      const float* row = ref.data() + r * V;
      for (size_t i = 0; i < V; ++i) idx[i] = (int32_t)i;
      std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int32_t a, int32_t b) { return row[a] > row[b]; });
      std::vector<float> lp(K);
      double top = 0;
      for (size_t k = 0; k < K; ++k) lp[k] = row[idx[k]], top += std::exp((double)lp[k]);
      const double rest = std::max(0.0, 1.0 - top);
      const float tail = rest > 0 ? (float)std::log(rest) : -1e38f;
      worst_tail = std::max(worst_tail, rest);
      fwrite(idx.data(), 4, K, fo);
      fwrite(lp.data(), 4, K, fo);
      fwrite(&tail, 4, 1, fo);
    }
    fclose(fo);
    printf("ARC_KL_PACK: %zu positions, top %zu each, the most probability left outside them %.4f\n", ref.size() / V, K, worst_tail);
    return 0;
  }
  warn_gpu_neighbours();
  if (gpus > 1 && split_mode == "layer") {
    if (eo.mtp) { zlog::eprintf("--split layer: no --mtp yet (drafting over the stages: QUEUE)\n"); return 1; }
    return run_pp(model, eo, gpus, split_shares, toks, n_gen, bench_pp, dump_logits, show, prof || prof_gen, prof_gen);
  }
  if (gpus > 1) {
    if (bench_batch > 1) eo.slots = std::min(bench_batch, Engine::kMaxSlots);  // (a split model's slots: made at load)
    return run_tp(model, eo, gpus, toks, n_gen, bench_pp, dump_logits, show, kl_ref, bench_batch, prof || prof_gen, prof_gen);
  }
  // (--bench-batch on one GPU: the engine sized for its slots at load, as the split model's above, so forward_batch's
  //  rows are not capped at 1 + drafts; add_slots below then finds them made. hibagus, #48, 2026-11-22)
  if (bench_batch > 1) eo.slots = std::min(bench_batch, Engine::kMaxSlots);
  Engine eng(model, eo);
  // (the logits that are tokens: Inkling 2's head has rows past its vocabulary, -inf; ref_forward writes the vocabulary)
  const size_t VK = eng.hp.inkling ? (size_t)eng.ink_vocab() : (size_t)eng.hp.n_vocab;
  // (every position's logits are read after the prompt: no 8-token passes, see Engine::keep_rows; a reference of
  //  the last position only, or --kl-decode, keeps the passes the server takes)
  if (!ppl_file.empty()) eng.keep_rows = true;
  if (!kl_ref.empty() && !kl_decode) {
    std::error_code ec;
    const uintmax_t rb = std::filesystem::file_size(kl_ref, ec);
    eng.keep_rows = !ec && rb != (uintmax_t)VK * 4;
  }
  Vocab vocab(eng.gguf());
  printf("loaded: %d layers, n_embd %d, vocab %d | GPU buffers %.2f GiB | W8 %s | chunk %d | ctx %d | load %.0f ms\n",
         eng.hp.n_layer, eng.hp.n_embd, eng.hp.n_vocab, eng.gpu_bytes / 1073741824.0,
         eng.w8_resident ? "resident" : "just-in-time", eng.max_chunk, eng.ctx, eng.load_ms);
  {  // (more GPU memory than the share less the driver's part: Windows pages GPU memory and everything slows down many
     //  times over, Qwen3.8 27B with a 131K context on the laptop's 16.5 GiB share: 16.5 GiB used, prompt 8.6 instead of
     //  ~120 tok/s. The server's --ctx auto stays inside; runz takes the context it is given)
    const double GiB = 1073741824.0, share = (double)eng.device_mem();
    fflush(stdout);
    if (getenv("ARC_FIT_PARTS")) {  // (diagnostics: the memory count the fit check reads, by part)
      const auto m = eng.mem_parts(eng.ctx);
      zlog::eprintf("inferzero: memory: GPU %.3f GiB = loaded %.3f (weights %.3f, MTP %.3f, buffers %.3f, context %.3f at %d) + "
                    "after the load %.3f; ctx_bytes / %d tokens\n",
                    eng.gpu_bytes / GiB, eng.loaded_bytes() / GiB, m.weights / GiB, m.mtp / GiB, m.buffers / GiB,
                    m.context / GiB, eng.ctx, ((double)eng.gpu_bytes - (double)eng.loaded_bytes()) / GiB, eng.ctx);
    }
    if (!eng.fits_share()) {
      const int fit = eng.max_fitting_ctx();
      zlog::eprintf("inferzero: does not fit: %.2f GiB of GPU memory for a context of %d%s%s, over the %.2f GiB share "
                      "less %.1f for the driver; expect paging and far lower speeds (%s)\n",
              eng.gpu_bytes / GiB, eng.ctx, eo.mtp ? " with drafting" : "",
              eng.host_weight_bytes() ? (", " + std::to_string((int)(eng.host_weight_bytes() / 1048576)) + " MiB of the weights in host memory too").c_str() : "",
              share / GiB, Engine::driver_slack() / GiB,
              fit > 0 ? ("--ctx " + std::to_string(fit) + " fits with every weight on the GPU").c_str() : "no context fits: a smaller model or fewer weights on the GPU");
      if (must_fit) return 6;  // (code 6, not 3: 3 is a GPU hang since 2026-11-23, docs/gpu-failsafe.md. --must-fit: bench\website_bench.ps1 records the size as not fitting)
    }
  }

  // --ring-snap-test (Gemma): the prompt to a, a snapshot, 6,000 more tokens (the sliding-window rings move on, so
  // rewind() to a fails), the snapshot restored, 600 tokens from a: their logits against the same tokens fed after a
  // fresh start (the same pieces), bit for bit
  if (ring_snap_test) {
    // (a past every Gemma 4 window, 512-1024: below it the copy covers position 0 on, and a rewind there is right;
    //  mid past the rings' reach: the copy taken on the way)
    const int a = 2000, mid = eng.ring_cap() + 1000, tail = 600;
    if (!eng.hp.swa_rings || (int)toks.size() < a + mid + tail) {
      zlog::eprintf("--ring-snap-test: a Gemma 4 model and a prompt of at least %d tokens\n", a + mid + tail);
      return 1;
    }
    auto feed = [&](int from, int to) {
      std::vector<float> l;
      for (int p = from; p < to;) {
        const int n = std::min(eng.max_chunk, to - p);
        l = eng.forward(toks.data() + p, n);
        p += n;
      }
      return l;
    };
    eng.reset();
    feed(0, a);
    const auto t0 = std::chrono::steady_clock::now();
    eng.snapshot(0);
    const double snap_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const size_t before = eng.snapshot_host_bytes(0);  // (none yet: the copy waits until the rings reach it)
    const auto tm = std::chrono::steady_clock::now();
    feed(a, a + mid);
    const double mid_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tm).count();
    const bool rew = eng.rewind(a);
    const auto t1 = std::chrono::steady_clock::now();
    const bool ok = !rew && eng.restore(0) && eng.pos == a;
    const double rest_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    const std::vector<float> got = feed(a, a + tail);
    // (after the restore the rings hold only the window before a: a rewind to just before a must be refused, its
    //  slots still hold the 6,000 tokens fed since; audit 2026-10-31)
    const bool below = eng.rewind(a - 16);
    const size_t host = eng.snapshot_host_bytes(0);
    eng.reset();
    feed(0, a);
    const std::vector<float> want = feed(a, a + tail);
    const bool same = ok && got.size() == want.size() && memcmp(got.data(), want.data(), got.size() * 4) == 0;
    printf("ring snapshot at %d: rewind after %d more %s, restore %s; logits of %d tokens after it %s; a rewind below it "
           "%s (snapshot %.1f ms with %zu bytes copied, the copy on the way: %.1f MB host memory; the %d tokens %.0f ms; "
           "restore %.1f ms)\n", a, mid, rew ? "worked (test not valid)" : "refused", ok ? "ok" : "FAILED", tail,
           same ? "bit-identical" : "DIFFER", below ? "ACCEPTED (stale slots)" : "refused", snap_ms, before,
           host / 1048576.0, mid, mid_ms, rest_ms);
    return same && !below && before == 0 && host > 0 ? 0 : 1;
  }
  // --snap-test (DeepSeek-V4.1, or any model with snapshots; Gemma: the device snapshot only, as the server): the prompt
  // to an odd position a, a snapshot (the device one, then the host one), an even number of other tokens fed past it,
  // at least a window's worth (they overwrite every window-ring slot, the pending compressor rows, the token history),
  // the snapshot restored, then the prompt's next tokens: their logits against the same pieces fed after a fresh
  // start, bit for bit
  if (snap_test) {
    const int n = (int)toks.size(), a = std::max(1, n / 3) | 1;
    const int mid = std::max(n / 3, eng.ds_window() + 2) / 2 * 2, tail = n - a - mid;
    if (tail < 4) { zlog::eprintf("--snap-test: a prompt of at least %d tokens\n", a + mid + 4); return 1; }
    auto feed = [&](const int* t, int n) {
      std::vector<float> l;
      for (int p = 0; p < n;) {
        const int k = std::min(eng.max_chunk, n - p);
        l = eng.forward(t + p, k);
        p += k;
      }
      return l;
    };
    std::vector<int> other(toks.rbegin(), toks.rbegin() + mid);  // (other tokens: the prompt's end reversed)
    eng.reset();
    feed(toks.data(), a);
    const std::vector<float> want = feed(toks.data() + a, tail);
    bool all = true;
    for (int host = 0; host < (eng.hp.swa_rings ? 1 : 2); ++host) {
      eng.reset();
      feed(toks.data(), a);
      if (host) eng.snapshot_host(); else eng.snapshot(0);
      feed(other.data(), mid);
      const bool ok = (host ? eng.restore_host() : eng.restore(0)) && eng.pos == a;
      const std::vector<float> got = ok ? feed(toks.data() + a, tail) : std::vector<float>();
      const bool same = ok && got.size() == want.size() && memcmp(got.data(), want.data(), got.size() * 4) == 0;
      printf("%s snapshot at %d, %d other tokens, restore %s; the next %d tokens' logits %s\n", host ? "host" : "device", a,
             mid, ok ? "ok" : "FAILED", tail, same ? "bit-identical" : "DIFFER");
      all = all && same;
      if (host) eng.drop_host_snapshot(); else eng.drop_snapshot(0);
    }
    // Flash Next verification embeds directly: forward() after restore would hide stale host PLE IDs.
    // Fixed draft IDs keep this independent of the drafter's predictions; every vocabulary logit is compared.
    // Both snapshots: the device one (restore(slot)) had the same gap as the host one.
    if (eng.hp.ple_layer >= 0 && eng.has_mtp) {
      const std::vector<int> drafts{toks[a + 1]};
      eng.reset();
      feed(toks.data(), a);
      const std::vector<float> cold = eng.verify(toks[a], drafts);
      eng.verify_done(0);
      for (int host = 0; host < 2; ++host) {
        eng.reset();
        feed(toks.data(), a);
        if (host) eng.snapshot_host(); else eng.snapshot(0);
        feed(other.data(), mid);
        const bool ok = (host ? eng.restore_host() : eng.restore(0)) && eng.pos == a;
        const std::vector<float> back = ok ? eng.verify(toks[a], drafts) : std::vector<float>();
        const bool same = ok && back.size() == cold.size() && memcmp(back.data(), cold.data(), back.size() * 4) == 0;
        if (ok) eng.verify_done(0);
        if (host) eng.drop_host_snapshot(); else eng.drop_snapshot(0);
        printf("%s snapshot followed directly by verify: %s\n", host ? "host" : "device", same ? "bit-identical" : "DIFFER");
        all = all && same;
      }
    }
    // a saved conversation: the prompt to a and a snapshot there, saved to a file; the engine's caches then overwritten
    // by other tokens (reset() leaves the caches' contents), reset, the file loaded: the next tokens, then the loaded
    // snapshot restored after other tokens and the next tokens again. A file cut in half must be refused. (Gemma: a
    // loaded snapshot has no ring copy, so not for it; a model whose state cannot be saved: skipped)
    if (!eng.hp.swa_rings) {
      eng.reset();
      feed(toks.data(), a);
      eng.snapshot(0);
      const std::string tmp = (std::filesystem::temp_directory_path() / "runz_snap_test.state").string();
      FILE* fo = fopen(tmp.c_str(), "wb");
      const bool saved = fo && eng.save_state(fo, a);
      if (fo) fclose(fo);
      if (!saved) {
        printf("saved conversation: not supported here, skipped\n");
        std::remove(tmp.c_str());
        return all ? 0 : 1;
      }
      std::vector<int> scramble(toks.rbegin(), toks.rbegin() + std::min(n, a + mid));
      eng.reset();
      feed(scramble.data(), (int)scramble.size());
      eng.reset();
      eng.drop_snapshot(0);
      bool cut_refused = false;
      {  // (the first half of the file: refused, the engine left at position 0)
        std::vector<char> all_bytes;
        if (FILE* fr = fopen(tmp.c_str(), "rb")) {
          char buf[65536];
          for (size_t k; (k = fread(buf, 1, sizeof buf, fr)) > 0;) all_bytes.insert(all_bytes.end(), buf, buf + k);
          fclose(fr);
        }
        const std::string half = tmp + ".half";
        if (FILE* fh = fopen(half.c_str(), "wb")) {
          fwrite(all_bytes.data(), 1, all_bytes.size() / 2, fh);
          fclose(fh);
        }
        FILE* fh = fopen(half.c_str(), "rb");
        cut_refused = fh && !eng.load_state(fh) && eng.pos == 0;
        if (fh) fclose(fh);
        std::remove(half.c_str());
      }
      FILE* fi = fopen(tmp.c_str(), "rb");
      const bool loaded = fi && eng.load_state(fi) && eng.pos == a;
      if (fi) fclose(fi);
      std::remove(tmp.c_str());
      const std::vector<float> g1 = loaded ? feed(toks.data() + a, tail) : std::vector<float>();
      bool ok2 = false;
      std::vector<float> g2;
      if (loaded) {
        feed(other.data(), mid);
        ok2 = eng.restore(0) && eng.pos == a;
        if (ok2) g2 = feed(toks.data() + a, tail);
      }
      auto eq = [&](const std::vector<float>& g) { return g.size() == want.size() && memcmp(g.data(), want.data(), g.size() * 4) == 0; };
      const bool s1 = loaded && eq(g1), s2 = ok2 && eq(g2);
      printf("saved conversation at %d: half the file %s; load %s; the next %d tokens %s; its snapshot restored after %d "
             "other tokens %s, the next tokens %s\n", a, cut_refused ? "refused" : "ACCEPTED", loaded ? "ok" : "FAILED",
             tail, s1 ? "bit-identical" : "DIFFER", mid, ok2 ? "ok" : "FAILED", s2 ? "bit-identical" : "DIFFER");
      all = all && s1 && s2 && cut_refused;
    }
    return all ? 0 : 1;
  }
  if (bench_batch > 1) {  // batched decoding (forward_batch) over 1..N slots holding the prompt, against one slot alone
    const int NS = bench_batch;
    if ((int)toks.size() < NS + 8) { zlog::eprintf("--bench-batch %d: the prompt needs at least %d tokens\n", NS, NS + 8); return 1; }
    eng.add_slots(NS);
    if (!eng.batch_ok()) { zlog::eprintf("--bench-batch: no batched decoding for this model\n"); return 1; }
    {  // (2026-11-19) the check the split model's form has: K greedy steps of every slot in one pass against each
       // slot alone, logits compared bit for bit; where they differ, the largest difference and the top tokens
      const int K = 8;
      std::vector<std::vector<std::vector<float>>> alone(NS);
      std::vector<int> sl(NS), tk(NS);
      auto argmax = [](const std::vector<float>& l, size_t off, size_t V) {
        return (int)(std::max_element(l.begin() + off, l.begin() + off + V) - (l.begin() + off));
      };
      // slot s: the prompt less its last s tokens (each slot its own conversation)
      auto feed = [&](int s) -> const std::vector<float>& {
        eng.bind(s);
        eng.reset();
        return eng.forward(toks.data(), (int)toks.size() - s);
      };
      size_t V = 0;
      for (int s = 0; s < NS; ++s) {
        const std::vector<float>& l0 = feed(s);
        V = l0.size();
        int t = argmax(l0, 0, V);
        for (int k = 0; k < K; ++k) {
          const std::vector<float>& l = eng.forward(&t, 1);
          alone[s].push_back(l);
          t = argmax(l, 0, V);
        }
      }
      for (int s = 0; s < NS; ++s) {
        sl[s] = s;
        tk[s] = argmax(feed(s), 0, V);
      }
      int diff_rows = 0, top_same = 0;
      double max_d = 0;
      for (int k = 0; k < K; ++k) {
        const std::vector<float>& l = eng.forward_batch(sl, tk);
        for (int s = 0; s < NS; ++s) {
          const float* b = l.data() + (size_t)s * V;
          const std::vector<float>& a = alone[s][k];
          if (memcmp(a.data(), b, V * 4)) ++diff_rows;
          for (size_t i = 0; i < V; ++i) max_d = std::max(max_d, (double)std::fabs(a[i] - b[i]));
          tk[s] = argmax(l, (size_t)s * V, V);
          top_same += tk[s] == argmax(a, 0, V);
        }
      }
      if (!diff_rows)
        printf("batched vs each slot alone, %d slots x %d steps: logits bit-identical\n", NS, K);
      else
        printf("batched vs each slot alone, %d slots x %d steps: logits DIFFER in %d of %d rows (largest difference "
               "%.4f), top token the same in %d / %d\n", NS, K, diff_rows, NS * K, max_d, top_same, NS * K);
    }
    for (int s = 0; s < NS; ++s) {
      eng.bind(s);
      eng.reset();
      eng.forward(toks.data(), (int)toks.size());
    }
    eng.bind(0);
    const int reps = 16;
    for (int B = 1; B <= std::min(NS, eng.batch_rows()); ++B) {
      double best = 1e30, one = 1e30;
      std::vector<int> sl, tk;
      for (int b = 0; b < B; ++b) { sl.push_back(b); tk.push_back(toks[b]); }
      for (int round = 0; round < 3; ++round) {
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) eng.forward_batch(sl, tk);
        best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps);
        t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) eng.forward(&tk[0], 1);
        one = std::min(one, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps);
      }
      printf("B %d: %.1f ms per batched step (%.1f tok/s together) | one slot alone %.1f ms (%.1f tok/s): %.2fx\n", B, best,
             1000.0 * B / best, one, 1000.0 / one, one * B / best);
      if (prof_gen) {  // (--prof-gen: GPU time per kernel of a step, alone and batched, over `reps` steps each)
        std::map<std::string, double> a, b;
        auto run_prof = [&](std::map<std::string, double>& out, bool batched) {
          eng.prof_ms.clear();
          eng.prof_n.clear();
          eng.profile = true;
          for (int r = 0; r < reps; ++r) {
            if (batched) eng.forward_batch(sl, tk);
            else eng.forward(&tk[0], 1);
          }
          eng.profile = false;
          for (auto& [n, ms] : eng.prof_ms) out[n] = ms / reps;
        };
        run_prof(a, false);
        run_prof(b, true);
        std::vector<std::pair<double, std::string>> rows;
        double ta = 0, tb = 0;
        for (auto& [n, ms] : b) rows.push_back({ms, n});
        for (auto& [n, ms] : a) {
          ta += ms;
          if (!b.count(n)) rows.push_back({0, n});
        }
        for (auto& [n, ms] : b) tb += ms;
        std::sort(rows.rbegin(), rows.rend());
        printf("  GPU ms per step by kernel: alone %.1f, batched %.1f (B x alone %.1f)\n", ta, tb, B * ta);
        for (auto& [ms, n] : rows)
          if (ms > 0.2 || a[n] > 0.2) printf("    %-20s alone %7.2f  batched %7.2f  (%.2fx)\n", n.c_str(), a[n], ms, a[n] > 0 ? ms / a[n] : 0.0);
      }
    }
    if (eng.has_mtp)  // with drafts (as many as B slots' rows fit one pass; none accepted, the same work every step)
      for (int B = 2; B <= NS && 2 * B <= eng.batch_rows(); ++B) {
        const int nd = std::min(eng.opt.mtp_drafts, eng.batch_rows() / B - 1);
        double alt = 1e30, bat = 1e30;
        for (int round = 0; round < 3; ++round) {
          auto t0 = std::chrono::steady_clock::now();
          for (int r = 0; r < reps / 2; ++r)
            for (int b = 0; b < B; ++b) {  // each slot alone, in turn
              eng.bind(b);
              eng.verify(toks[b], eng.mtp_draft(toks[b], nd));
              eng.verify_done(0);
            }
          alt = std::min(alt, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / (reps / 2));
          t0 = std::chrono::steady_clock::now();
          for (int r = 0; r < reps / 2; ++r) {  // the drafts of every slot, one verification pass, each settled
            std::vector<int> sl;
            std::vector<std::vector<int>> rows;
            for (int b = 0; b < B; ++b) {
              eng.bind(b);
              std::vector<int> row{toks[b]};
              for (int d : eng.mtp_draft(toks[b], nd)) row.push_back(d);
              sl.push_back(b);
              rows.push_back(row);
            }
            eng.verify_batch(sl, rows);
            for (int b = 0; b < B; ++b) {
              eng.bind(b);
              eng.verify_done(0);
            }
          }
          bat = std::min(bat, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / (reps / 2));
        }
        eng.bind(0);
        printf("B %d with %d drafts each: %.1f ms per step of every slot in turn, %.1f ms batched: %.2fx\n", B, nd, alt, bat,
               alt / bat);
      }
    return 0;
  }
  if (slots_check) {
    // conversation A = the first half of the prompt, B = the second half; each prompt fed in two calls, then n_gen
    // greedy tokens one by one, then (when the engine drafts) n_gen more with drafts. Alone: one after the other in
    // slot 0; interleaved: A in slot 0, B in slot 1, every call alternating; batched (dense / MoE Qwen): the one-by-one
    // tokens of both slots in one pass (forward_batch), then the drafted ones alternating. The FNV hash of every
    // logits row read is compared.
    if (toks.size() < 8) { zlog::eprintf("--slots-check needs a prompt of at least 8 tokens\n"); return 1; }
    eng.add_slots(2);
    {  // (--ctx auto counts with slot_bytes(): it must not count less than a slot takes)
      const bool ok = eng.slot_real_bytes_ <= eng.slot_est_bytes_ + eng.slot_est_bytes_ / 100;
      printf("slot bytes: %zu real, %zu counted: %s\n", eng.slot_real_bytes_, eng.slot_est_bytes_, ok ? "OK" : "MISMATCH");
      if (!ok) return 1;
    }
    const int V = eng.hp.n_vocab, n_all = eng.has_mtp ? 2 * n_gen : n_gen;
    // (drafts per step: as many as two slots' rows fit one batched pass, in every run)
    const int nd = std::max(1, std::min(eng.opt.mtp_drafts, eng.batch_rows() / 2 - 1));
    auto fnv = [](const float* p, size_t n) {
      uint64_t h = 1469598103934665603ull;
      const uint8_t* b = (const uint8_t*)p;
      for (size_t i = 0; i < n * 4; ++i) h = (h ^ b[i]) * 1099511628211ull;
      return h;
    };
    auto argmax = [&](const float* p) { return (int)(std::max_element(p, p + V) - p); };
    struct Conv {
      std::vector<int> P, out;
      std::vector<uint64_t> h;
      std::vector<float> lg;
      int stage = 0;
    };
    // one call of a conversation: prompt part 1, prompt part 2, a token, or a verification pass with drafts
    auto step = [&](Conv& c) {
      const int half = (int)c.P.size() / 2;
      if (c.stage == 0) {
        eng.reset();
        eng.reset_drafter_choice();
        c.lg = eng.forward(c.P.data(), half);
        c.h.push_back(fnv(c.lg.data(), V));
      } else if (c.stage == 1) {
        c.lg = eng.forward(c.P.data() + half, (int)c.P.size() - half);
        c.h.push_back(fnv(c.lg.data(), V));
      } else {
        const int t = argmax(c.lg.data());
        c.out.push_back(t);
        if ((int)c.out.size() > n_gen) {
          const std::vector<int> d = eng.mtp_draft(t, nd);
          const std::vector<float>& lv = eng.verify(t, d);
          size_t acc = 0;
          while (acc < d.size() && argmax(&lv[acc * V]) == d[acc]) ++acc;
          for (size_t k = 0; k <= acc; ++k) c.h.push_back(fnv(&lv[k * V], V));
          eng.verify_done((int)acc);
          c.out.insert(c.out.end(), d.begin(), d.begin() + acc);
          c.lg.assign(lv.begin() + acc * V, lv.begin() + (acc + 1) * V);
        } else {
          c.lg = eng.forward(&t, 1);
          c.h.push_back(fnv(c.lg.data(), V));
        }
      }
      ++c.stage;
    };
    auto done = [&](const Conv& c) { return c.stage >= 2 && (int)c.out.size() >= n_all; };
    const int mid = (int)toks.size() / 2;
    Conv a1, b1, a2, b2, a3, b3;
    a1.P = a2.P = a3.P = std::vector<int>(toks.begin(), toks.begin() + mid);
    b1.P = b2.P = b3.P = std::vector<int>(toks.begin() + mid, toks.end());
    for (Conv* c : {&a1, &b1})
      while (!done(*c)) step(*c);
    while (!done(a2) || !done(b2)) {
      if (!done(a2)) { eng.bind(0); step(a2); }
      if (!done(b2)) { eng.bind(1); step(b2); }
    }
    const bool batched = eng.batch_ok();
    if (batched) {
      for (int k = 0; k < 2; ++k) {
        eng.bind(0); step(a3);
        eng.bind(1); step(b3);
      }
      while ((int)a3.out.size() < n_gen || (int)b3.out.size() < n_gen) {  // (both the same number of plain tokens)
        const int ta = argmax(a3.lg.data()), tb = argmax(b3.lg.data());
        const std::vector<float>& lv = eng.forward_batch({0, 1}, {ta, tb});
        for (int r = 0; r < 2; ++r) {
          Conv& c = r ? b3 : a3;
          c.out.push_back(r ? tb : ta);
          c.lg.assign(lv.begin() + (size_t)r * V, lv.begin() + (size_t)(r + 1) * V);
          c.h.push_back(fnv(c.lg.data(), V));
          ++c.stage;
        }
      }
      // then the drafted steps of both slots verified in one pass (verify_batch), each settled in its slot
      while (eng.batch_verify_ok() && !done(a3) && !done(b3)) {
        std::vector<std::vector<int>> rows;
        for (int r = 0; r < 2; ++r) {
          Conv& c = r ? b3 : a3;
          const int t = argmax(c.lg.data());
          c.out.push_back(t);
          eng.bind(r);
          std::vector<int> row{t};
          for (int d : eng.mtp_draft(t, nd)) row.push_back(d);
          rows.push_back(row);
        }
        const std::vector<float> lv = eng.verify_batch({0, 1}, rows);
        size_t r0 = 0;
        for (int r = 0; r < 2; ++r) {
          Conv& c = r ? b3 : a3;
          const std::vector<int> d(rows[r].begin() + 1, rows[r].end());
          size_t acc = 0;
          while (acc < d.size() && argmax(&lv[(r0 + acc) * V]) == d[acc]) ++acc;
          for (size_t k = 0; k <= acc; ++k) c.h.push_back(fnv(&lv[(r0 + k) * V], V));
          eng.bind(r);
          eng.verify_done((int)acc);
          c.out.insert(c.out.end(), d.begin(), d.begin() + acc);
          c.lg.assign(lv.begin() + (r0 + acc) * V, lv.begin() + (r0 + acc + 1) * V);
          ++c.stage;
          r0 += rows[r].size();
        }
      }
      while (!done(a3) || !done(b3)) {
        if (!done(a3)) { eng.bind(0); step(a3); }
        if (!done(b3)) { eng.bind(1); step(b3); }
      }
    }
    eng.bind(0);
    bool ok = true;
    auto cmp = [&](const Conv* x, const Conv* y, const char* name) {
      size_t same = 0;
      while (same < x->h.size() && same < y->h.size() && x->h[same] == y->h[same]) ++same;
      const bool eq = same == x->h.size() && same == y->h.size() && x->out == y->out;
      ok = ok && eq;
      printf("%s: %zu prompt tokens, %zu generated, %zu logits rows: %s", name, x->P.size(), x->out.size(), x->h.size(),
             eq ? "bit-identical\n" : "DIFFERENT");
      if (!eq) printf(" (first difference at row %zu of %zu / %zu)\n", same, x->h.size(), y->h.size());
    };
    cmp(&a1, &a2, "conversation A interleaved");
    cmp(&b1, &b2, "conversation B interleaved");
    if (batched) {
      cmp(&a1, &a3, "conversation A batched");
      cmp(&b1, &b3, "conversation B batched");
      // misuse: refused before anything changes (and the slots work afterwards)
      auto refused = [&](const char* what, auto&& call) {
        bool threw = false;
        try { call(); } catch (const std::exception&) { threw = true; }
        ok = ok && threw;
        printf("refused %s: %s\n", what, threw ? "yes" : "NO");
      };
      const int t0 = argmax(a3.lg.data());
      refused("a slot out of range", [&] { eng.forward_batch({0, 2}, {t0, t0}); });
      refused("a negative slot", [&] { eng.forward_batch({-1, 0}, {t0, t0}); });
      refused("a slot twice", [&] { eng.forward_batch({0, 0}, {t0, t0}); });
      refused("bind out of range", [&] { eng.bind(2); });
      if (eng.batch_verify_ok()) {  // a verification pass not settled: no batch over that slot until verify_done
        eng.bind(0);
        std::vector<int> row{t0};
        for (int d : eng.mtp_draft(t0, nd)) row.push_back(d);
        eng.bind(1);
        const int tb = argmax(b3.lg.data());
        std::vector<int> rowb{tb};
        for (int d : eng.mtp_draft(tb, nd)) rowb.push_back(d);
        eng.verify_batch({0, 1}, {row, rowb});
        refused("a batch over unsettled drafts", [&] { eng.forward_batch({0, 1}, {t0, tb}); });
        for (int r = 0; r < 2; ++r) {
          eng.bind(r);
          eng.verify_done(0);
        }
        bool works = true;
        try { eng.forward_batch({0, 1}, {t0, tb}); } catch (const std::exception&) { works = false; }
        ok = ok && works;
        printf("batch after the drafts were settled: %s\n", works ? "ok" : "FAILED");
      }
    }
    printf("slots check: %s\n", ok ? "passed" : "FAILED");
    return ok ? 0 : 2;
  }

  if (!ppl_file.empty()) {
    // Perplexity like llama-perplexity: split the tokenized text into ppl_ctx chunks, evaluate each from an
    // empty state and score the second half (logits at j predict token j+1, j in [ctx/2, ctx-1)).
    std::ifstream fin(ppl_file, std::ios::binary);
    std::stringstream ss;
    ss << fin.rdbuf();
    const std::vector<int> all = vocab.encode(ss.str(), false);
    const bool add_bos = eng.gguf().get_int("tokenizer.ggml.add_bos_token", 0) != 0 || eng.gguf().get_str("tokenizer.ggml.pre") == "lfm2";
    int nchunks = (int)(all.size() / ppl_ctx);
    if (ppl_chunks > 0) nchunks = std::min(nchunks, ppl_chunks);
    if (nchunks <= 0)  // (a text shorter than one chunk: nothing to score, the result would be nan)
      throw std::runtime_error("--perplexity: the text has " + std::to_string(all.size()) + " tokens, fewer than one chunk of --ppl-ctx " + std::to_string(ppl_ctx));
    if (!ppl_dump.empty()) {  // write the chunks' token ids (one comma-separated line per chunk) for ref_forward
      FILE* fo = fopen(ppl_dump.c_str(), "w");
      if (!fo) throw std::runtime_error("cannot write " + ppl_dump);
      for (int ch = 0; ch < nchunks; ++ch) {
        for (int j = 0; j < ppl_ctx; ++j) fprintf(fo, j ? ",%d" : "%d", all[(size_t)ch * ppl_ctx + j]);
        fprintf(fo, "\n");
      }
      fclose(fo);
      return 0;
    }
    double nll = 0;
    long count = 0;
    // (--ppl-kl-save F / --ppl-kl F: per scored token the base run's top 64 probabilities, and KL(base || this)
    //  over those 64 plus the rest of the mass as one bucket; also how often the top token is the same)
    constexpr int KT = 64;
    FILE* kls = ppl_kl_save.empty() ? nullptr : fopen(ppl_kl_save.c_str(), "wb");
    FILE* klr = ppl_kl.empty() ? nullptr : fopen(ppl_kl.c_str(), "rb");
    double kl_sum = 0, kl_max = 0;
    long same_top = 0;
    std::vector<float> lpv;
    std::vector<int> idx;
    auto t0 = std::chrono::steady_clock::now();
    for (int ch = 0; ch < nchunks; ++ch) {
      // like llama-perplexity: a model that expects BOS gets it as the first token of every chunk
      std::vector<int> chunk_toks(all.begin() + (size_t)ch * ppl_ctx, all.begin() + (size_t)(ch + 1) * ppl_ctx);
      if (add_bos && vocab.bos >= 0) chunk_toks[0] = vocab.bos;
      const int* tk = chunk_toks.data();
      eng.reset();
      // (fed in pieces of the chunk size: logits_at reads rows of the last piece only. --ppl-score-last N: only the
      //  last N positions are scored; --ppl-score-step S: those are fed S tokens at a time (S <= 8: the generation
      //  kernels), the rest in prompt chunks)
      const int piece = std::max(1, chunk);
      const int from = ppl_score_last > 0 ? std::max(1, ppl_ctx - 1 - ppl_score_last) : ppl_ctx / 2;
      for (int o = 0; o < ppl_ctx;) {
        const int n = o < from ? std::min(piece, from - o) : std::min(ppl_score_step > 0 ? ppl_score_step : piece, ppl_ctx - o);
        eng.forward(tk + o, n);
        for (int j = std::max(o, from); j < std::min(o + n, ppl_ctx - 1); ++j) {
          const auto& lg = eng.logits_at(j - o);
          const double mx = *std::max_element(lg.begin(), lg.end());
          double z = 0;
          for (float l : lg) z += std::exp((double)l - mx);
          nll += -((double)lg[tk[j + 1]] - mx - std::log(z));
          ++count;
          if (kls || klr) {
            const double lz = mx + std::log(z);
            if (kls) {
              idx.resize(lg.size());
              for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
              std::partial_sort(idx.begin(), idx.begin() + KT, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
              for (int k = 0; k < KT; ++k) {
                const float lp = (float)(lg[idx[k]] - lz);
                fwrite(&idx[k], 4, 1, kls);
                fwrite(&lp, 4, 1, kls);
              }
            } else {
              int id[KT];
              float bl[KT];
              for (int k = 0; k < KT; ++k) {
                if (fread(&id[k], 4, 1, klr) != 1 || fread(&bl[k], 4, 1, klr) != 1) { zlog::eprintf("--ppl-kl: file too short\n"); return 1; }
              }
              double kl = 0, pb_top = 0, pq_top = 0;
              for (int k = 0; k < KT; ++k) {
                const double pb = std::exp((double)bl[k]), lq = lg[id[k]] - lz;
                kl += pb * (bl[k] - lq);
                pb_top += pb;
                pq_top += std::exp(lq);
              }
              const double tb = std::max(1e-12, 1 - pb_top), tq = std::max(1e-12, 1 - pq_top);
              kl += tb * std::log(tb / tq);
              kl_sum += kl;
              kl_max = std::max(kl_max, kl);
              same_top += (std::max_element(lg.begin(), lg.end()) - lg.begin()) == id[0];
            }
          }
        }
        o += n;
      }
      if (klr) printf("[%d] %.4f  KL %.5f  same top %.2f %%\n", ch + 1, std::exp(nll / count), kl_sum / count, 100.0 * same_top / count);
      else printf("[%d] %.4f\n", ch + 1, std::exp(nll / count));
      fflush(stdout);
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("perplexity: %.4f over %d chunks of %d (%ld scored tokens, %zu tokens in file) in %.1f s\n", std::exp(nll / count),
           nchunks, ppl_ctx, count, all.size(), s);
    if (klr) printf("KL vs base: mean %.5f, max %.4f, same top token %.2f %%\n", kl_sum / count, kl_max, 100.0 * same_top / count);
    if (kls) fclose(kls);
    if (klr) fclose(klr);
    return 0;
  }
  if (bench_pp > 0) {
    std::vector<int> bt(bench_pp);
    for (int i = 0; i < bench_pp; ++i) bt[i] = 1000 + (i * 7919) % 50000;
    eng.forward(bt.data(), std::min(bench_pp, 64));  // warm-up (kernel JIT, page-in)
    // (the one-token kernels too, as the server warms up before serving: the prompt pass above builds only the prompt
    //  path, so a first generated token paid their compilation inside tg: a B580's 35B, step 0 4,224.9 ms against a
    //  12.6 ms median, tg128 21.5 tok/s for a ~79 tok/s steady state; tenken, issue #53. warm_up_once leaves the
    //  engine reset, and every prompt run below resets it too)
    if (n_gen > 0) eng.warm_up_once(bt[0], false);
    // (ARC_BENCH_RUNS: prompt runs, default 3; long prompts: 1)
    const int runs = getenv("ARC_BENCH_RUNS") ? std::max(1, atoi(getenv("ARC_BENCH_RUNS"))) : 3;
    for (int r = 0; r < runs; ++r) {
      eng.reset();
      eng.prof_ms.clear();
      eng.prof_n.clear();
      eng.profile = prof && r == runs - 1;
      auto t0 = std::chrono::steady_clock::now();
      eng.forward(bt.data(), bench_pp);
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      printf("pp%d run %d: %.1f ms  ->  %.1f tokens/s\n", bench_pp, r, ms, bench_pp / ms * 1000.0);
    }
    eng.profile = prof_gen;  // (--prof-gen: the per-kernel profile below is of generation instead)
    if (prof_gen) { eng.prof_ms.clear(); eng.prof_n.clear(); prof = true; }
    int tok = 1000;
    const uint64_t b0 = eng.rd_bytes;
    std::vector<double> step_ms;
    const uint64_t ln0 = zrt::launch_n, lt0 = zrt::launch_ns, sa0 = zrt::setarg_ns, sy0 = zrt::sync_n;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_gen; ++i) {
      const auto ts = std::chrono::steady_clock::now();
      const auto& lg = eng.forward(&tok, 1);
      tok = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
      step_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    // (--n 0: the prompt only; the summary reads step_ms[0], which crashed before the pp lines were flushed, 2026-11-12)
    if (n_gen > 0) {
      printf("tg%d: %.1f ms  ->  %.2f tokens/s\n", n_gen, ms, n_gen / ms * 1000.0);
      // (--bench-pp's generation is one token a step, never drafted: with --mtp it measured plain decoding plus the MTP
      //  block's upkeep, and an A770 owner read it as MTP losing to plain; issue #45, 2026-11-18)
      if (eng.has_mtp)
        printf("(tg%d is plain decoding even with --mtp: MTP's speed comes from a prompt, e.g. runz <model> \"<text>\" --n 256 --mtp)\n", n_gen);
      if (zrt::launch_stats)  // (ARC_LAUNCH_STATS: the host's share of a step)
        printf("launches: %.0f per token, host waits %.1f; host %.2f ms appending + %.2f ms setting arguments per token (%.1f us per launch)\n",
               (double)(zrt::launch_n - ln0) / n_gen, (double)(zrt::sync_n - sy0) / n_gen, (zrt::launch_ns - lt0) / 1e6 / n_gen, (zrt::setarg_ns - sa0) / 1e6 / n_gen,
               (zrt::launch_ns - lt0 + zrt::setarg_ns - sa0) / 1e3 / std::max<uint64_t>(1, zrt::launch_n - ln0));
      {  // (a whole slow run or single slow steps)
        std::vector<double> st = step_ms;
        std::sort(st.begin(), st.end());
        printf("tg steps: min %.1f, median %.1f, max %.1f ms; first 8:", st.front(), st[st.size() / 2], st.back());
        for (int i = 0; i < 8 && i < (int)step_ms.size(); ++i) printf(" %.1f", step_ms[i]);
        printf("\n");
      }
      const double gb = (eng.rd_bytes - b0) / 1e9;
      printf("weights read: %.0f MB per token, %.1f GB/s%s\n", gb * 1000 / n_gen, gb / ms * 1000,
             peak_note(eng, gb / ms * 1000).c_str());
    }
    // (the experts line and its host times after --bench-pp too, as after a text prompt: a MoE model's cache hits,
    //  reads and ARC_XCPU's times; the Researcher, #57, 2026-11-22)
    if (!eng.expert_stats().empty()) printf("%s\n", eng.expert_stats().c_str());
  } else {
    if (toks.empty()) { zlog::eprintf("no tokens\n"); return 1; }
    if (kl_decode && getenv("ARC_KL_SAVE")) {
      // (ARC_KL_SAVE=F, 2026-11-23: this run's own log-probs of every decoded position, f32 x vocab, the file --kl-ref
      //  reads: a long-context reference made by the GPU's default path, for a switch's KL against it where an f32 CPU
      //  reference of 16K+ tokens would take hours (bench/dec_attn_kl.ps1). ARC_KL_FROM=N: the first N tokens as a
      //  prompt, positions N.. decoded one by one and saved)
      const size_t V = VK;
      const size_t from = getenv("ARC_KL_FROM") ? std::min<size_t>(atoi(getenv("ARC_KL_FROM")), toks.size() - 1) : 0;
      // (each position is the whole vocabulary in f32, ~1 MB for the 27B: at most 2,048 of them, so a missing ARC_KL_FROM
      //  does not write 16 GB; the Researcher's review)
      if (toks.size() - from > 2048) {
        zlog::eprintf("ARC_KL_SAVE: %zu positions to save (give ARC_KL_FROM so that at most 2,048 are)\n", toks.size() - from);
        return 1;
      }
      FILE* fo = fopen(getenv("ARC_KL_SAVE"), "wb");
      if (!fo) { zlog::eprintf("cannot write %s\n", getenv("ARC_KL_SAVE")); return 1; }
      if (from > 0) eng.forward(toks.data(), (int)from);
      std::vector<float> lp(V);
      for (size_t r = from; r < toks.size(); ++r) {
        const auto& l = eng.forward(&toks[r], 1);
        const double mx = *std::max_element(l.begin(), l.begin() + V);
        double z = 0;
        for (size_t i = 0; i < V; ++i) z += std::exp((double)l[i] - mx);
        const double lz = std::log(z);
        for (size_t i = 0; i < V; ++i) lp[i] = (float)((double)l[i] - mx - lz);
        if (fwrite(lp.data(), 4, V, fo) != V) { fclose(fo); zlog::eprintf("cannot write %s\n", getenv("ARC_KL_SAVE")); return 1; }
      }
      fclose(fo);
      printf("ARC_KL_SAVE: %zu positions (%zu..%zu) of %zu log-probs written to %s\n", toks.size() - from, from,
             toks.size() - 1, V, getenv("ARC_KL_SAVE"));
      return 0;
    }
    if (kl_decode) {
      // the single-token (decode) path at every prompt position against the reference log-probs of that position
      const size_t V = VK;
      const std::vector<float> ref = read_kl_ref(kl_ref, V);
      if (ref.empty()) { zlog::eprintf("cannot read %s\n", kl_ref.c_str()); return 1; }
      const size_t got = ref.size();
      // (ARC_KL_FROM=N: the reference holds positions N.. only (tools/ref_forward REF_ALL_FROM); the tokens before N go
      //  in as a prompt, the rest one by one)
      const size_t from = getenv("ARC_KL_FROM") ? std::min<size_t>(atoi(getenv("ARC_KL_FROM")), toks.size() - 1) : 0;
      if (got != ref.size() || ref.size() != V * (toks.size() - from)) { zlog::eprintf("reference must hold every position\n"); return 1; }
      double sum = 0, mxk = 0;
      int agree = 0;
      if (from > 0) eng.forward(toks.data(), (int)from);
      for (size_t r = from; r < toks.size(); ++r) {
        const auto& l = eng.forward(&toks[r], 1);
        const double mx = *std::max_element(l.begin(), l.end());
        double z = 0;
        for (float v : l) z += std::exp((double)v - mx);
        const float* rr = ref.data() + (r - from) * V;
        double kl = 0;
        for (size_t i = 0; i < V; ++i)  // (a reference log-prob of -inf adds nothing)
          if (rr[i] > -1e30f) kl += std::exp((double)rr[i]) * (rr[i] - (l[i] - mx - std::log(z)));
        kl += kl_tail(l.data(), rr);  // (a compact reference's rest, read_kl_ref)
        sum += kl;
        if (getenv("ARC_KL_SHOW") && kl > atof(getenv("ARC_KL_SHOW"))) printf("pos %zu KL %.3f\n", r, kl);  // (ARC_KL_SHOW=x: positions above x)
        mxk = std::max(mxk, kl);
        agree += (std::max_element(l.begin(), l.end()) - l.begin()) == (std::max_element(rr, rr + V) - rr); kl_flip(r, l.data(), rr, V);
      }
      const size_t n = toks.size() - from;
      printf("KL decode path (ref || inferzero) = %.5f (mean over %zu positions, max %.5f), top-1 agreement %d/%zu\n",
             sum / n, n, mxk, agree, n);
      return 0;
    }
    // warm-up so kernel compilation is not counted in the time to first token
    // (with MTP, the server's warm-up: after an update every draft count's verification and drafting kernels are
    //  built once before the timed run (warm_up_once, a marker in the kernel cache). Without it the first run after a
    //  build compiled them inside the generation: the laptop's 35B 21.3 tok/s plain, 4.6 with MTP that first time
    //  and ~35 after; a B70's 27B 8.4-8.7 tok/s in its first MTP round, issue #27, 2026-11-18)
    if (eng.has_mtp) eng.warm_up_once(toks[0], false);
    else { int w = toks[0]; eng.forward(&w, 1); eng.reset(); }
    // (ARC_PROMPT_WARM=N: N untimed runs of the whole prompt first, e.g. to fill the expert cache)
    for (int r = 0, nw = getenv("ARC_PROMPT_WARM") ? atoi(getenv("ARC_PROMPT_WARM")) : 0; r < nw; ++r) {
      auto tw = std::chrono::steady_clock::now();
      eng.forward(toks.data(), (int)toks.size());
      eng.reset();
      printf("warm-up prompt run %d: %.1f ms\n", r,
             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw).count());
    }
    eng.profile = prof;
    auto t0 = std::chrono::steady_clock::now();
    // --feed-step N: the prompt goes through in pieces of N tokens (N <= 8: the single-token / small-batch kernels)
    if (feed_step > 0)
      for (size_t i = 0; i + feed_step < toks.size(); i += feed_step) eng.forward(toks.data() + i, feed_step);
    const size_t rest0 = feed_step > 0 ? (toks.size() - 1) / feed_step * feed_step : 0;
    // (ARC_DUMP_ROWS=file: the residual row of the last prompt token after each block, f32, for tools/ref_forward REF_DUMP)
    const char* dump_rows = getenv("ARC_DUMP_ROWS");
    eng.dbg_rows = dump_rows != nullptr;
    const auto& lg = eng.forward(toks.data() + rest0, (int)(toks.size() - rest0));
    if (dump_rows) {
      FILE* fd = fopen(dump_rows, "wb");
      for (const auto& r : eng.dbg_) fwrite(r.data(), 4, r.size(), fd);
      fclose(fd);
      eng.dbg_rows = false;
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("prompt: %zu tokens | time to first token %.1f ms (%.1f t/s)\n", toks.size(), ms, toks.size() / ms * 1000.0);
    std::vector<std::pair<int, float>> tk;
    top_k(lg, 10, tk);
    printf("top-10 next token:");
    for (auto& [id, p] : tk) printf(" %d(%.4f)", id, p);
    if (!kl_ref.empty()) {
      // KL(reference || inferzero) over the full vocabulary; the reference file holds f32 log-probs of the last
      // position, or of every prompt position (then the KL is averaged over the prompt; single-chunk prompts).
      const size_t V = eng.hp.inkling ? VK : lg.size();
      auto kl_of = [&](const std::vector<float>& l, const float* ref) {
        const double mx = *std::max_element(l.begin(), l.end());
        double z = 0;
        for (float v : l) z += std::exp((double)v - mx);
        double kl = 0;
        for (size_t i = 0; i < V; ++i)  // (a reference log-prob of -inf adds nothing)
          if (ref[i] > -1e30f) kl += std::exp((double)ref[i]) * (ref[i] - (l[i] - mx - std::log(z)));
        return kl + kl_tail(l.data(), ref);
      };
      const std::vector<float> ref = read_kl_ref(kl_ref, VK);
      // (ARC_KL_FROM=N: the reference holds positions N.. only; they must lie in the last chunk, which x_ holds)
      // (kl_c0: the first position whose logits the last forward() left: the last call starts at rest0 (--feed-step),
      //  and of its rows x_ holds the last chunk. It was the chunk's start counted from 0, so with --feed-step the rows
      //  of the last piece were scored against the reference's first positions, then logits_at threw: issue #23)
      const size_t kl_from = getenv("ARC_KL_FROM") ? std::min<size_t>(atoi(getenv("ARC_KL_FROM")), toks.size() - 1) : 0,
                   kl_c0 = rest0 + (toks.size() - rest0 - 1) / eng.max_chunk * eng.max_chunk;
      if (ref.size() == V) {
        printf("\nKL(ref || inferzero) = %.5f", kl_of(lg, ref.data()));
      } else if (ref.size() == V * (toks.size() - kl_from) && kl_from >= kl_c0) {
        const std::vector<float> last(lg.begin(), lg.end());  // logits_at() overwrites the shared logits buffer
        double sum = 0, mx = 0;
        int agree = 0;  // (positions whose top token is the reference's)
        // (the logits' size against the reference's, each row's mean taken out: log-probs and logits differ by a constant)
        double d2 = 0, r2 = 0, worst_cos = 1, top5 = 0;
        std::vector<int> ia(V), ib(V);
        for (size_t r = kl_from; r < toks.size(); ++r) {
          const std::vector<float>& l = eng.logits_at((int)(r - kl_c0));
          const float* rr = ref.data() + (r - kl_from) * V;
          const double k = kl_of(l, rr);
          {
            double ma = 0, mb = 0;
            for (size_t i = 0; i < V; ++i) ma += l[i], mb += rr[i];
            ma /= (double)V, mb /= (double)V;
            double ab = 0, aa = 0, bb = 0;
            for (size_t i = 0; i < V; ++i) {
              const double x = l[i] - ma, y = rr[i] - mb;
              ab += x * y, aa += x * x, bb += y * y, d2 += (x - y) * (x - y);
            }
            r2 += bb;
            worst_cos = std::min(worst_cos, ab / std::sqrt(aa * bb));
            for (size_t i = 0; i < V; ++i) ia[i] = ib[i] = (int)i;
            std::partial_sort(ia.begin(), ia.begin() + 5, ia.end(), [&](int a, int b) { return l[a] > l[b]; });
            std::partial_sort(ib.begin(), ib.begin() + 5, ib.end(), [&](int a, int b) { return rr[a] > rr[b]; });
            for (int a = 0; a < 5; ++a) top5 += std::find(ib.begin(), ib.begin() + 5, ia[a]) != ib.begin() + 5 ? 0.2 : 0.0;
          }
          if (getenv("ARC_KL_SHOW") && k > atof(getenv("ARC_KL_SHOW"))) {  // (ARC_KL_SHOW=x: positions above x)
            // (with each side's top token and its probability of the prompt's next token: which one the text agrees with)
            double lm = *std::max_element(l.begin(), l.end()), z = 0;
            for (float v : l) z += std::exp((double)v - lm);
            const size_t te = std::max_element(l.begin(), l.end()) - l.begin(), tr = std::max_element(rr, rr + V) - rr;
            if (r + 1 < toks.size()) {
              const int nx = toks[r + 1];
              printf("pos %zu KL %.3f | next %d: inferzero p %.4f, ref p %.4f | top inferzero %zu (%.3f), ref %zu (%.3f)\n", r, k,
                     nx, std::exp((double)l[nx] - lm) / z, std::exp((double)rr[nx]), te, std::exp((double)l[te] - lm) / z, tr,
                     std::exp((double)rr[tr]));
            } else {
              printf("pos %zu KL %.3f\n", r, k);
            }
          }
          sum += k;
          mx = std::max(mx, k);
          agree += std::max_element(l.begin(), l.end()) - l.begin() == std::max_element(rr, rr + V) - rr;
          kl_flip(r, l.data(), rr, V);
        }
        const size_t n = toks.size() - kl_from;
        printf("\nKL(ref || inferzero) = %.5f (mean over %zu positions, max %.5f), top token agrees %d / %zu",
               sum / n, n, mx, agree, n);
        if (g_ref_tail.empty())  // (a compact reference has no full rows to compare: read_kl_ref)
          printf("\nlogits against the reference: relative L2 %.3f %%, worst cosine %.6f, top-5 overlap %.3f",
                 100.0 * std::sqrt(d2 / r2), worst_cos, top5 / (double)n);
        std::copy(last.begin(), last.end(), const_cast<float*>(lg.data()));
      } else if (ref.size() == V * (toks.size() - kl_from)) {
        printf("\n--kl-ref: the reference starts at position %zu, but only positions %zu.. are left to read (the last "
               "%s); give ARC_KL_FROM=%zu and a reference of those positions (ref_forward with REF_ALL_FROM=%zu)",
               kl_from, kl_c0, feed_step > 0 ? "--feed-step piece's last chunk" : "prompt chunk", kl_c0, kl_c0);
      } else {
        printf("\n--kl-ref: %s holds %zu floats, not one position (%zu) or positions %zu..%zu (%zu)", kl_ref.c_str(),
               ref.size(), V, kl_from, toks.size() - 1, V * (toks.size() - kl_from));
      }
    }
    eng.profile = prof_gen;
    if (prof_gen) { eng.prof_ms.clear(); eng.prof_n.clear(); prof = true; }
    std::vector<int> gen;
    int tok = tk[0].first;
    const uint64_t rb0 = eng.rd_bytes;
    auto t1 = std::chrono::steady_clock::now();
    int steps = 0, accepted = 0, drafted = 0, copy_steps = 0, copy_drafted = 0, copy_accepted = 0;
    double mtp_draft_ms = 0, mtp_verify_ms = 0, mtp_done_ms = 0;  // (where an MTP step's time goes, 2026-11-18)
    if (eng.has_mtp) {
      // speculative decoding with the MTP block: up to two drafts per step, verified by one main-model pass
      const size_t V = eng.hp.n_vocab;
      auto argmax_row = [&](const std::vector<float>& l, int r) {
        return (int)(std::max_element(l.begin() + r * V, l.begin() + (r + 1) * V) - (l.begin() + r * V));
      };
      // (--dump-logits: the prompt's logits, then each verification pass's rows that chose a token: the same rows,
      //  in the same order, as the plain loop below writes, so the two files compare bitwise)
      FILE* dl = dump_logits.empty() ? nullptr : fopen(dump_logits.c_str(), "wb");
      struct Close { FILE*& f; ~Close() { if (f) fclose(f); } } close_dl{dl};
      if (dl) fwrite(lg.data(), sizeof(float), lg.size(), dl);
      std::vector<int> hist(toks.begin(), toks.end());  // (the tokens before `tok`: copy drafting's context, R1)
      while ((int)gen.size() < n_gen) {
        gen.push_back(tok);
        if (tok == vocab.eos || (int)gen.size() >= n_gen) break;
        const auto c0 = std::chrono::steady_clock::now();
        hist.resize(toks.size());
        hist.insert(hist.end(), gen.begin(), gen.end() - 1);
        const std::vector<int> drafts = eng.mtp_draft(tok, eng.opt.mtp_drafts, &hist);
        const bool copied = eng.last_draft_copy();
        const auto c1 = std::chrono::steady_clock::now();
        const auto& lv = eng.verify(tok, drafts);
        const auto c2 = std::chrono::steady_clock::now();
        mtp_draft_ms += std::chrono::duration<double, std::milli>(c1 - c0).count();
        mtp_verify_ms += std::chrono::duration<double, std::milli>(c2 - c1).count();
        int acc = 0, next = argmax_row(lv, 0);
        if (dl) fwrite(lv.data(), sizeof(float), V, dl);
        bool stop = false;
        while (acc < (int)drafts.size() && next == drafts[acc]) {
          gen.push_back(next);
          ++acc;
          if (next == vocab.eos || (int)gen.size() >= n_gen) { stop = true; break; }
          next = argmax_row(lv, acc);
          if (dl) fwrite(lv.data() + acc * V, sizeof(float), V, dl);
        }
        const auto c3 = std::chrono::steady_clock::now();
        eng.verify_done(acc);
        mtp_done_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c3).count();
        ++steps;
        drafted += (int)drafts.size();
        accepted += acc;
        if (copied) ++copy_steps, copy_drafted += (int)drafts.size(), copy_accepted += acc;
        if (stop) break;
        tok = next;
      }
    } else {
      FILE* dl = dump_logits.empty() ? nullptr : fopen(dump_logits.c_str(), "wb");
      struct Close { FILE*& f; ~Close() { if (f) fclose(f); } } close_dl{dl};
      if (dl) fwrite(lg.data(), sizeof(float), lg.size(), dl);  // the prompt's logits first
      for (int i = 0; i < n_gen; ++i) {
        gen.push_back(tok);
        if (tok == vocab.eos) break;
        const auto& l2 = eng.forward(&tok, 1);
        if (dl) fwrite(l2.data(), sizeof(float), l2.size(), dl);
        tok = (int)(std::max_element(l2.begin(), l2.end()) - l2.begin());
      }
    }
    double gms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    printf("\ngreedy:");
    for (int id : gen) printf(" %d", id);
    printf("\ngeneration: %zu tokens in %.1f ms (%.2f t/s)\n", gen.size(), gms, gen.size() / gms * 1000.0);
    if (!eng.expert_stats().empty()) printf("%s\n", eng.expert_stats().c_str());
    {
      const double gb = (eng.rd_bytes - rb0) / 1e9;
      printf("weights read: %.0f MB per pass, %.1f GB/s%s (drafter weights not counted)\n",
             gb * 1000 / std::max<size_t>(1, steps ? (size_t)steps : gen.size()), gb / gms * 1000,
             peak_note(eng, gb / gms * 1000).c_str());
    }
    if (steps) printf("MTP: %d drafts, %d accepted (%.1f %%), %.2f tokens per main-model pass (%d passes)\n", drafted,
                      accepted, 100.0 * accepted / std::max(drafted, 1), (double)gen.size() / steps, steps);
    if (copy_steps)
      printf("copy drafting: %d of %d steps, %d drafts, %d accepted (%.1f %%)\n", copy_steps, steps, copy_drafted,
             copy_accepted, 100.0 * copy_accepted / std::max(copy_drafted, 1));
    if (steps)
      printf("MTP time: drafting %.1f ms, verification %.1f ms (%.1f ms a pass), verify_done %.1f ms\n", mtp_draft_ms,
             mtp_verify_ms, mtp_verify_ms / steps, mtp_done_ms);
    if (eng.dsp_on() && eng.dsp_cmp_n_)  // (ARC_DSP_CMP=1: the GPU chain's drafts against the host's)
      printf("DSpark compare: %llu of %llu drafts the same on the GPU chain as on the host's (up to the first that differs)\n",
             (unsigned long long)eng.dsp_cmp_same_, (unsigned long long)eng.dsp_cmp_n_);
    if (eng.dsp_on() && eng.dsp_checks_)  // (ARC_DSP_CHECK=1: the conv-state rollback against ink_conv_state)
      printf("DSpark check: %llu verifications, the conv states %s\n", (unsigned long long)eng.dsp_checks_,
             eng.dsp_check_bad_ ? ("DIFFER in " + std::to_string(eng.dsp_check_bad_)).c_str() : "the same bit for bit every time");
    if (show) printf("----- text -----\n%s\n----------------\n", vocab.decode(gen).c_str());
  }
  if (prof) {
    double total = 0;
    for (auto& [n, ms] : eng.prof_ms) total += ms;
    std::vector<std::pair<double, std::string>> rows;
    for (auto& [n, ms] : eng.prof_ms) rows.push_back({ms, n});
    std::sort(rows.rbegin(), rows.rend());
    printf("GPU time by kernel (last profiled forward): total %.1f ms\n", total);
    for (auto& [ms, n] : rows) printf("  %-20s %8.2f ms  %5.1f%%  (%d launches)\n", n.c_str(), ms, 100 * ms / total, eng.prof_n[n]);
  }
  return 0;
}
