// LFM2 / LFM2.5 MoE (GGUF architecture lfm2moe: LFM2.5-24B-A2B). A part of class Engine
// (zero/engine.h), included inside the class body: not a header of its own.
//
// The model: n_layer blocks, each x += op(rmsnorm(x)); x += ffn(rmsnorm(x)). op is a short convolution (layers
// without KV heads: in_proj to [b | c | x], a causal depthwise convolution of 3 taps over b * x, times c, out_proj)
// or attention (q / k RMS norms per head, NEOX RoPE, 1 / sqrt(hd): the Qwen3 dense path, layer_qwen3_attn). ffn is
// a dense SwiGLU in the first lfm_dense layers (layer_ffn) and a mixture of experts after them: sigmoid router,
// the n_expert_used largest of probs + bias, weights probs / their sum, SwiGLU experts.
// The experts are ordinary resident weights (the whole model fits a 16 GB card): each expert is a view of its
// layer's stacked rows, run through matmul() on the token rows routed to it. Reference: llama.cpp
// src/models/lfm2.cpp, lfm2moe.cpp, lfm-base.cpp.

  void load_hparams_lfm2() {
    const std::string a = hp.arch;
    auto I = [&](const std::string& k, int64_t def = -1) {
      int64_t v = f_.get_int(a + "." + k, def);
      if (v < 0) throw std::runtime_error("missing hparam " + k);
      return (int)v;
    };
    hp.qwen3 = true;  // (the float attention, the stacked q / k / v and gate / up, the plain FFN)
    hp.lfm = true;
    kf16_ = true;
    hp.n_layer = I("block_count");
    if (hp.n_layer <= 0 || hp.n_layer > 4096) throw std::runtime_error("invalid block_count in the GGUF");
    hp.n_embd = I("embedding_length");
    hp.n_ff = I("feed_forward_length");
    hp.n_head = I("attention.head_count");
    const auto* kv = f_.get_ints(a + ".attention.head_count_kv");
    if (!kv || (int)kv->size() < hp.n_layer) throw std::runtime_error("lfm2moe: attention.head_count_kv per layer expected");
    hp.kv_heads.assign(kv->begin(), kv->begin() + hp.n_layer);
    hp.n_head_kv = 0;
    for (int v : hp.kv_heads) {
      if (v && hp.n_head_kv && v != hp.n_head_kv) throw std::runtime_error("lfm2moe: attention layers of different KV head counts");
      if (v) hp.n_head_kv = v;
    }
    if (hp.n_head <= 0 || hp.n_head_kv <= 0 || hp.n_embd % hp.n_head) throw std::runtime_error("invalid model dimensions in the GGUF");
    hp.head_dim = hp.n_embd / hp.n_head;
    hp.n_rot = hp.head_dim;
    hp.full_attn_interval = 1;
    hp.eps = (float)f_.get_float(a + ".attention.layer_norm_rms_epsilon", 1e-5);
    hp.rope_base = (float)f_.get_float(a + ".rope.freq_base", 1e6);
    hp.conv_k = I("shortconv.l_cache");
    hp.ssm_d_inner = hp.n_embd;  // (conv_dim() = n_embd: the conv state's rows, the chunk buffers of its width)
    hp.lfm_dense = (int)f_.get_int(a + ".leading_dense_block_count", 0);
    hp.n_expert = I("expert_count");
    hp.n_expert_used = I("expert_used_count");
    hp.n_ff_exp = I("expert_feed_forward_length");
    hp.lfm_wscale = (float)f_.get_float(a + ".expert_weights_scale", 0.0);
    tok_embd_ = &f_.get("token_embd.weight");
    if (tok_embd_->dims.size() < 2 || tok_embd_->dims[1] <= 0 || tok_embd_->dims[0] != (uint64_t)hp.n_embd)
      throw std::runtime_error("token_embd.weight is not n_embd wide in the GGUF");
    hp.n_vocab = (int)tok_embd_->dims[1];
    if (hp.head_dim != 64 || hp.conv_k != 3 || hp.n_embd % 256 || hp.n_ff % 256 || hp.n_ff_exp % 256 ||
        hp.n_head % hp.n_head_kv || (hp.n_head_kv * hp.head_dim) % 256 || hp.n_expert < hp.n_expert_used ||
        hp.n_expert_used < 1 || hp.n_expert > 1024 || f_.get_int(a + ".expert_gating_func", 2) != 2 ||
        hp.lfm_dense < 0 || hp.lfm_dense > hp.n_layer)
      throw std::runtime_error("unsupported LFM2 MoE dimensions");
    if (!opt.attn_int8) throw std::runtime_error("f32 attention (--attn-f32) does not support LFM2 models");
    // (slots: the convolution state and the expert buffers are one conversation's; a split model: only the heads
    //  are split; no drafter knows the convolution layers)
    if (opt.slots > 1 || opt.mtp || !opt.dflash.empty() || !opt.dspark.empty() || opt.tp_size > 1)
      throw std::runtime_error("LFM2: slots, drafters and several GPUs are not supported yet");
  }

  // a matrix with a staging role of the caller's choice (load_w takes it from the tensor's name)
  GpuWeight load_w_role(const std::string& name, int role) {
    const GgufTensor& t = mat(T(name));
    return make_weight(t.type, t.data, t.nbytes, (int)t.dims[1], (int)t.dims[0], role);
  }

  void load_lfm_conv(Layer& L, const std::string& p) {
    const int E = hp.n_embd;
    L.lf_in = load_w_role(p + "shortconv.in_proj.weight", 0);   // (roles 0 and 2: the DeltaNet's, unused here)
    L.lf_out = load_w_role(p + "shortconv.out_proj.weight", 2);
    expect(L.lf_in, 3 * E, E, p + "shortconv.in_proj.weight");
    expect(L.lf_out, E, E, p + "shortconv.out_proj.weight");
    const GgufTensor& cw = f_.get(p + "shortconv.conv.weight");
    if (cw.nelements() != (uint64_t)hp.conv_k * E) throw std::runtime_error("unexpected shortconv.conv.weight shape");
    L.conv_w = load_f32(p + "shortconv.conv.weight");
    L.conv_state = buf(conv_bytes());
    L.S = buf(state_bytes());
  }

  // The experts of a layer: each tensor uploaded whole (expert-major rows), an expert's matrix a view of its rows.
  void load_lfm_moe(Layer& L, const std::string& p) {
    const int E = hp.n_embd, F = hp.n_ff_exp, NE = hp.n_expert;
    static const char* nm[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
    static const int role[3] = {7, 8, 9};  // (ffn_gate / ffn_up / ffn_down)
    for (int k = 0; k < 3; ++k) {
      const GgufTensor& t = T(p + nm[k]);
      const int K = k == 2 ? F : E, N = k == 2 ? E : F;
      if (t.dims.size() != 3 || t.dims[0] != (uint64_t)K || t.dims[1] != (uint64_t)N || t.dims[2] != (uint64_t)NE || !is_w8_type(t.type))
        throw std::runtime_error("unexpected expert tensor " + p + nm[k]);
      bool tiled = false;
      cl_mem all = upload_padded(t.type, t.data, t.nbytes, false, &tiled);
      const size_t xb = (size_t)N * (K / 256) * raw_padded_sb_bytes(t.type);  // (an expert's rows in the GPU raw form)
      L.lf_x[k].resize(NE);
      for (int e = 0; e < NE; ++e) {
        cl_mem v = opt.dry ? nullptr : from(all, (size_t)e * xb);
        L.lf_x[k][e] = make_weight(t.type, nullptr, 0, N, K, role[k], opt.dry ? &all : &v);
      }
    }
    L.lf_router = load_f32(p + "ffn_gate_inp.weight");
    const GgufTensor& tb = f_.get(p + "exp_probs_b.bias");
    L.lf_bias.assign(NE, 0.0f);
    if ((int)tb.nelements() != NE) throw std::runtime_error("unexpected exp_probs_b.bias shape");
    dequant_row(tb.type, tb.data, NE, L.lf_bias.data());
  }

  void alloc_lfm() {
    const size_t M = max_chunk;
    lf_probs_ = buf(M * hp.n_expert * 4);
    // (each expert's rows padded to a GEMM tile: at most 32 rows of padding an expert)
    const size_t slots = M * hp.n_expert_used + (size_t)hp.n_expert * 32;
    lf_idx_ = buf(slots * 4);
    lf_wts_ = buf(slots * 4);
    lf_acc_ = buf(M * hp.n_embd * 4);
    lf_probs_host_.resize(M * hp.n_expert);
  }

  cl_kernel lfk(const char* name) { return kernel("lfm", "", name); }

  // x += out_proj(c * conv3(b * x)), [b | c | x] = in_proj(rmsnorm(x))
  void layer_lfm_conv(Layer& L, int T, int Mp) {
    const int E = hp.n_embd;
    const int a_in = act32_for(false);
    rmsnorm_quant(x_, 0, L.attn_norm, E, Mp, T, a_in);
    matmul("gemm_conv_in", L.lf_in, Mp, big_, 3 * E, 0, 0, a_in);
    {
      cl_kernel k = lfk("lfm_conv");
      int EE = E;
      set_args(k, 0, big_, L.conv_state, L.conv_w, EE, o_);
      size_t gws[2] = {(size_t)E, (size_t)T};
      run("lfm_conv", k, 2, gws, nullptr);
      cl_kernel k2 = lfk("lfm_conv_state");
      int TT = T;
      set_args(k2, 0, big_, L.conv_state, EE, TT);
      size_t g2 = (size_t)E;
      run("lfm_conv", k2, 1, &g2, nullptr);
    }
    const int a_out = act32_for(true);
    quant_rows(o_, E, Mp, T, a_out);
    matmul("gemm_conv_out", L.lf_out, Mp, x_, E, 0, 1, a_out);
  }

  // One token (generation): aq_ holds the quantized h already (rmsnorm_quant), so each chosen expert's gate and up run
  // on it as they are (one launch a pair, gemv_gu), the four SwiGLUs are quantized as four rows of one launch, each
  // down projection reads its row through views of the activation buffers, and one kernel adds the weighted sum.
  // The same arithmetic as the general path below. ARC_LFM_FAST1=0: the general path.
  static bool lfm_fast1() {
    static const bool on = !getenv("ARC_LFM_FAST1") || atoi(getenv("ARC_LFM_FAST1")) != 0;
    return on;
  }
  void lfm_moe_one(Layer& L, int Mp) {
    const int E = hp.n_embd, F = hp.n_ff_exp, NE = hp.n_expert, U = hp.n_expert_used;
    cl_mem d = qrope_;
    float* p = lf_probs_host_.data();
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, lf_probs_, CL_TRUE, 0, (size_t)NE * 4, p, 0, nullptr, nullptr));
    int order[1024];
    float sel[1024];
    for (int e = 0; e < NE; ++e) { sel[e] = p[e] + L.lf_bias[e]; order[e] = e; }
    std::partial_sort(order, order + U, order + NE, [&](int a, int b) { return sel[a] > sel[b] || (sel[a] == sel[b] && a < b); });
    float sum = 0.0f, w[4] = {};  // (U == 4: layer_lfm_moe takes this path for 4 experts a token only)
    for (int i = 0; i < U; ++i) sum += p[order[i]];
    const float sc = (hp.lfm_wscale != 0.0f && hp.lfm_wscale != 1.0f ? hp.lfm_wscale : 1.0f) / sum;
    for (int i = 0; i < U; ++i) w[i] = p[order[i]] * sc;
    static const bool gu = !getenv("ARC_LFM_GU") || atoi(getenv("ARC_LFM_GU")) != 0;
    for (int i = 0; i < U; ++i) {
      const int e = order[i];
      cl_mem C = from(big_, (size_t)i * 2 * F * 4);
      if (gu && gemv_gu(L.lf_x[0][e], L.lf_x[1][e], Mp, C, true)) continue;
      matmul("moe_gate", L.lf_x[0][e], Mp, C, 2 * F, 0, 0, 1);
      matmul("moe_up", L.lf_x[1][e], Mp, C, 2 * F, F, 0, 1);
    }
    {
      int gelu = 0, FF = F, MM = Mp, TT = U, a32 = 1, fu = 0;
      set_args(k_swiglu_, 0, big_, FF, MM, TT, aq_, asc_, absum_, a32, fu, gelu);
      size_t gws[2] = {(size_t)16 * (F / 256), (size_t)U}, lws[2] = {16, 1};
      run("moe_swiglu", k_swiglu_, 2, gws, lws);
    }
    {
      const cl_mem a = aq_, s = asc_, b = absum_;
      struct Back { cl_mem& a; cl_mem& s; cl_mem& b; cl_mem a0, s0, b0; ~Back() { a = a0; s = s0; b = b0; } } back{aq_, asc_, absum_, a, s, b};
      for (int i = 0; i < U; ++i) {
        aq_ = from(a, (size_t)i * F);   // (row i of [M][F] int8; of [F / 32][M] f32 and [F / 16][M] shorts)
        asc_ = from(s, (size_t)i * 4);
        absum_ = from(b, (size_t)i * 2);
        matmul("moe_down", L.lf_x[2][order[i]], Mp, from(d, (size_t)i * E * 4), E, 0, 0, 1);
      }
    }
    cl_kernel k = lfk("lfm_combine4");
    int EE = E;
    set_args(k, 0, x_, d, EE, w[0], w[1], w[2], w[3]);
    size_t g1 = (size_t)E;
    run("moe_add", k, 1, &g1, nullptr);
  }

  // x += sum over each token's experts of weight * down(silu(gate(h)) * up(h)), h = rmsnorm(x)
  void layer_lfm_moe(Layer& L, int T, int Mp) {
    const int E = hp.n_embd, F = hp.n_ff_exp, NE = hp.n_expert, U = hp.n_expert_used;
    cl_mem xn = conv_out_, g = o_, d = qrope_;  // (chunk buffers of n_embd floats a row, free between the blocks)
    {
      float eps = hp.eps;
      int EE = E, MM = Mp, TT = T, r0 = 0, wh = 1, a32 = 1;
      set_args(k_rmsnorm_, 0, x_, r0, L.post_norm, EE, MM, TT, eps, aq_, asc_, absum_, xn, wh, a32);
      size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
      run("rmsnorm_quant", k_rmsnorm_, 2, gws, lws);
    }
    {
      cl_kernel k = lfk("lfm_router");
      int EE = E, ne = NE;
      set_args(k, 0, xn, L.lf_router, EE, ne, lf_probs_);
      size_t gws[2] = {(size_t)NE * 16, (size_t)T}, lws[2] = {16, 1};
      run("moe_router", k, 2, gws, lws);
    }
    if (T == 1 && U == 4 && batch_.empty() && !q2mm_ && lfm_fast1()) return lfm_moe_one(L, Mp);
    float* pr = lf_probs_host_.data();
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, lf_probs_, CL_TRUE, 0, (size_t)T * NE * 4, pr, 0, nullptr, nullptr));
    // each expert's token rows and weights
    lf_rows_.resize(NE);
    lf_w_.resize(NE);
    for (int e = 0; e < NE; ++e) { lf_rows_[e].clear(); lf_w_[e].clear(); }
    int order[1024];
    float sel[1024];
    for (int t = 0; t < T; ++t) {
      const float* p = pr + (size_t)t * NE;
      for (int e = 0; e < NE; ++e) { sel[e] = p[e] + L.lf_bias[e]; order[e] = e; }
      std::partial_sort(order, order + U, order + NE, [&](int a, int b) { return sel[a] > sel[b] || (sel[a] == sel[b] && a < b); });
      float sum = 0.0f;
      for (int i = 0; i < U; ++i) sum += p[order[i]];
      const float sc = (hp.lfm_wscale != 0.0f && hp.lfm_wscale != 1.0f ? hp.lfm_wscale : 1.0f) / sum;
      for (int i = 0; i < U; ++i) {
        lf_rows_[order[i]].push_back(t);
        lf_w_[order[i]].push_back(p[order[i]] * sc);
      }
    }
    const int tm = T == 1 ? 16 : std::max(16, gemm_tm());
    std::vector<int> offs(NE, 0);
    lf_id_.clear();
    lf_ww_.clear();
    for (int e = 0; e < NE; ++e) {
      const int n = (int)lf_rows_[e].size();
      if (!n) continue;
      offs[e] = (int)lf_id_.size();
      const int np = (n + tm - 1) / tm * tm;
      for (int i = 0; i < np; ++i) {
        lf_id_.push_back(lf_rows_[e][i < n ? i : 0]);
        lf_ww_.push_back(i < n ? lf_w_[e][i] : 0.0f);
      }
    }
    CL_CHECK(clEnqueueWriteBuffer(cl_.queue, lf_idx_, CL_TRUE, 0, lf_id_.size() * 4, lf_id_.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueWriteBuffer(cl_.queue, lf_wts_, CL_TRUE, 0, lf_ww_.size() * 4, lf_ww_.data(), 0, nullptr, nullptr));
    fill(lf_acc_, (size_t)T * E * 4);
    const int saved_T = cur_T_;
    struct Back { int& v; int s; ~Back() { v = s; } } back{cur_T_, saved_T};
    cl_kernel kg = lfk("lfm_gather"), ks = lfk("lfm_scatter");
    for (int e = 0; e < NE; ++e) {
      const int n = (int)lf_rows_[e].size();
      if (!n) continue;
      const int np = (n + tm - 1) / tm * tm;
      int nn = n, EE = E, off = offs[e];
      set_args(kg, 0, xn, lf_idx_, off, nn, EE, g);
      size_t gg[2] = {(size_t)E, (size_t)np};
      run("moe_gather", kg, 2, gg, nullptr);
      cur_T_ = n;
      quant_rows(g, E, np, n, 1);
      const int fused = n > 8 && !q2mm_ ? 1 : 0;  // (two passes: no fused epilogue)
      if (fused) {
        matmul("moe_gate", L.lf_x[0][e], np, big_, F, 0, 0, 1);
        matmul("moe_up", L.lf_x[1][e], np, big_, F, 0, 2, 1);
      } else {
        matmul("moe_gate", L.lf_x[0][e], np, big_, 2 * F, 0, 0, 1);
        matmul("moe_up", L.lf_x[1][e], np, big_, 2 * F, F, 0, 1);
      }
      {
        int gelu = 0, FF = F, MM = np, TT = n, a32 = 1, fu = fused;
        set_args(k_swiglu_, 0, big_, FF, MM, TT, aq_, asc_, absum_, a32, fu, gelu);
        size_t gws[2] = {256, (size_t)np}, lws[2] = {256, 1};
        run("moe_swiglu", k_swiglu_, 2, gws, lws);
      }
      matmul("moe_down", L.lf_x[2][e], np, d, E, 0, 0, 1);
      set_args(ks, 0, d, lf_idx_, lf_wts_, off, EE, lf_acc_);
      size_t gs[2] = {(size_t)E, (size_t)n};
      run("moe_scatter", ks, 2, gs, nullptr);
    }
    cur_T_ = saved_T;
    {
      cl_kernel k = lfk("lfm_add");
      int EE = E;
      set_args(k, 0, x_, lf_acc_, EE);
      size_t gws[2] = {(size_t)E, (size_t)T};
      run("moe_add", k, 2, gws, nullptr);
    }
  }
