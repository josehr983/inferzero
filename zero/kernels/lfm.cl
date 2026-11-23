// LFM2 / LFM2.5 MoE (GGUF architecture lfm2moe, Liquid AI) on inferzero: the operations that are its own
// (zero/engine_lfm.h). The matrix products run on the engine's int8 GEMM / GEMV; these are f32 row kernels.
// Reference: llama.cpp src/models/lfm-base.cpp (build_lfm_shortconv, build_lfm_moe_ffn).

// Short convolution: bcx[m] = [b | c | x] (3 E floats a row, the in_proj output). bx = b * x; the causal depthwise
// convolution of l_cache = 3 taps over bx (state: bx of the two positions before the chunk), times c.
// w[ch * 3 + j]: tap j of channel ch, tap 2 the current position. Grid (E, T).
__kernel void lfm_conv(__global const float* bcx, __global const float* state, __global const float* w, int E,
                       __global float* out) {
  const int ch = get_global_id(0), m = get_global_id(1);
  __global const float* r0 = bcx + (size_t)m * 3 * E;
  const float bx0 = r0[ch] * r0[2 * E + ch];
  float bx1, bx2;  // (one and two positions back)
  if (m >= 1) {
    __global const float* r1 = r0 - 3 * E;
    bx1 = r1[ch] * r1[2 * E + ch];
  } else {
    bx1 = state[E + ch];
  }
  if (m >= 2) {
    __global const float* r2 = r0 - 6 * E;
    bx2 = r2[ch] * r2[2 * E + ch];
  } else {
    bx2 = state[(m + 0) * E + ch];  // (m = 0: state row 0; m = 1: state row 1)
  }
  out[(size_t)m * E + ch] = r0[E + ch] * (w[ch * 3] * bx2 + w[ch * 3 + 1] * bx1 + w[ch * 3 + 2] * bx0);
}

// The state after a chunk of T rows: bx of its last two positions (T = 1: the old last one moves up). Grid (E).
__kernel void lfm_conv_state(__global const float* bcx, __global float* state, int E, int T) {
  const int ch = get_global_id(0);
  __global const float* rl = bcx + (size_t)(T - 1) * 3 * E;
  const float last = rl[ch] * rl[2 * E + ch];
  float prev;
  if (T >= 2) {
    __global const float* rp = rl - 3 * E;
    prev = rp[ch] * rp[2 * E + ch];
  } else {
    prev = state[E + ch];
  }
  state[ch] = prev;
  state[E + ch] = last;
}

// Router: probs[m][e] = sigmoid(xn[m] . wr[e]) (wr [NE][E] f32). Grid (NE * 16, T), local (16, 1).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void lfm_router(__global const float* xn, __global const float* wr, int E, int NE, __global float* probs) {
  const int e = get_global_id(0) / 16, m = get_global_id(1), lane = get_sub_group_local_id();
  __global const float* a = xn + (size_t)m * E;
  __global const float* b = wr + (size_t)e * E;
  float s = 0.0f;
  for (int i = lane; i < E; i += 16) s += a[i] * b[i];
  s = sub_group_reduce_add(s);
  if (lane == 0) probs[(size_t)m * NE + e] = 1.0f / (1.0f + exp(-s));
}

// dst[i] = src[idx[off + i]] (rows of E floats; i < n valid, the rest zero). Grid (E, np).
__kernel void lfm_gather(__global const float* src, __global const int* idx, int off, int n, int E, __global float* dst) {
  const int c = get_global_id(0), i = get_global_id(1);
  dst[(size_t)i * E + c] = i < n ? src[(size_t)idx[off + i] * E + c] : 0.0f;
}

// acc[idx[off + i]] += wts[off + i] * d[i]. Grid (E, n): the rows of one expert are distinct.
__kernel void lfm_scatter(__global const float* d, __global const int* idx, __global const float* wts, int off, int E,
                          __global float* acc) {
  const int c = get_global_id(0), i = get_global_id(1);
  acc[(size_t)idx[off + i] * E + c] += wts[off + i] * d[(size_t)i * E + c];
}

// x += acc. Grid (E, T).
__kernel void lfm_add(__global float* x, __global const float* acc, int E) {
  const int c = get_global_id(0), m = get_global_id(1);
  x[(size_t)m * E + c] += acc[(size_t)m * E + c];
}

// One token: x += w0 d[0] + w1 d[1] + w2 d[2] + w3 d[3] (d: four rows of E floats). Grid (E).
__kernel void lfm_combine4(__global float* x, __global const float* d, int E, float w0, float w1, float w2, float w3) {
  const int c = get_global_id(0);
  x[c] += w0 * d[c] + w1 * d[E + c] + w2 * d[2 * E + c] + w3 * d[3 * E + c];
}
