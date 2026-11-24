// sam2_ncnn.cpp - SAM2 Hiera-Small encoder/decoder on ncnn (Vulkan or CPU).
//
// The C ABI in include/sam2_ncnn.h is consumed by the pure-Go FFI client:
//   - models are the pnnx-converted sam2_*_fixed.ncnn.param/.bin pairs
//   - point/prompt embedding is computed here, driven by constants from
//     sam2_prompt.bin (next to the decoder .param) - no torch, no .pt
//   - provider strings are "Vulkan" / "Cpu"
//
// Handles are thread-safe but serialized: concurrent calls on the same handle
// run one at a time. ncnn paths are plain char*; keep them ASCII.

#include "sam2_ncnn.h"
#if !defined(SAM2_VULKAN)
#define SAM2_VULKAN 0
#endif

#include <gpu.h>
#include <mat.h>
#include <net.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr int kInputSize = 1024;       // encoder input H/W
constexpr int kNumInputChannels = 3;
constexpr int kEmbedCh = 256;          // FPN/embedding channels (neck output)
constexpr int kEmbedHw = 64;           // vision feature map H/W
constexpr int kSparseDim = 256;
constexpr const char* kProviderCpu = "Cpu";
#if SAM2_VULKAN
constexpr const char* kProviderPrimary = "Vulkan";
#else
constexpr const char* kProviderPrimary = "Cpu";
#endif

// ImageNet normalization constants used by anylabeling/SAM2.
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

thread_local std::string g_last_error;

void set_error(const std::string& msg) {
    g_last_error = msg.size() > 2000 ? msg.substr(0, 2000) : msg;
}

void clear_error() { g_last_error.clear(); }

// ---------------------------------------------------------------------------
// host-side prompt constants (sam2_prompt.bin)
// ---------------------------------------------------------------------------
struct PromptConsts {
    std::vector<float> pe_gauss;      // (2, F) frozen random projection
    int gauss_f = 0;
    std::vector<float> point_table;   // (5, 256): [not_a_point, label0..3]
    std::vector<float> no_mask;       // (256)
    std::vector<float> image_pe;      // (256, 64, 64) CHW
};

uint32_t read_u32(FILE* f, const char* what) {
    uint32_t v = 0;
    if (fread(&v, sizeof(v), 1, f) != 1) {
        set_error(std::string("sam2_prompt.bin: truncated header at ") + what);
        return 0;
    }
    return v;
}

bool load_prompt_consts(const std::string& prompt_path, PromptConsts& pc) {
    FILE* f = nullptr;
    fopen_s(&f, prompt_path.c_str(), "rb");
    if (f == nullptr) {
        set_error("sam2_prompt.bin: cannot open " + prompt_path);
        return false;
    }
    bool ok = false;
    do {
        uint32_t magic = read_u32(f, "magic");
        uint32_t version = read_u32(f, "version");
        uint32_t count = read_u32(f, "count");
        if (magic != 0x42503253u || version != 1) {  // 'S2PB'
            set_error("sam2_prompt.bin: bad magic/version");
            break;
        }
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t nl = read_u32(f, "name len");
            if (nl == 0 || nl > 63) break;
            char name[64] = {0};
            if (fread(name, 1, nl, f) != nl) break;
            uint32_t nd = read_u32(f, "ndim");
            if (nd == 0 || nd > 4) break;
            uint32_t dims[4] = {0, 0, 0, 0};
            size_t total = 1;
            bool dims_ok = true;
            for (uint32_t d = 0; d < nd; ++d) {
                dims[d] = read_u32(f, "dim");
                if (dims[d] == 0) dims_ok = false;
                total *= dims[d];
            }
            if (!dims_ok) break;
            std::vector<float> data(total);
            if (fread(data.data(), sizeof(float), total, f) != total) break;
            const std::string nm(name, nl);
            if (nm == "pe_gauss") {
                pc.pe_gauss = std::move(data);
                pc.gauss_f = static_cast<int>(dims[1]);
            } else if (nm == "point_table") {
                pc.point_table = std::move(data);
            } else if (nm == "no_mask") {
                pc.no_mask = std::move(data);
            } else if (nm == "image_pe") {
                pc.image_pe = std::move(data);
            }
            if (i + 1 == count) ok = true;
        }
    } while (false);
    fclose(f);
    if (ok) {
        if (pc.pe_gauss.empty() || pc.point_table.size() != 5 * kSparseDim ||
            pc.no_mask.size() != kSparseDim ||
            pc.image_pe.size() != static_cast<size_t>(kEmbedCh) * kEmbedHw * kEmbedHw) {
            set_error("sam2_prompt.bin: missing or wrongly shaped arrays");
            ok = false;
        }
    }
    return ok;
}

// Derive <base>.bin from <base>.param.
bool bin_path_from_param(const char* param_path, std::string& bin_path) {
    bin_path = param_path;
    const std::string suffix = ".param";
    if (bin_path.size() < suffix.size() ||
        bin_path.compare(bin_path.size() - suffix.size(), suffix.size(), suffix) != 0) {
        set_error(std::string("model path must end in .param: ") + param_path);
        return false;
    }
    bin_path.replace(bin_path.size() - suffix.size(), suffix.size(), ".bin");
    return true;
}

std::string dir_of(const std::string& path) {
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? std::string() : path.substr(0, pos + 1);
}

// Copy an ncnn Mat (c, h, w) into a packed CHW vector.
void mat_to_chw(const ncnn::Mat& m, std::vector<float>& out) {
    out.resize(static_cast<size_t>(m.c) * m.h * m.w);
    for (int c = 0; c < m.c; ++c) {
        const ncnn::Mat plane = m.channel(c);
        std::memcpy(out.data() + static_cast<size_t>(c) * m.h * m.w,
                    plane.data, sizeof(float) * static_cast<size_t>(m.h) * m.w);
    }
}

// Configure an ncnn::Net for the selected backend. Returns false (with
// fallback_reason filled) when the requested Vulkan device cannot be used.
bool configure_net(ncnn::Net& net, int device_id, std::string& provider,
                   std::string& fallback_reason) {
    // Packing changes the memory layout of extracted Mats; this DLL reads
    // channel planes directly, so keep everything unpacked.
    net.opt.use_packing_layout = false;
#if SAM2_VULKAN
    if (ncnn::get_gpu_count() > 0 && device_id < ncnn::get_gpu_count()) {
        net.opt.use_vulkan_compute = true;
        net.set_vulkan_device(device_id);
        // fp32 end-to-end: fp16 storage costs real accuracy in the wide
        // stage-4 GEMMs (measured ~14% peak error vs the PyTorch reference)
        net.opt.use_fp16_packed = false;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        provider = kProviderPrimary;
        return true;
    }
    fallback_reason = "Vulkan backend requested but device_id=" +
                      std::to_string(device_id) + " is unavailable (" +
                      std::to_string(ncnn::get_gpu_count()) +
                      " GPU(s) visible); using Cpu";
#else
    (void)device_id;
    (void)fallback_reason;
#endif
    net.opt.use_vulkan_compute = false;
    provider = kProviderCpu;
    return true;
}

// Bilinear resize (half-pixel centers, the OpenCV INTER_LINEAR grid) + /255 +
// ImageNet mean/std + HWC->CHW; identical recipe to the ORT build.
void preprocess_rgb(const unsigned char* rgb, int width, int height, int stride,
                    std::vector<float>& out) {
    out.resize(static_cast<size_t>(kNumInputChannels) * kInputSize * kInputSize);
    const float scale_x = static_cast<float>(width) / static_cast<float>(kInputSize);
    const float scale_y = static_cast<float>(height) / static_cast<float>(kInputSize);

    for (int dy = 0; dy < kInputSize; ++dy) {
        float sy = (static_cast<float>(dy) + 0.5f) * scale_y - 0.5f;
        if (sy < 0.0f) sy = 0.0f;
        if (sy > static_cast<float>(height - 1)) sy = static_cast<float>(height - 1);
        const int y0 = static_cast<int>(sy);
        const int y1 = std::min(y0 + 1, height - 1);
        const float fy = sy - static_cast<float>(y0);
        const unsigned char* row0 = rgb + static_cast<size_t>(y0) * stride;
        const unsigned char* row1 = rgb + static_cast<size_t>(y1) * stride;

        for (int dx = 0; dx < kInputSize; ++dx) {
            float sx = (static_cast<float>(dx) + 0.5f) * scale_x - 0.5f;
            if (sx < 0.0f) sx = 0.0f;
            if (sx > static_cast<float>(width - 1)) sx = static_cast<float>(width - 1);
            const int x0 = static_cast<int>(sx);
            const int x1 = std::min(x0 + 1, width - 1);
            const float fx = sx - static_cast<float>(x0);

            const float w00 = (1.0f - fx) * (1.0f - fy);
            const float w01 = fx * (1.0f - fy);
            const float w10 = (1.0f - fx) * fy;
            const float w11 = fx * fy;

            const unsigned char* p00 = row0 + static_cast<size_t>(x0) * 3;
            const unsigned char* p01 = row0 + static_cast<size_t>(x1) * 3;
            const unsigned char* p10 = row1 + static_cast<size_t>(x0) * 3;
            const unsigned char* p11 = row1 + static_cast<size_t>(x1) * 3;

            const size_t idx = static_cast<size_t>(dy) * kInputSize + dx;
            for (int c = 0; c < kNumInputChannels; ++c) {
                const float v = p00[c] * w00 + p01[c] * w01 + p10[c] * w10 + p11[c] * w11;
                out[idx + static_cast<size_t>(c) * kInputSize * kInputSize] =
                    (v / 255.0f - kMean[c]) / kStd[c];
            }
        }
    }
}

// Point embedding: port of sam2 PromptEncoder._embed_points(pad=True).
// pts are already in the 1024x1024 encoder frame. rows_out = n + 1 (pad row).
void embed_points(const PromptConsts& pc, const float* pts, const int32_t* labels,
                  int n, std::vector<float>& out) {
    const int rows = n + 1;
    const int f = pc.gauss_f;
    out.assign(static_cast<size_t>(rows) * kSparseDim, 0.0f);
    for (int i = 0; i < rows; ++i) {
        const bool pad = (i == n);
        float* row = out.data() + static_cast<size_t>(i) * kSparseDim;
        if (pad) {
            std::memcpy(row, pc.point_table.data(), sizeof(float) * kSparseDim);
            continue;
        }
        const float x = pts[i * 2 + 0] + 0.5f;   // shift to pixel center
        const float y = pts[i * 2 + 1] + 0.5f;
        const float nx = x / static_cast<float>(kInputSize) * 2.0f - 1.0f;
        const float ny = y / static_cast<float>(kInputSize) * 2.0f - 1.0f;
        const float* g = pc.pe_gauss.data();
        for (int j = 0; j < f; ++j) {
            const float angx = 2.0f * 3.14159265358979f * (nx * g[j] + ny * g[f + j]);
            row[j] = std::sin(angx);
            row[f + j] = std::cos(angx);
        }
        const int li = labels[i] + 1;   // -1..3 -> 0..4
        if (li >= 0 && li < 5) {
            const float* te = pc.point_table.data() + static_cast<size_t>(li) * kSparseDim;
            for (int j = 0; j < kSparseDim; ++j) row[j] += te[j];
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// errors (exported; every other sam2_* call may set this text)
// ---------------------------------------------------------------------------
const char* sam2_get_last_error(void) { return g_last_error.c_str(); }

void sam2_shutdown(void) {
#if SAM2_VULKAN
    ncnn::destroy_gpu_instance();
#endif
}

// ---------------------------------------------------------------------------
// encoder
// ---------------------------------------------------------------------------
struct sam2_encoder {
    std::mutex mu;
    ncnn::Net net;
    std::string provider;
    int device_id = 0;
    std::vector<float> scratch;
};

sam2_encoder_t* sam2_encoder_create(const char* model_path_utf8, int device_id) {
    clear_error();
    if (model_path_utf8 == nullptr || model_path_utf8[0] == '\0') {
        set_error("sam2_encoder_create: model_path is NULL or empty");
        return nullptr;
    }
    if (device_id < 0) {
        set_error("sam2_encoder_create: device_id must be >= 0");
        return nullptr;
    }
    try {
        std::unique_ptr<sam2_encoder> enc(new sam2_encoder());
        enc->device_id = device_id;
        std::string bin_path;
        if (!bin_path_from_param(model_path_utf8, bin_path)) return nullptr;
        std::string fallback_reason;
        configure_net(enc->net, device_id, enc->provider, fallback_reason);
        if (enc->net.load_param(model_path_utf8) != 0 ||
            enc->net.load_model(bin_path.c_str()) != 0) {
            set_error(std::string("sam2_encoder_create: failed to load ") +
                      model_path_utf8 + " / " + bin_path);
            return nullptr;
        }

        // Warmup: compile the graph with a zero image so the first real call
        // is not charged for it.
        std::vector<float> zeros(static_cast<size_t>(kNumInputChannels) * kInputSize * kInputSize, 0.0f);
        ncnn::Mat in(kInputSize, kInputSize, kNumInputChannels, zeros.data());
        ncnn::Extractor ex = enc->net.create_extractor();
        ex.input("in0", in);
        ncnn::Mat warm;
        if (ex.extract("out2", warm) != 0) {
            set_error("sam2_encoder_create: warmup inference failed");
            return nullptr;
        }
        if (enc->provider == kProviderPrimary) clear_error();
        else if (!fallback_reason.empty()) set_error(fallback_reason);
        return enc.release();
    } catch (const std::exception& e) {
        set_error(std::string("sam2_encoder_create: ") + e.what());
        return nullptr;
    }
}

void sam2_encoder_destroy(sam2_encoder_t* enc) { delete enc; }

const char* sam2_encoder_provider(const sam2_encoder_t* enc) {
    if (enc == nullptr) return "";
    return enc->provider.c_str();
}

struct sam2_embedding {
    int image_w = 0;
    int image_h = 0;
    std::vector<float> hr0;   // (256, 256, 256) CHW
    std::vector<float> hr1;   // (256, 128, 128) CHW
    std::vector<float> emb;   // (256, 64, 64)   CHW
};

sam2_embedding_t* sam2_encoder_encode(sam2_encoder_t* enc, const unsigned char* rgb,
                                      int width, int height, int stride) {
    clear_error();
    if (enc == nullptr) {
        set_error("sam2_encoder_encode: NULL encoder handle");
        return nullptr;
    }
    if (rgb == nullptr) {
        set_error("sam2_encoder_encode: NULL pixel buffer");
        return nullptr;
    }
    if (width <= 0 || height <= 0) {
        set_error("sam2_encoder_encode: invalid image size " + std::to_string(width) + "x" +
                  std::to_string(height));
        return nullptr;
    }
    if (stride < width * 3) {
        set_error("sam2_encoder_encode: stride " + std::to_string(stride) +
                  " is smaller than width*3 = " + std::to_string(width * 3));
        return nullptr;
    }
    try {
        std::lock_guard<std::mutex> lock(enc->mu);
        preprocess_rgb(rgb, width, height, stride, enc->scratch);

        ncnn::Mat in(kInputSize, kInputSize, kNumInputChannels, enc->scratch.data());
        ncnn::Extractor ex = enc->net.create_extractor();
        ex.input("in0", in);
        ncnn::Mat m0, m1, m2;
        if (ex.extract("out0", m0) != 0 || ex.extract("out1", m1) != 0 ||
            ex.extract("out2", m2) != 0) {
            set_error("sam2_encoder_encode: extractor failed");
            return nullptr;
        }

        std::unique_ptr<sam2_embedding> emb(new sam2_embedding());
        emb->image_w = width;
        emb->image_h = height;
        mat_to_chw(m0, emb->hr0);
        mat_to_chw(m1, emb->hr1);
        mat_to_chw(m2, emb->emb);

        // SAM2_DEBUG_DUMP=<dir>: write the raw encoder inputs/outputs for
        // offline comparison against the reference pipeline (debug aid).
        if (const char* dump = std::getenv("SAM2_DEBUG_DUMP")) {
            auto write = [dump](const char* name, const std::vector<float>& v, int c, int h, int w) {
                std::string path = std::string(dump) + "/" + name;
                FILE* f = nullptr;
                fopen_s(&f, path.c_str(), "wb");
                if (!f) return;
                int32_t hdr[4] = {c, h, w, 0};
                fwrite(hdr, sizeof(int32_t), 4, f);
                fwrite(v.data(), sizeof(float), v.size(), f);
                fclose(f);
            };
            write("dbg_in.bin", enc->scratch, kNumInputChannels, kInputSize, kInputSize);
            write("dbg_hr0.bin", emb->hr0, 256, 256, 256);
            write("dbg_hr1.bin", emb->hr1, 256, 128, 128);
            write("dbg_emb.bin", emb->emb, 256, 64, 64);
        }
        return emb.release();
    } catch (const std::exception& e) {
        set_error(std::string("sam2_encoder_encode: ") + e.what());
        return nullptr;
    }
}

void sam2_embedding_destroy(sam2_embedding_t* emb) { delete emb; }

// ---------------------------------------------------------------------------
// decoder
// ---------------------------------------------------------------------------
struct sam2_decoder {
    std::mutex mu;
    ncnn::Net net;
    std::string provider;
    int device_id = 0;
    PromptConsts pc;
};

sam2_decoder_t* sam2_decoder_create(const char* model_path_utf8, int device_id) {
    clear_error();
    if (model_path_utf8 == nullptr || model_path_utf8[0] == '\0') {
        set_error("sam2_decoder_create: model_path is NULL or empty");
        return nullptr;
    }
    if (device_id < 0) {
        set_error("sam2_decoder_create: device_id must be >= 0");
        return nullptr;
    }
    try {
        std::unique_ptr<sam2_decoder> dec(new sam2_decoder());
        dec->device_id = device_id;
        std::string bin_path;
        if (!bin_path_from_param(model_path_utf8, bin_path)) return nullptr;
        if (!load_prompt_consts(dir_of(model_path_utf8) + "sam2_prompt.bin", dec->pc)) {
            return nullptr;
        }
        // The decoder net is pinned to the Cpu backend in every build. Its
        // Vulkan execution corrupts multi-point prompts (README "Known
        // issues"), and the decoder is the cheap half of the pipeline, so Cpu
        // costs little (decode ~75ms vs ~38ms) while staying correct for any
        // number of points. The encoder keeps configure_net's backend.
        dec->net.opt.use_packing_layout = false;
        dec->net.opt.use_vulkan_compute = false;
        dec->provider = kProviderCpu;
        if (dec->net.load_param(model_path_utf8) != 0 ||
            dec->net.load_model(bin_path.c_str()) != 0) {
            set_error(std::string("sam2_decoder_create: failed to load ") +
                      model_path_utf8 + " / " + bin_path);
            return nullptr;
        }

        // Warmup with a near-zero embedding and one center point. All-zero
        // inputs crash some ncnn builds on this graph (NaN paths in the
        // reference-run layers), so use a tiny constant instead.
        sam2_embedding dummy;
        dummy.image_w = kInputSize;
        dummy.image_h = kInputSize;
        const float kWarm = 1e-3f;
        dummy.emb.assign(static_cast<size_t>(kEmbedCh) * kEmbedHw * kEmbedHw, kWarm);
        dummy.hr0.assign(static_cast<size_t>(kEmbedCh) * (kInputSize / 4) * (kInputSize / 4), kWarm);
        dummy.hr1.assign(static_cast<size_t>(kEmbedCh) * (kInputSize / 8) * (kInputSize / 8), kWarm);
        const float pt[2] = {512.0f, 512.0f};
        const int32_t lbl[1] = {1};
        std::unique_ptr<sam2_decode_result_t> warm(
            sam2_decoder_decode(dec.get(), &dummy, pt, lbl, 1));
        if (warm == nullptr) return nullptr;
        sam2_decode_result_destroy(warm.release());
        // The decoder is Cpu-only (see above): no Vulkan fallback to explain.
        clear_error();
        return dec.release();
    } catch (const std::exception& e) {
        set_error(std::string("sam2_decoder_create: ") + e.what());
        return nullptr;
    }
}

void sam2_decoder_destroy(sam2_decoder_t* dec) { delete dec; }

const char* sam2_decoder_provider(const sam2_decoder_t* dec) {
    if (dec == nullptr) return "";
    return dec->provider.c_str();
}

sam2_decode_result_t* sam2_decoder_decode(sam2_decoder_t* dec, const sam2_embedding_t* emb,
                                          const float* points_xy, const int32_t* labels,
                                          int32_t n_points) {
    clear_error();
    if (dec == nullptr || emb == nullptr) {
        set_error("sam2_decoder_decode: NULL decoder or embedding handle");
        return nullptr;
    }
    if (n_points <= 0 || points_xy == nullptr || labels == nullptr) {
        set_error("sam2_decoder_decode: invalid points");
        return nullptr;
    }
    if (emb->image_w <= 0 || emb->image_h <= 0) {
        set_error("sam2_decoder_decode: embedding has no source image size");
        return nullptr;
    }
    try {
        std::lock_guard<std::mutex> lock(dec->mu);

        // Scale original-image pixels into the 1024x1024 encoder space, then
        // embed (pixel-center shift + random-projection PE + label tables).
        const float sx = static_cast<float>(kInputSize) / static_cast<float>(emb->image_w);
        const float sy = static_cast<float>(kInputSize) / static_cast<float>(emb->image_h);
        std::vector<float> pts(static_cast<size_t>(n_points) * 2);
        for (int32_t i = 0; i < n_points; ++i) {
            pts[static_cast<size_t>(i) * 2 + 0] = points_xy[static_cast<size_t>(i) * 2 + 0] * sx;
            pts[static_cast<size_t>(i) * 2 + 1] = points_xy[static_cast<size_t>(i) * 2 + 1] * sy;
        }
        std::vector<float> sparse;
        embed_points(dec->pc, pts.data(), labels, n_points, sparse);
        const int rows = n_points + 1;

        // dense = no_mask_embed broadcast to (256, 64, 64)
        std::vector<float> dense(static_cast<size_t>(kEmbedCh) * kEmbedHw * kEmbedHw);
        for (int c = 0; c < kEmbedCh; ++c) {
            std::fill(dense.begin() + static_cast<size_t>(c) * kEmbedHw * kEmbedHw,
                      dense.begin() + static_cast<size_t>(c + 1) * kEmbedHw * kEmbedHw,
                      dec->pc.no_mask[static_cast<size_t>(c)]);
        }

        ncnn::Extractor ex = dec->net.create_extractor();
        ex.input("in0", ncnn::Mat(kEmbedHw, kEmbedHw, kEmbedCh,
                                  const_cast<float*>(emb->emb.data())));
        ex.input("in1", ncnn::Mat(kEmbedHw, kEmbedHw, kEmbedCh,
                                  const_cast<float*>(dec->pc.image_pe.data())));
        ex.input("in2", ncnn::Mat(kSparseDim, rows, sparse.data()));
        ex.input("in3", ncnn::Mat(kEmbedHw, kEmbedHw, kEmbedCh, dense.data()));
        ex.input("in4", ncnn::Mat(kInputSize / 4, kInputSize / 4, kEmbedCh,
                                  const_cast<float*>(emb->hr0.data())));
        ex.input("in5", ncnn::Mat(kInputSize / 8, kInputSize / 8, kEmbedCh,
                                  const_cast<float*>(emb->hr1.data())));

        ncnn::Mat m_masks, m_iou;
        if (ex.extract("out0", m_masks) != 0 || ex.extract("out1", m_iou) != 0) {
            set_error("sam2_decoder_decode: extractor failed");
            return nullptr;
        }

        if (const char* dump = std::getenv("SAM2_DEBUG_DUMP")) {
            auto write = [dump](const char* name, const float* p, size_t n) {
                std::string path = std::string(dump) + "/" + name;
                FILE* f = nullptr;
                fopen_s(&f, path.c_str(), "wb");
                if (!f) return;
                int32_t hdr[4] = {1, 1, static_cast<int32_t>(n), 0};
                fwrite(hdr, sizeof(int32_t), 4, f);
                fwrite(p, sizeof(float), n, f);
                fclose(f);
            };
            write("dbg_sparse.bin", sparse.data(), sparse.size());
            write("dbg_dense.bin", dense.data(), dense.size());
            write("dbg_in_emb.bin", emb->emb.data(), emb->emb.size());
        }

        std::unique_ptr<sam2_decode_result_t> res(new sam2_decode_result_t());
        res->num_masks = 1;
        res->mask_h = m_masks.h;
        res->mask_w = m_masks.w;
        const size_t n = static_cast<size_t>(res->num_masks) * res->mask_h * res->mask_w;
        res->masks = static_cast<float*>(std::malloc(n * sizeof(float)));
        const ncnn::Mat mask_plane = m_masks.channel(0);
        std::memcpy(res->masks, mask_plane.data, n * sizeof(float));
        res->iou = static_cast<float*>(std::malloc(sizeof(float)));
        std::memcpy(res->iou, m_iou.data, sizeof(float));
        return res.release();
    } catch (const std::exception& e) {
        set_error(std::string("sam2_decoder_decode: ") + e.what());
        return nullptr;
    }
}

void sam2_decode_result_destroy(sam2_decode_result_t* res) {
    if (res == nullptr) return;
    std::free(res->masks);
    std::free(res->iou);
    delete res;
}
