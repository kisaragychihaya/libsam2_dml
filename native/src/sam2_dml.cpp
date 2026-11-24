// sam2_dml.cpp - SAM2 Hiera-Small encoder/decoder on ONNX Runtime, built against
// one execution provider selected at build time (CMake -DSAM2_ORT_EP=...).
//
// Layout of the exported C ABI lives in include/sam2_dml.h. Everything below is
// the implementation detail: an Ort::Session per handle, a mutex per handle, and
// the preprocessing that mirrors anylabeling's sam2_onnx.py (bilinear resize to
// 1024x1024, /255, ImageNet mean/std, NCHW float32).

#include "sam2_dml.h"

#include <windows.h>

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#if defined(SAM2_EP_DIRECTML)
#include <dml_provider_factory.h>   // ships only with the DirectML package
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

namespace {

constexpr int kInputSize = 1024;       // encoder input H/W
constexpr int kNumInputChannels = 3;
constexpr const char* kProviderCpu = "CPUExecutionProvider";

// ImageNet normalization constants used by anylabeling/SAM2.
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

thread_local std::string g_last_error;

void set_error(const std::string& msg) {
    g_last_error = msg.size() > 2000 ? msg.substr(0, 2000) : msg;
}

void clear_error() { g_last_error.clear(); }

// Build-time EP selection (CMake -DSAM2_ORT_EP=..., default directml). The EP is
// attached through the versioned OrtApi table where possible so this file links
// against every onnxruntime import lib; on a runtime that lacks the EP the call
// returns an error status, which make_session_options turns into the documented
// CPU fallback. DirectML has no vtable entry for its legacy attach call, so that
// one stays a direct import - compiled in only for directml builds, which link
// the DirectML package anyway.
#if defined(SAM2_EP_CUDA)
constexpr const char* kProviderPrimary = "CUDAExecutionProvider";
constexpr const char* kEpName = "CUDA";

OrtStatus* attach_primary_ep(Ort::SessionOptions& options, int device_id) {
    const OrtApi& api = Ort::GetApi();
    OrtCUDAProviderOptionsV2* opts = nullptr;
    OrtStatus* status = api.CreateCUDAProviderOptions(&opts);
    if (status == nullptr) {
        const char* keys[] = {"device_id"};
        const char* values[] = {std::to_string(device_id).c_str()};
        status = api.UpdateCUDAProviderOptions(opts, keys, values, 1);
        if (status == nullptr) {
            status = api.SessionOptionsAppendExecutionProvider_CUDA_V2(options, opts);
        }
        api.ReleaseCUDAProviderOptions(opts);
    }
    return status;
}
#elif defined(SAM2_EP_TENSORRT)
constexpr const char* kProviderPrimary = "TensorrtExecutionProvider";
constexpr const char* kEpName = "TensorRT";

OrtStatus* attach_primary_ep(Ort::SessionOptions& options, int device_id) {
    const OrtApi& api = Ort::GetApi();
    OrtTensorRTProviderOptionsV2* opts = nullptr;
    OrtStatus* status = api.CreateTensorRTProviderOptions(&opts);
    if (status == nullptr) {
        const char* keys[] = {"device_id"};
        const char* values[] = {std::to_string(device_id).c_str()};
        status = api.UpdateTensorRTProviderOptions(opts, keys, values, 1);
        if (status == nullptr) {
            status = api.SessionOptionsAppendExecutionProvider_TensorRT_V2(options, opts);
        }
        api.ReleaseTensorRTProviderOptions(opts);
    }
    return status;
}
#elif defined(SAM2_EP_CPU)
constexpr const char* kProviderPrimary = "CPUExecutionProvider";
constexpr const char* kEpName = "CPU";

OrtStatus* attach_primary_ep(Ort::SessionOptions&, int) { return nullptr; }  // CPU is appended below
#else  // SAM2_EP_DIRECTML
constexpr const char* kProviderPrimary = "DmlExecutionProvider";
constexpr const char* kEpName = "DirectML";

OrtStatus* attach_primary_ep(Ort::SessionOptions& options, int device_id) {
    return OrtSessionOptionsAppendExecutionProvider_DML(options, device_id);
}
#endif

std::wstring utf8_to_wide(const char* utf8) {
    if (utf8 == nullptr) return std::wstring();
    const int len = ::MultiByteToWideChar(CP_UTF8, 0, utf8, -1, nullptr, 0);
    if (len <= 0) return std::wstring();
    std::wstring out(static_cast<size_t>(len - 1), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, utf8, -1, out.data(), len);
    return out;
}

std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
}

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) {
        if (d < 0) return -1;
        n *= d;
    }
    return n;
}

// Collects output names in the model's declared order.
std::vector<std::string> output_names(Ort::Session& session, Ort::AllocatorWithDefaultOptions& alloc) {
    std::vector<std::string> names;
    const size_t n = session.GetOutputCount();
    names.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        Ort::AllocatedStringPtr p = session.GetOutputNameAllocated(i, alloc);
        names.emplace_back(p.get());
    }
    return names;
}

std::vector<std::string> input_names(Ort::Session& session, Ort::AllocatorWithDefaultOptions& alloc) {
    std::vector<std::string> names;
    const size_t n = session.GetInputCount();
    names.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        Ort::AllocatedStringPtr p = session.GetInputNameAllocated(i, alloc);
        names.emplace_back(p.get());
    }
    return names;
}

// Builds the session options: the build-time EP first (falls back to CPU),
// then CPU. Also records which provider actually got attached.
std::string make_session_options(Ort::SessionOptions& options, int device_id,
                                 const std::string& tag) {
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    std::string provider = kProviderCpu;
    if (device_id < 0) {
        set_error(tag + ": device_id must be >= 0");
        return std::string();
    }

    OrtStatus* status = attach_primary_ep(options, device_id);
    if (status != nullptr) {
        const char* msg = Ort::GetApi().GetErrorMessage(status);
        const std::string detail = msg ? msg : "unknown error";
        Ort::GetApi().ReleaseStatus(status);
        // Keep going on CPU; the failure is reported by sam2_*_provider() as
        // "CPUExecutionProvider" and surfaces in the warmup log if it also fails.
        set_error(tag + ": " + kEpName + " EP unavailable (device_id=" + std::to_string(device_id) +
                  "): " + detail);
        provider = kProviderCpu;
    } else {
        provider = kProviderPrimary;
    }
    options.AppendExecutionProvider_CPU(1);
    return provider;
}

std::unique_ptr<Ort::Session> create_session(Ort::Env& env, Ort::SessionOptions& options,
                                             const std::wstring& wide_path,
                                             const std::string& tag) {
    try {
        return std::make_unique<Ort::Session>(env, wide_path.c_str(), options);
    } catch (const Ort::Exception& e) {
        set_error(tag + ": failed to load model: " + e.what());
        return nullptr;
    } catch (const std::exception& e) {
        set_error(tag + ": failed to load model: " + e.what());
        return nullptr;
    }
}

// Bilinear resize (half-pixel centers, i.e. the same sampling grid OpenCV's
// cv2.resize(INTER_LINEAR) uses) + /255 + ImageNet mean/std + HWC->CHW.
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

}  // namespace

// ---------------------------------------------------------------------------
// encoder
// ---------------------------------------------------------------------------
struct sam2_encoder {
    std::mutex mu;
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "sam2_dml"};
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo mem{nullptr};
    std::string provider;
    std::string in_name;
    std::vector<std::string> out_names;
    std::vector<const char*> out_name_ptrs;
    std::vector<float> scratch;
};

// ---------------------------------------------------------------------------
// embedding
// ---------------------------------------------------------------------------
struct sam2_embedding {
    std::vector<float> hr0;   // high_res_feats_0
    std::vector<float> hr1;   // high_res_feats_1
    std::vector<float> emb;   // image_embed
    std::vector<int64_t> hr0_shape;
    std::vector<int64_t> hr1_shape;
    std::vector<int64_t> emb_shape;
    int32_t image_w = 0;
    int32_t image_h = 0;
};

// ---------------------------------------------------------------------------
// decoder
// ---------------------------------------------------------------------------
struct sam2_decoder {
    std::mutex mu;
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "sam2_dml"};
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo mem{nullptr};
    std::string provider;
    std::vector<std::string> in_names;
    std::vector<std::string> out_names;
};

extern "C" {

const char* sam2_get_last_error(void) { return g_last_error.c_str(); }

// --------------------------------------------------------------------------
sam2_encoder_t* sam2_encoder_create(const char* model_path_utf8, int device_id) {
    clear_error();
    if (model_path_utf8 == nullptr || model_path_utf8[0] == '\0') {
        set_error("sam2_encoder_create: model_path is NULL or empty");
        return nullptr;
    }
    try {
        std::unique_ptr<sam2_encoder> enc(new sam2_encoder());
        enc->mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::SessionOptions options;
        enc->provider = make_session_options(options, device_id, "sam2_encoder_create");
        if (enc->provider.empty()) return nullptr;

        const std::wstring wide_path = utf8_to_wide(model_path_utf8);
        enc->session = create_session(enc->env, options, wide_path, "sam2_encoder_create");
        if (!enc->session) return nullptr;

        Ort::AllocatorWithDefaultOptions alloc;
        std::vector<std::string> ins = input_names(*enc->session, alloc);
        enc->out_names = output_names(*enc->session, alloc);
        if (ins.empty() || enc->out_names.size() < 3) {
            set_error("sam2_encoder_create: unexpected model I/O (inputs=" +
                      std::to_string(ins.size()) + ", outputs=" +
                      std::to_string(enc->out_names.size()) + ")");
            return nullptr;
        }
        enc->in_name = ins[0];
        enc->out_name_ptrs.reserve(enc->out_names.size());
        for (const std::string& s : enc->out_names) enc->out_name_ptrs.push_back(s.c_str());

        // Warmup: compile the provider graph with a zero image so the first real
        // encode() call is not charged for graph compilation.
        std::vector<float> zeros(static_cast<size_t>(kNumInputChannels) * kInputSize * kInputSize, 0.0f);
        const std::array<int64_t, 4> shape{1, kNumInputChannels, kInputSize, kInputSize};
        Ort::Value input = Ort::Value::CreateTensor<float>(
            enc->mem, zeros.data(), zeros.size(), shape.data(), shape.size());
        const char* in_names[1] = {enc->in_name.c_str()};
        std::vector<Ort::Value> outs = enc->session->Run(Ort::RunOptions{nullptr}, in_names, &input, 1,
                                                         enc->out_name_ptrs.data(),
                                                         enc->out_name_ptrs.size());
        if (outs.size() < 3) {
            set_error("sam2_encoder_create: warmup returned " + std::to_string(outs.size()) + " outputs");
            return nullptr;
        }
        // Keep the primary-EP failure text if we fell back to CPU so callers can
        // read sam2_get_last_error() to learn why.
        if (enc->provider == kProviderPrimary) clear_error();
        return enc.release();
    } catch (const Ort::Exception& e) {
        set_error(std::string("sam2_encoder_create: ") + e.what());
        return nullptr;
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

        const std::array<int64_t, 4> shape{1, kNumInputChannels, kInputSize, kInputSize};
        Ort::Value input = Ort::Value::CreateTensor<float>(
            enc->mem, enc->scratch.data(), enc->scratch.size(), shape.data(), shape.size());
        const char* in_names[1] = {enc->in_name.c_str()};
        std::vector<Ort::Value> outs = enc->session->Run(
            Ort::RunOptions{nullptr}, in_names, &input, 1, enc->out_name_ptrs.data(),
            enc->out_name_ptrs.size());
        if (outs.size() < 3) {
            set_error("sam2_encoder_encode: unexpected output count " + std::to_string(outs.size()));
            return nullptr;
        }

        std::unique_ptr<sam2_embedding> emb(new sam2_embedding());
        emb->image_w = width;
        emb->image_h = height;

        std::vector<float>* dst[3] = {&emb->hr0, &emb->hr1, &emb->emb};
        std::vector<int64_t>* dst_shape[3] = {&emb->hr0_shape, &emb->hr1_shape, &emb->emb_shape};
        for (size_t i = 0; i < 3; ++i) {
            Ort::TensorTypeAndShapeInfo info = outs[i].GetTensorTypeAndShapeInfo();
            *dst_shape[i] = info.GetShape();
            const int64_t n = numel(*dst_shape[i]);
            if (n < 0) {
                set_error("sam2_encoder_encode: output " + std::to_string(i) +
                          " has dynamic shape " + shape_to_string(*dst_shape[i]));
                return nullptr;
            }
            dst[i]->resize(static_cast<size_t>(n));
            std::memcpy(dst[i]->data(), outs[i].GetTensorMutableData<float>(),
                        sizeof(float) * static_cast<size_t>(n));
        }
        return emb.release();
    } catch (const Ort::Exception& e) {
        set_error(std::string("sam2_encoder_encode: ") + e.what());
        return nullptr;
    } catch (const std::exception& e) {
        set_error(std::string("sam2_encoder_encode: ") + e.what());
        return nullptr;
    }
}

void sam2_embedding_destroy(sam2_embedding_t* emb) { delete emb; }

// --------------------------------------------------------------------------
sam2_decoder_t* sam2_decoder_create(const char* model_path_utf8, int device_id) {
    clear_error();
    if (model_path_utf8 == nullptr || model_path_utf8[0] == '\0') {
        set_error("sam2_decoder_create: model_path is NULL or empty");
        return nullptr;
    }
    try {
        std::unique_ptr<sam2_decoder> dec(new sam2_decoder());
        dec->mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::SessionOptions options;
        dec->provider = make_session_options(options, device_id, "sam2_decoder_create");
        if (dec->provider.empty()) return nullptr;

        const std::wstring wide_path = utf8_to_wide(model_path_utf8);
        dec->session = create_session(dec->env, options, wide_path, "sam2_decoder_create");
        if (!dec->session) return nullptr;

        Ort::AllocatorWithDefaultOptions alloc;
        dec->in_names = input_names(*dec->session, alloc);
        dec->out_names = output_names(*dec->session, alloc);
        if (dec->in_names.size() < 7 || dec->out_names.size() < 2) {
            set_error("sam2_decoder_create: unexpected model I/O (inputs=" +
                      std::to_string(dec->in_names.size()) + ", outputs=" +
                      std::to_string(dec->out_names.size()) + ")");
            return nullptr;
        }

        // Warmup with the same prompt shape the real call uses (1 point, no mask).
        const std::array<int64_t, 4> emb_shape{1, 256, 64, 64};
        const std::array<int64_t, 4> hr0_shape{1, 32, 256, 256};
        const std::array<int64_t, 4> hr1_shape{1, 64, 128, 128};
        const std::array<int64_t, 3> pt_shape{1, 1, 2};
        const std::array<int64_t, 2> lbl_shape{1, 1};
        const std::array<int64_t, 4> mask_shape{1, 1, 256, 256};
        const std::array<int64_t, 1> has_shape{1};

        std::vector<float> emb_data(256 * 64 * 64, 0.0f);
        std::vector<float> hr0_data(32 * 256 * 256, 0.0f);
        std::vector<float> hr1_data(64 * 128 * 128, 0.0f);
        std::vector<float> pt_data{0.0f, 0.0f};
        std::vector<float> lbl_data{1.0f};
        std::vector<float> mask_data(256 * 256, 0.0f);
        std::vector<float> has_data{0.0f};

        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(dec->mem, emb_data.data(), emb_data.size(),
                                                         emb_shape.data(), emb_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(dec->mem, hr0_data.data(), hr0_data.size(),
                                                         hr0_shape.data(), hr0_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(dec->mem, hr1_data.data(), hr1_data.size(),
                                                         hr1_shape.data(), hr1_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(dec->mem, pt_data.data(), pt_data.size(),
                                                         pt_shape.data(), pt_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(dec->mem, lbl_data.data(), lbl_data.size(),
                                                         lbl_shape.data(), lbl_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(dec->mem, mask_data.data(), mask_data.size(),
                                                         mask_shape.data(), mask_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(dec->mem, has_data.data(), has_data.size(),
                                                         has_shape.data(), has_shape.size()));

        std::vector<const char*> in_ptrs;
        in_ptrs.reserve(dec->in_names.size());
        for (const std::string& s : dec->in_names) in_ptrs.push_back(s.c_str());
        std::vector<const char*> out_ptrs;
        out_ptrs.reserve(dec->out_names.size());
        for (const std::string& s : dec->out_names) out_ptrs.push_back(s.c_str());

        std::vector<Ort::Value> outs = dec->session->Run(Ort::RunOptions{nullptr}, in_ptrs.data(),
                                                         inputs.data(), inputs.size(),
                                                         out_ptrs.data(), out_ptrs.size());
        if (outs.size() < 2) {
            set_error("sam2_decoder_create: warmup returned " + std::to_string(outs.size()) + " outputs");
            return nullptr;
        }
        if (dec->provider == kProviderPrimary) clear_error();
        return dec.release();
    } catch (const Ort::Exception& e) {
        set_error(std::string("sam2_decoder_create: ") + e.what());
        return nullptr;
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
    if (dec == nullptr) {
        set_error("sam2_decoder_decode: NULL decoder handle");
        return nullptr;
    }
    if (emb == nullptr) {
        set_error("sam2_decoder_decode: NULL embedding handle");
        return nullptr;
    }
    if (n_points <= 0) {
        set_error("sam2_decoder_decode: n_points must be > 0, got " + std::to_string(n_points));
        return nullptr;
    }
    if (points_xy == nullptr || labels == nullptr) {
        set_error("sam2_decoder_decode: NULL points_xy or labels");
        return nullptr;
    }
    if (emb->image_w <= 0 || emb->image_h <= 0) {
        set_error("sam2_decoder_decode: embedding has no source image size");
        return nullptr;
    }
    try {
        std::lock_guard<std::mutex> lock(dec->mu);

        // Prompts: scale original-image pixels into the 1024x1024 encoder space.
        const float sx = static_cast<float>(kInputSize) / static_cast<float>(emb->image_w);
        const float sy = static_cast<float>(kInputSize) / static_cast<float>(emb->image_h);
        std::vector<float> pt(static_cast<size_t>(n_points) * 2);
        for (int32_t i = 0; i < n_points; ++i) {
            pt[static_cast<size_t>(i) * 2 + 0] = points_xy[static_cast<size_t>(i) * 2 + 0] * sx;
            pt[static_cast<size_t>(i) * 2 + 1] = points_xy[static_cast<size_t>(i) * 2 + 1] * sy;
        }
        std::vector<float> lbl(static_cast<size_t>(n_points));
        for (int32_t i = 0; i < n_points; ++i) lbl[static_cast<size_t>(i)] = static_cast<float>(labels[i]);

        const int32_t mask_hw = kInputSize / 4;
        std::vector<float> mask_input(static_cast<size_t>(mask_hw) * mask_hw, 0.0f);
        std::vector<float> has_mask{0.0f};

        const std::array<int64_t, 3> pt_shape{1, n_points, 2};
        const std::array<int64_t, 2> lbl_shape{1, n_points};
        const std::array<int64_t, 4> mask_shape{1, 1, mask_hw, mask_hw};
        const std::array<int64_t, 1> has_shape{1};

        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(dec->mem, const_cast<float*>(emb->emb.data()),
                                                         emb->emb.size(),
                                                         emb->emb_shape.data(), emb->emb_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(dec->mem, const_cast<float*>(emb->hr0.data()),
                                                         emb->hr0.size(), emb->hr0_shape.data(),
                                                         emb->hr0_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(dec->mem, const_cast<float*>(emb->hr1.data()),
                                                         emb->hr1.size(), emb->hr1_shape.data(),
                                                         emb->hr1_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(dec->mem, pt.data(), pt.size(),
                                                         pt_shape.data(), pt_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(dec->mem, lbl.data(), lbl.size(),
                                                         lbl_shape.data(), lbl_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(dec->mem, mask_input.data(), mask_input.size(),
                                                         mask_shape.data(), mask_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(dec->mem, has_mask.data(), has_mask.size(),
                                                         has_shape.data(), has_shape.size()));

        std::vector<const char*> in_ptrs;
        in_ptrs.reserve(dec->in_names.size());
        for (const std::string& s : dec->in_names) in_ptrs.push_back(s.c_str());
        std::vector<const char*> out_ptrs;
        out_ptrs.reserve(dec->out_names.size());
        for (const std::string& s : dec->out_names) out_ptrs.push_back(s.c_str());

        std::vector<Ort::Value> outs = dec->session->Run(Ort::RunOptions{nullptr}, in_ptrs.data(),
                                                         inputs.data(), inputs.size(),
                                                         out_ptrs.data(), out_ptrs.size());
        if (outs.size() < 2) {
            set_error("sam2_decoder_decode: unexpected output count " + std::to_string(outs.size()));
            return nullptr;
        }

        Ort::TensorTypeAndShapeInfo mask_info = outs[0].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> mask_shape_out = mask_info.GetShape();
        if (mask_shape_out.size() != 4) {
            set_error("sam2_decoder_decode: masks output rank is " +
                      std::to_string(mask_shape_out.size()) + ", expected 4 " +
                      shape_to_string(mask_shape_out));
            return nullptr;
        }
        const int64_t total = numel(mask_shape_out);
        if (total < 0) {
            set_error("sam2_decoder_decode: masks output has dynamic shape " +
                      shape_to_string(mask_shape_out));
            return nullptr;
        }

        Ort::TensorTypeAndShapeInfo iou_info = outs[1].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> iou_shape = iou_info.GetShape();
        const int64_t iou_n = numel(iou_shape);
        if (iou_n < 0) {
            set_error("sam2_decoder_decode: iou output has dynamic shape " + shape_to_string(iou_shape));
            return nullptr;
        }

        std::unique_ptr<sam2_decode_result_t> res(new sam2_decode_result_t());
        res->num_masks = static_cast<int32_t>(std::max<int64_t>(mask_shape_out[1], 0));
        res->mask_h = static_cast<int32_t>(std::max<int64_t>(mask_shape_out[2], 0));
        res->mask_w = static_cast<int32_t>(std::max<int64_t>(mask_shape_out[3], 0));
        res->masks = new float[static_cast<size_t>(total)];
        std::memcpy(res->masks, outs[0].GetTensorMutableData<float>(),
                    sizeof(float) * static_cast<size_t>(total));
        res->iou = new float[static_cast<size_t>(iou_n)];
        std::memcpy(res->iou, outs[1].GetTensorMutableData<float>(),
                    sizeof(float) * static_cast<size_t>(iou_n));
        return res.release();
    } catch (const Ort::Exception& e) {
        set_error(std::string("sam2_decoder_decode: ") + e.what());
        return nullptr;
    } catch (const std::exception& e) {
        set_error(std::string("sam2_decoder_decode: ") + e.what());
        return nullptr;
    }
}

void sam2_decode_result_destroy(sam2_decode_result_t* res) {
    if (res == nullptr) return;
    delete[] res->masks;
    delete[] res->iou;
    delete res;
}

}  // extern "C"
