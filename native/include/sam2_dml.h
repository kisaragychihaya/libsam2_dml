/* sam2_dml.h - C ABI for the SAM2 (Hiera-Small) ONNX Runtime + DirectML DLL.
 *
 * This header is the contract shared with the pure-Go FFI client (syscall.NewLazyDLL).
 * Field order/types in sam2_decode_result_t are mapped 1:1 by unsafe.Pointer on the
 * Go side - do not reorder/rename without updating goqc/main.go.
 *
 * Threading: every handle is internally serialized with a mutex. A handle may be used
 * from different threads, but concurrent calls on the same handle run one at a time.
 * Error text is per-thread and stays valid until the next call on that thread.
 */
#ifndef SAM2_DML_H
#define SAM2_DML_H

#include <stdint.h>

#ifdef _WIN32
#  ifdef SAM2_DML_BUILD
#    define SAM2_API __declspec(dllexport)
#  else
#    define SAM2_API __declspec(dllimport)
#  endif
#else
#  define SAM2_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sam2_encoder sam2_encoder_t;
typedef struct sam2_decoder sam2_decoder_t;
typedef struct sam2_embedding sam2_embedding_t;

/* Thread-local, NUL-terminated, plain ASCII. Never free. Valid until the next
 * sam2_* call on the same thread. Returns "" when no error is pending.
 * Note: when sam2_*_create() succeeds but had to fall back to the CPU provider,
 * this still holds the DirectML failure reason. */
SAM2_API const char* sam2_get_last_error(void);

/* model_path_utf8: UTF-8 path to sam2_hiera_small.encoder.onnx
 * device_id: adapter/device index, 0 = default device, must be >= 0.
 * Performs one dummy warmup inference so the first real call is not penalized
 * by provider graph compilation. Returns NULL on failure (see sam2_get_last_error). */
SAM2_API sam2_encoder_t* sam2_encoder_create(const char* model_path_utf8, int device_id);
SAM2_API void sam2_encoder_destroy(sam2_encoder_t* enc);

/* The execution provider attached at build time (e.g. "DmlExecutionProvider",
 * "CUDAExecutionProvider") when it attached successfully, otherwise
 * "CPUExecutionProvider". Returns "" for a NULL handle. */
SAM2_API const char* sam2_encoder_provider(const sam2_encoder_t* enc);

/* rgb: RGB24 pixels, row stride in bytes (pass width*3 for tightly packed).
 * The pixels are bilinearly resized to 1024x1024, scaled to [0,1] and normalized
 * with the ImageNet mean/std, then fed as NCHW float32.
 * The returned embedding is owned by the DLL and may be released with
 * sam2_embedding_destroy; it is independent of the encoder handle. */
SAM2_API sam2_embedding_t* sam2_encoder_encode(sam2_encoder_t* enc,
                                               const unsigned char* rgb,
                                               int width, int height, int stride);
SAM2_API void sam2_embedding_destroy(sam2_embedding_t* emb);

SAM2_API sam2_decoder_t* sam2_decoder_create(const char* model_path_utf8, int device_id);
SAM2_API void sam2_decoder_destroy(sam2_decoder_t* dec);
SAM2_API const char* sam2_decoder_provider(const sam2_decoder_t* dec);

/* masks: [num_masks][mask_h][mask_w] float32 logits, row-major.
 * iou:   [num_masks] float32 predicted IoU scores.
 * Both pointers are owned by the result handle. No thresholding / upsampling is
 * done here: the caller decides (mask_h/mask_w are 1/4 of the encoder size). */
typedef struct {
    float* masks;
    float* iou;
    int32_t num_masks;
    int32_t mask_h;
    int32_t mask_w;
} sam2_decode_result_t;

/* points_xy: [n_points][2] original-image pixel coordinates (x, y).
 * labels:    1 = positive point, 0 = negative point, 2 = box top-left, 3 = box bottom-right.
 * The coordinates are scaled into the 1024x1024 encoder space; mask_input is
 * all-zero and has_mask_input is 0, so no previous mask is used. */
SAM2_API sam2_decode_result_t* sam2_decoder_decode(sam2_decoder_t* dec,
                                                   const sam2_embedding_t* emb,
                                                   const float* points_xy,
                                                   const int32_t* labels,
                                                   int32_t n_points);
SAM2_API void sam2_decode_result_destroy(sam2_decode_result_t* res);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SAM2_DML_H */
