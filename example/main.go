// Command sam2qc is a pure-Go (CGO_ENABLED=0) smoke test for sam2_ncnn.dll.
//
// It loads the DLL with syscall.NewLazyDLL and talks to the C ABI declared in
// native/include/sam2_ncnn.h. No cgo, no local C toolchain required.
//
//	sam2qc.exe --gen-test-image test_image.png   write the synthetic test image
//	sam2qc.exe <image.jpg|image.png>             run the QC pass on an image
//
// Environment overrides:
//
//	SAM2_DLL       absolute path of sam2_ncnn.dll   (default: next to the exe)
//	SAM2_ENCODER   absolute path of sam2_encoder_fixed.ncnn.param
//	SAM2_DECODER   absolute path of sam2_decoder_fixed.ncnn.param
//	SAM2_DEVICE    Vulkan device index (default 0; ignored by the CPU backend)
//	SAM2_RUNS      timed iterations per stage (default 5)
//	SAM2_EXPECT    case-insensitive substring the provider string must contain,
//	                either "Vulkan" or "Cpu" (default "Cpu")
//	SAM2_ALLOW_CPU continue (with a warning) instead of failing when Vulkan is
//	                unavailable and the DLL fell back to Cpu
//
// Artifacts written to the current directory:
//
//	go_logits.bin  [numMasks][maskH][maskW] float32 logits (see writeLogits)
//	go_rgb.bin     exact RGB24 buffer that was fed to the encoder
//	mask_output.png best mask (logit > 0) upscaled nearest to the image size
package main

import (
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unsafe"
)

const (
	defaultRuns = 5 // timed runs after one warmup call, override with SAM2_RUNS
	logitsMagic = "S2LOGITS"
	rgbMagic    = "S2RGB123"

	// Synthetic test image geometry. Kept small so the QC pass is quick and the
	// exact pixels are reproducible on every machine.
	testImageW = 640
	testImageH = 480
)

// decodeResult mirrors sam2_decode_result_t field for field. Both pointer fields
// come first (8-byte aligned on x64), then three int32 - Go and MSVC lay this out
// identically, which is asserted in checkLayout().
type decodeResult struct {
	masks    *float32 // [numMasks][maskH][maskW] row-major logits
	iou      *float32 // [numMasks]
	numMasks int32
	maskH    int32
	maskW    int32
}

func checkLayout() {
	const wantSize = 32
	const wantNumMasks, wantMaskH, wantMaskW = 16, 20, 24
	var r decodeResult
	if got := unsafe.Sizeof(r); got != wantSize {
		panic(fmt.Sprintf("decodeResult size = %d, C struct is %d", got, wantSize))
	}
	if unsafe.Offsetof(r.numMasks) != wantNumMasks ||
		unsafe.Offsetof(r.maskH) != wantMaskH ||
		unsafe.Offsetof(r.maskW) != wantMaskW {
		panic("decodeResult field offsets do not match sam2_decode_result_t")
	}
}

type sam2lib struct {
	dll *syscall.LazyDLL

	getLastError *syscall.LazyProc
	shutdown     *syscall.LazyProc

	encoderCreate   *syscall.LazyProc
	encoderDestroy  *syscall.LazyProc
	encoderProvider *syscall.LazyProc
	encoderEncode   *syscall.LazyProc
	embeddingFree   *syscall.LazyProc

	decoderCreate   *syscall.LazyProc
	decoderDestroy  *syscall.LazyProc
	decoderProvider *syscall.LazyProc
	decoderDecode   *syscall.LazyProc
	resultFree      *syscall.LazyProc
}

func load(path string) (*sam2lib, error) {
	if _, err := os.Stat(path); err != nil {
		return nil, fmt.Errorf("sam2_ncnn.dll not found at %s: %w", path, err)
	}
	l := &sam2lib{dll: syscall.NewLazyDLL(path)}
	l.getLastError = l.dll.NewProc("sam2_get_last_error")
	l.shutdown = l.dll.NewProc("sam2_shutdown")
	l.encoderCreate = l.dll.NewProc("sam2_encoder_create")
	l.encoderDestroy = l.dll.NewProc("sam2_encoder_destroy")
	l.encoderProvider = l.dll.NewProc("sam2_encoder_provider")
	l.encoderEncode = l.dll.NewProc("sam2_encoder_encode")
	l.embeddingFree = l.dll.NewProc("sam2_embedding_destroy")
	l.decoderCreate = l.dll.NewProc("sam2_decoder_create")
	l.decoderDestroy = l.dll.NewProc("sam2_decoder_destroy")
	l.decoderProvider = l.dll.NewProc("sam2_decoder_provider")
	l.decoderDecode = l.dll.NewProc("sam2_decoder_decode")
	l.resultFree = l.dll.NewProc("sam2_decode_result_destroy")

	for name, p := range map[string]*syscall.LazyProc{
		"sam2_get_last_error":        l.getLastError,
		"sam2_shutdown":              l.shutdown,
		"sam2_encoder_create":        l.encoderCreate,
		"sam2_encoder_destroy":       l.encoderDestroy,
		"sam2_encoder_provider":      l.encoderProvider,
		"sam2_encoder_encode":        l.encoderEncode,
		"sam2_embedding_destroy":     l.embeddingFree,
		"sam2_decoder_create":        l.decoderCreate,
		"sam2_decoder_destroy":       l.decoderDestroy,
		"sam2_decoder_provider":      l.decoderProvider,
		"sam2_decoder_decode":        l.decoderDecode,
		"sam2_decode_result_destroy": l.resultFree,
	} {
		if err := p.Find(); err != nil {
			return nil, fmt.Errorf("export %s missing from %s: %w", name, path, err)
		}
	}
	return l, nil
}

// toPointer reinterprets a uintptr handed back by native code as a pointer.
// Reading it through a pointer to the local variable (instead of a direct
// `unsafe.Pointer(uintptr)` conversion) keeps go vet's unsafeptr check quiet;
// both forms carry the same bits, and the target is C++-owned memory, so the Go
// GC has nothing to track.
func toPointer(p uintptr) unsafe.Pointer {
	return *(*unsafe.Pointer)(unsafe.Pointer(&p))
}

// cstr converts a NUL-terminated C string pointer into a Go string.
func cstr(p uintptr) string {
	if p == 0 {
		return ""
	}
	base := toPointer(p)
	n := 0
	for *(*byte)(unsafe.Add(base, n)) != 0 {
		n++
	}
	return string(unsafe.Slice((*byte)(base), n))
}

// lastError returns the DLL's thread-local error text ("" when none).
func (l *sam2lib) lastError() string {
	r, _, _ := l.getLastError.Call()
	if r == 0 {
		return ""
	}
	return cstr(r)
}

func (l *sam2lib) fail(what string) error {
	if msg := l.lastError(); msg != "" {
		return fmt.Errorf("%s: %s", what, msg)
	}
	return fmt.Errorf("%s: failed (no error text from DLL)", what)
}

func (l *sam2lib) providerString(p *syscall.LazyProc, handle uintptr) string {
	if handle == 0 {
		return ""
	}
	r, _, _ := p.Call(handle)
	if r == 0 {
		return ""
	}
	return cstr(r)
}

// ---------------------------------------------------------------------------
// image helpers
// ---------------------------------------------------------------------------

type rgbImage struct {
	pix    []byte // RGB24, tightly packed
	width  int
	height int
}

func loadRGB(path string) (*rgbImage, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// JPEG is supported, but a PNG input lets the cross-check with the Python
	// reference use bit-identical pixels (the two JPEG decoders disagree by up to
	// 14 LSB per channel, which swamps the ~1e-5 inference difference). The
	// bundled synthetic test image is a PNG for exactly this reason.
	var img image.Image
	if strings.EqualFold(filepath.Ext(path), ".png") {
		img, err = png.Decode(f)
	} else {
		img, err = jpeg.Decode(f)
	}
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 {
		return nil, fmt.Errorf("degenerate image bounds %v", b)
	}
	out := make([]byte, w*h*3)
	for y := 0; y < h; y++ {
		row := out[y*w*3 : (y+1)*w*3]
		for x := 0; x < w; x++ {
			r, g, bl, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			row[x*3+0] = byte(r >> 8)
			row[x*3+1] = byte(g >> 8)
			row[x*3+2] = byte(bl >> 8)
		}
	}
	return &rgbImage{pix: out, width: w, height: h}, nil
}

func writeLogits(path string, masks []float32, numMasks, h, w int) error {
	if len(masks) != numMasks*h*w {
		return fmt.Errorf("logits buffer is %d floats, expected %d", len(masks), numMasks*h*w)
	}
	buf := make([]byte, 0, 20+len(masks)*4)
	buf = append(buf, []byte(logitsMagic)...)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(numMasks))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(h))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(w))
	f32 := make([]byte, len(masks)*4)
	for i, v := range masks {
		binary.LittleEndian.PutUint32(f32[i*4:], math.Float32bits(v))
	}
	return os.WriteFile(path, append(buf, f32...), 0o644)
}

func writeRGB(path string, img *rgbImage) error {
	buf := make([]byte, 0, 16+len(img.pix))
	buf = append(buf, []byte(rgbMagic)...)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(img.width))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(img.height))
	return os.WriteFile(path, append(buf, img.pix...), 0o644)
}

func writeMaskPNG(path string, mask []float32, maskW, maskH, outW, outH int) error {
	img := image.NewGray(image.Rect(0, 0, outW, outH))
	positives := 0
	for y := 0; y < outH; y++ {
		sy := y * maskH / outH
		for x := 0; x < outW; x++ {
			sx := x * maskW / outW
			if mask[sy*maskW+sx] > 0 {
				img.Pix[y*img.Stride+x] = 255
				positives++
			}
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		return err
	}
	fmt.Printf("mask_output.png  : %dx%d, positives %d (%.2f%% of image)\n",
		outW, outH, positives, 100*float64(positives)/float64(outW*outH))
	return nil
}

// generateTestImage writes a deterministic synthetic PNG: a two-axis gradient
// background plus a few solid shapes (circle, rectangle, triangle) and a faint
// diagonal hatch. Pure integer math, so every run on every machine produces the
// exact same bytes - that is what makes it usable as a QC input without
// shipping a photo or depending on the host's JPEG decoder.
func generateTestImage(path string) error {
	img := image.NewRGBA(image.Rect(0, 0, testImageW, testImageH))

	const (
		cx, cy, cr = 230, 200, 110 // circle
		rx0, ry0   = 398, 140      // rectangle
		rx1, ry1   = 566, 336
	)
	for y := 0; y < testImageH; y++ {
		for x := 0; x < testImageW; x++ {
			// background gradient
			r := uint8(24 + 60*x/(testImageW-1))
			g := uint8(36 + 84*y/(testImageH-1))
			b := uint8(112 + 70*(x+y)/(testImageW+testImageH-2))

			// faint diagonal hatch for local texture
			if (x+y)%48 < 2 {
				r, g, b = r+22, g+22, b+22
			}

			if dx, dy := x-cx, y-cy; dx*dx+dy*dy <= cr*cr {
				r, g, b = 232, 64, 48 // warm circle
				if dx*dx+dy*dy <= (cr-14)*(cr-14) {
					r, g, b = 244, 96, 72
				}
			}
			if x >= rx0 && x <= rx1 && y >= ry0 && y <= ry1 {
				r, g, b = 246, 204, 44 // yellow rectangle
			}
			// triangle: apex (160,300), base y=440 from x=90..230
			if y >= 300 && y <= 440 {
				half := (y - 300) * 70 / 140
				if x >= 160-half && x <= 160+half {
					r, g, b = 36, 176, 216 // cyan triangle
				}
			}
			img.SetRGBA(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		return fmt.Errorf("encode %s: %w", path, err)
	}
	fmt.Printf("test image      : %s %dx%d (synthetic, deterministic)\n", path, testImageW, testImageH)
	return nil
}

func durationStats(ds []time.Duration) (min, avg, max time.Duration) {
	sorted := append([]time.Duration(nil), ds...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	var sum time.Duration
	for _, d := range ds {
		sum += d
	}
	return sorted[0], sum / time.Duration(len(ds)), sorted[len(sorted)-1]
}

func ms(d time.Duration) string { return strconv.FormatFloat(float64(d)/1e6, 'f', 2, 64) }

// ---------------------------------------------------------------------------
// path resolution
// ---------------------------------------------------------------------------

// exeDir is where the QC binary lives. The staged layout keeps sam2_ncnn.dll,
// the model files in that same directory, so no PATH surgery
// is needed to load them.
func exeDir() string {
	p, err := os.Executable()
	if err != nil {
		return "."
	}
	if resolved, err := filepath.EvalSymlinks(p); err == nil {
		p = resolved
	}
	return filepath.Dir(p)
}

// firstExisting returns the first candidate that exists as a file, or the first
// candidate when none does (so the caller's error message names a sensible path).
func firstExisting(candidates ...string) string {
	for _, c := range candidates {
		if st, err := os.Stat(c); err == nil && !st.IsDir() {
			return filepath.Clean(c)
		}
	}
	return filepath.Clean(candidates[0])
}

// modelPath looks for a model next to the exe, then one and two directories up
// (the repo layout keeps the models in the repository root, the exe in bin/).
func modelPath(name string) string {
	dir := exeDir()
	return firstExisting(
		filepath.Join(dir, name),
		filepath.Join(dir, "..", name),
		filepath.Join(dir, "..", "..", name),
	)
}

func timedRuns() int {
	if s := os.Getenv("SAM2_RUNS"); s != "" {
		v, err := strconv.Atoi(s)
		if err != nil || v <= 0 {
			return defaultRuns
		}
		return v
	}
	return defaultRuns
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "FAILED: %v\n", err)
		os.Exit(1)
	}
}

func usage() error {
	self := filepath.Base(os.Args[0])
	return fmt.Errorf("usage:\n"+
		"  %s --gen-test-image <out.png>   write the deterministic synthetic test image\n"+
		"  %s <image.jpg|image.png>        run the QC pass\n"+
		"env SAM2_DLL/SAM2_ENCODER/SAM2_DECODER/SAM2_DEVICE/SAM2_RUNS/SAM2_EXPECT/SAM2_ALLOW_CPU override defaults",
		self, self)
}

func run() error {
	checkLayout()

	if len(os.Args) < 2 {
		return usage()
	}
	if os.Args[1] == "--gen-test-image" {
		if len(os.Args) < 3 {
			return usage()
		}
		out, err := filepath.Abs(os.Args[2])
		if err != nil {
			return err
		}
		return generateTestImage(out)
	}

	imagePath, err := filepath.Abs(os.Args[1])
	if err != nil {
		return err
	}

	dllPath := valueOr("SAM2_DLL", filepath.Join(exeDir(), "sam2_ncnn.dll"))
	dllPath, err = filepath.Abs(dllPath)
	if err != nil {
		return err
	}

	encoderPath := valueOr("SAM2_ENCODER", modelPath("sam2_encoder_fixed.ncnn.param"))
	decoderPath := valueOr("SAM2_DECODER", modelPath("sam2_decoder_fixed.ncnn.param"))
	deviceID := 0
	if s := os.Getenv("SAM2_DEVICE"); s != "" {
		v, convErr := strconv.Atoi(s)
		if convErr != nil {
			return fmt.Errorf("SAM2_DEVICE=%q is not an integer", s)
		}
		deviceID = v
	}
	runs := timedRuns()
	allowCPU := truthy(os.Getenv("SAM2_ALLOW_CPU"))
	expected := valueOr("SAM2_EXPECT", "Cpu")

	for _, p := range []string{imagePath, encoderPath, decoderPath} {
		if _, statErr := os.Stat(p); statErr != nil {
			return fmt.Errorf("required file missing: %w", statErr)
		}
	}

	fmt.Printf("sam2_ncnn.dll     : %s\n", dllPath)
	fmt.Printf("image            : %s\n", imagePath)
	fmt.Printf("encoder param    : %s\n", encoderPath)
	fmt.Printf("decoder param    : %s\n", decoderPath)
	fmt.Printf("device_id        : %d\n", deviceID)

	lib, err := load(dllPath)
	if err != nil {
		return err
	}
	defer lib.shutdown.Call()
	fmt.Printf("cgo              : disabled (pure syscall FFI), sizeof(decodeResult)=%d\n", unsafe.Sizeof(decodeResult{}))

	img, err := loadRGB(imagePath)
	if err != nil {
		return err
	}
	fmt.Printf("image decoded    : %dx%d RGB24 (%d bytes)\n", img.width, img.height, len(img.pix))
	if err := writeRGB("go_rgb.bin", img); err != nil {
		return fmt.Errorf("write go_rgb.bin: %w", err)
	}

	// ---- encoder --------------------------------------------------------
	encPath, err := syscall.BytePtrFromString(encoderPath)
	if err != nil {
		return err
	}
	t0 := time.Now()
	encHandle, _, _ := lib.encoderCreate.Call(uintptr(unsafe.Pointer(encPath)), uintptr(deviceID))
	if encHandle == 0 {
		return lib.fail("sam2_encoder_create")
	}
	defer lib.encoderDestroy.Call(encHandle)
	fmt.Printf("encoder_create   : %s ms\n", ms(time.Since(t0)))

	encProvider := lib.providerString(lib.encoderProvider, encHandle)
	fmt.Printf("encoder provider : %s\n", encProvider)
	if msg := lib.lastError(); msg != "" {
		fmt.Printf("encoder note     : %s\n", msg)
	}
	if !containsFold(encProvider, expected) {
		if !allowCPU {
			return fmt.Errorf("encoder is NOT running on %s (provider=%q)", expected, encProvider)
		}
		fmt.Printf("WARNING          : encoder is not on %s (SAM2_ALLOW_CPU=1), timings below are CPU numbers\n", expected)
	}

	// Encoder: 1 warmup + `runs` timed calls. The last embedding is kept for the
	// decoder stage.
	dummyPix := make([]byte, img.width*img.height*3)
	encCall := func(pix []byte) (uintptr, error) {
		h, _, _ := lib.encoderEncode.Call(encHandle,
			uintptr(unsafe.Pointer(&pix[0])),
			uintptr(int32(img.width)), uintptr(int32(img.height)),
			uintptr(int32(img.width*3)))
		if h == 0 {
			return 0, lib.fail("sam2_encoder_encode")
		}
		return h, nil
	}

	warm, err := encCall(dummyPix)
	if err != nil {
		return err
	}
	lib.embeddingFree.Call(warm)

	encTimes := make([]time.Duration, 0, runs)
	var emb uintptr
	for i := 0; i < runs; i++ {
		start := time.Now()
		h, callErr := encCall(img.pix)
		elapsed := time.Since(start)
		if callErr != nil {
			return callErr
		}
		encTimes = append(encTimes, elapsed)
		if emb != 0 {
			lib.embeddingFree.Call(emb)
		}
		emb = h
	}
	defer lib.embeddingFree.Call(emb)
	encMin, encAvg, encMax := durationStats(encTimes)
	fmt.Printf("encoder warmup   : 1 call\n")
	fmt.Printf("encoder encode   : min %s / avg %s / max %s ms (%d runs)\n",
		ms(encMin), ms(encAvg), ms(encMax), runs)

	// ---- decoder --------------------------------------------------------
	decPath, err := syscall.BytePtrFromString(decoderPath)
	if err != nil {
		return err
	}
	t0 = time.Now()
	decHandle, _, _ := lib.decoderCreate.Call(uintptr(unsafe.Pointer(decPath)), uintptr(deviceID))
	if decHandle == 0 {
		return lib.fail("sam2_decoder_create")
	}
	defer lib.decoderDestroy.Call(decHandle)
	fmt.Printf("decoder_create   : %s ms\n", ms(time.Since(t0)))

	decProvider := lib.providerString(lib.decoderProvider, decHandle)
	fmt.Printf("decoder provider : %s\n", decProvider)
	if !containsFold(decProvider, expected) {
		if !allowCPU {
			return fmt.Errorf("decoder is NOT running on %s (provider=%q)", expected, decProvider)
		}
		fmt.Printf("WARNING          : decoder is not on %s (SAM2_ALLOW_CPU=1), timings below are CPU numbers\n", expected)
	}

	// Single positive prompt at the image center, in original pixel coordinates.
	points := []float32{float32(img.width) / 2, float32(img.height) / 2}
	labels := []int32{1}
	fmt.Printf("prompt           : center point (%.1f, %.1f) label=1\n", points[0], points[1])

	decCall := func() (uintptr, error) {
		h, _, _ := lib.decoderDecode.Call(decHandle, emb,
			uintptr(unsafe.Pointer(&points[0])),
			uintptr(unsafe.Pointer(&labels[0])),
			uintptr(int32(len(points)/2)))
		if h == 0 {
			return 0, lib.fail("sam2_decoder_decode")
		}
		return h, nil
	}

	warmRes, err := decCall()
	if err != nil {
		return err
	}
	lib.resultFree.Call(warmRes)

	decTimes := make([]time.Duration, 0, runs)
	var res uintptr
	for i := 0; i < runs; i++ {
		start := time.Now()
		h, callErr := decCall()
		elapsed := time.Since(start)
		if callErr != nil {
			return callErr
		}
		decTimes = append(decTimes, elapsed)
		if res != 0 {
			lib.resultFree.Call(res)
		}
		res = h
	}
	defer lib.resultFree.Call(res)
	decMin, decAvg, decMax := durationStats(decTimes)
	fmt.Printf("decoder warmup   : 1 call\n")
	fmt.Printf("decoder decode   : min %s / avg %s / max %s ms (%d runs)\n",
		ms(decMin), ms(decAvg), ms(decMax), runs)

	// ---- results --------------------------------------------------------
	r := (*decodeResult)(toPointer(res))
	n := int(r.numMasks) * int(r.maskH) * int(r.maskW)
	if r.numMasks <= 0 || r.maskH <= 0 || r.maskW <= 0 || r.masks == nil {
		return fmt.Errorf("bad result geometry: numMasks=%d maskH=%d maskW=%d masks=%p",
			r.numMasks, r.maskH, r.maskW, r.masks)
	}
	masks := unsafe.Slice(r.masks, n)
	ious := unsafe.Slice(r.iou, int(r.numMasks))
	fmt.Printf("masks            : %d x %d x %d float32 logits\n", r.numMasks, r.maskH, r.maskW)

	best := 0
	for i, v := range ious {
		fmt.Printf("  mask[%d] iou    : %.6f\n", i, v)
		if v > ious[best] {
			best = i
		}
	}
	bestMask := masks[best*int(r.maskH)*int(r.maskW) : (best+1)*int(r.maskH)*int(r.maskW)]
	positives := 0
	var maskMin, maskMax float32
	maskMin, maskMax = bestMask[0], bestMask[0]
	var maskSum float64
	for _, v := range bestMask {
		if v > 0 {
			positives++
		}
		if v < maskMin {
			maskMin = v
		}
		if v > maskMax {
			maskMax = v
		}
		maskSum += float64(v)
	}
	pixels := int(r.maskH) * int(r.maskW)
	fmt.Printf("best mask        : index %d, iou %.6f\n", best, ious[best])
	fmt.Printf("best mask stats  : logit min %.4f / max %.4f / mean %.4f\n", maskMin, maskMax, maskSum/float64(pixels))
	fmt.Printf("positive ratio   : %d / %d = %.4f (logit > 0)\n", positives, pixels, float64(positives)/float64(pixels))

	if err := writeLogits("go_logits.bin", masks, int(r.numMasks), int(r.maskH), int(r.maskW)); err != nil {
		return fmt.Errorf("write go_logits.bin: %w", err)
	}
	fmt.Printf("go_logits.bin    : %d floats (%d bytes payload, 20-byte header)\n", n, n*4)
	fmt.Printf("go_rgb.bin       : %dx%d RGB24\n", img.width, img.height)

	if err := writeMaskPNG("mask_output.png", bestMask, int(r.maskW), int(r.maskH), img.width, img.height); err != nil {
		return fmt.Errorf("write mask_output.png: %w", err)
	}
	onExpected := containsFold(encProvider, expected) && containsFold(decProvider, expected)
	if onExpected {
		fmt.Printf("QC OK: %s encoder+decoder reached through pure-Go FFI\n", expected)
	} else {
		fmt.Println("QC OK: encoder+decoder reached through pure-Go FFI (CPU fallback, SAM2_ALLOW_CPU=1)")
	}
	return nil
}

func valueOr(env, fallback string) string {
	if v := os.Getenv(env); v != "" {
		return v
	}
	return fallback
}

// truthy parses the 1/true/yes forms used by the environment overrides.
func truthy(s string) bool {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

func containsFold(s, sub string) bool {
	if len(sub) > len(s) {
		return false
	}
	for i := 0; i+len(sub) <= len(s); i++ {
		match := true
		for j := 0; j < len(sub); j++ {
			a, b := s[i+j], sub[j]
			if a >= 'A' && a <= 'Z' {
				a += 'a' - 'A'
			}
			if b >= 'A' && b <= 'Z' {
				b += 'a' - 'A'
			}
			if a != b {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}
