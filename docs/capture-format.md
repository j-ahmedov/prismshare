# Capture format — the contract between the phone and the analysis

`prism-share-capture` (Android) writes it. `prism_share.ingest` (Python) reads it.
Both sides implement this document. Neither side invents fields.

Schema version 1.

---

## Where a run lives on the phone

    /sdcard/Android/data/io.github.ahmedov.prismshare.capture/files/captures/<run_id>/

Pull it with:

    adb pull /sdcard/Android/data/io.github.ahmedov.prismshare.capture/files/captures/<run_id> ./data/

## What a run folder contains

    <run_id>/
      manifest.json
      frames.jsonl
      frame_0000_y.png      Y plane,  full resolution,     8-bit greyscale, lossless
      frame_0000_u.png      U plane,  quarter resolution,  8-bit greyscale, lossless
      frame_0000_v.png      V plane,  quarter resolution,  8-bit greyscale, lossless
      frame_0000_rgb.png    RGB,      full resolution,     8-bit, lossless
      frame_0001_y.png
      ...

Index is zero-padded to four digits. The U and V planes are at their **native**
subsampled resolution and are never upsampled on the phone — that is the point of
saving them.

On the JPEG comparison path the only file per frame is `frame_NNNN.jpg`. That path
is a baseline for comparison and is never used for a headline measurement.

Capture resolution is not fixed. The app requests the largest `YUV_420_888` size
the device offers and records it in `camera.capture_resolution`. **Python reads the
resolution from the manifest and never assumes one.**

---

## manifest.json — one object per run

```json
{
  "schema_version": 1,
  "run_id": "d07_a30_lux200_oled",
  "captured_at": "2026-09-14T16:22:31+02:00",
  "notes": "free text from the operator",

  "device": {
    "manufacturer": "samsung",
    "model": "SM-A346E",
    "android_release": "16",
    "api_level": 36,
    "build_fingerprint": "samsung/a34xnsxx/a34x:16/..."
  },

  "camera": {
    "camera_id": "0",
    "hardware_level": "LIMITED",
    "capabilities": ["BACKWARD_COMPATIBLE", "BURST_CAPTURE"],
    "active_array_size": [4000, 3000],
    "pixel_array_size": [4080, 3060],
    "color_filter_arrangement": "GRBG",
    "capture_template": "TEMPLATE_STILL_CAPTURE",
    "capture_resolution": [4000, 3000]
  },

  "control": {
    "protocol": "locked_auto",
    "reference_frame_id": "ref_v1_1024",
    "clip_low_pct": 0.02,
    "clip_high_pct": 0.00,
    "frozen": {
      "sensor_sensitivity": 320,
      "sensor_exposure_time_ns": 16666667,
      "sensor_frame_duration_ns": 33333333,
      "color_correction_gains": [1.94, 1.0, 1.0, 1.72],
      "color_correction_transform": [[1,0,0],[0,1,0],[0,0,1]],
      "lens_focus_distance": 2.44,
      "lens_aperture": 1.8,
      "ae_exposure_compensation": 0
    },
    "applied_modes": {
      "tonemap": "FAST",
      "noise_reduction": "OFF",
      "edge": "OFF",
      "antibanding": "OFF",
      "video_stabilization": "OFF",
      "optical_stabilization": "OFF"
    }
  },

  "color": {
    "pixel_format": "YUV_420_888",
    "yuv_matrix": "BT601",
    "yuv_range": "limited",
    "yuv_range_source": "assumed",
    "dataspace": null
  },

  "frames": {
    "requested": 200,
    "written": 200,
    "tainted": 0
  },

  "app": { "version": "0.1.0", "git_commit": "a1b2c3d" }
}
```

Notes on specific fields:

- `control.protocol` — `"locked_auto"` or `"manual"`. On a device without
  `MANUAL_SENSOR` it is always `"locked_auto"`.
- `control.frozen.*` — every value **read back from `CaptureResult`** at lock time,
  never the requested value. Devices clamp silently.
- `color.yuv_range_source` — `"read"` when taken from `Image.getDataSpace()` on API
  34+, `"assumed"` otherwise. The analysis must be able to tell the difference.
- `camera.hardware_level` and `capabilities` use the Camera2 constant names as
  strings, not their integer values. `INFO_SUPPORTED_HARDWARE_LEVEL` is not
  numerically ordered and integers invite an ordering bug.

---

## frames.jsonl — one JSON object per line, one line per frame

```json
{"index":0,"file_stem":"frame_0000","sensor_timestamp_ns":88123456789,"ae_state":"LOCKED","awb_state":"LOCKED","af_state":"FOCUSED_LOCKED","sensor_sensitivity":320,"sensor_exposure_time_ns":16666667,"color_correction_gains":[1.94,1.0,1.0,1.72],"rolling_shutter_skew_ns":28000000,"tainted":false,"taint_reasons":[]}
```

`file_stem` is the explicit link from record to files — append `_y.png`, `_u.png`,
`_v.png`, `_rgb.png`. Do not reconstruct filenames from `index` by convention.

`taint_reasons` is a **closed set**. Any value outside it is an error, not a new
category:

| value | meaning |
|---|---|
| `ae_state` | `CONTROL_AE_STATE` was not `LOCKED` |
| `awb_state` | `CONTROL_AWB_STATE` was not `LOCKED` |
| `sensitivity_drift` | `SENSOR_SENSITIVITY` differed from the frozen value |
| `exposure_drift` | `SENSOR_EXPOSURE_TIME` differed from the frozen value |
| `gains_drift` | `COLOR_CORRECTION_GAINS` differed from the frozen values |

`tainted` is `true` if and only if `taint_reasons` is non-empty.

---

## The join: the phone records the camera, the laptop records the stimulus

The phone has no idea which codec configuration was on screen. That lives on the
laptop, in the run definition under `experiments/`.

**`run_id` is the join key**, and it is typed by hand on the phone. Ingest must
therefore verify that a laptop-side run definition exists with a matching `run_id`
and refuse the run otherwise. A typo at 11pm in a dark room is the likeliest way
this experiment loses a day, and it costs three lines to catch.

---

## Rules ingest must enforce

1. Reject any run whose `schema_version` is unknown.
2. Reject any run with no matching laptop-side run definition.
3. Drop every frame with `tainted: true`, and record how many were dropped and
   the count per `taint_reasons` value.
4. Refuse the whole run when `frames.tainted / frames.written > 0.01`, with a
   message saying it must be repeated. A run where the camera lock failed is not
   partially usable.
5. Warn when `frames.written != frames.requested`.
6. Read resolution, YUV matrix and YUV range from the manifest. Never assume.
