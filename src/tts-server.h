#pragma once
// tts-server.h: shared OpenAI-compatible TTS HTTP core for the *.cpp ports.
//
// One synthesis context lives GPU resident for the process lifetime. The
// project tool fills a tts_backend adapter that wires its own ABI
// (qt_synthesize / ov_synthesize) into the generic sink, then calls
// tts_server_run. The HTTP layer, tuning, OAI parsing and audio framing
// are identical across projects ; only the adapter differs.
//
// Endpoints:
//   POST /v1/audio/speech   OAI text-to-speech
//   POST /v1/voices         upload voice reference (audio + text + id)
//   GET  /v1/models         single loaded model
//   GET  /v1/voices         list saved voices
//   GET  /health            liveness probe
//
// Audio out: response_format "pcm" streams s16le 24 kHz mono chunked as it
// is generated (real time), "wav" returns a one-shot RIFF file, "mp3" returns
// a one-shot MP3 file (encoded via ffmpeg). pcm is the
// default so streaming is on unless the client asks for a file.

#include "../vendor/cpp-httplib/httplib.h"
#include "audio-io.h"
#include "yyjson.h"

#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <filesystem>
#include <sstream>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

// Decode any audio format (mp3, ogg, m4a, etc.) to WAV buffer using ffmpeg
static std::string audio_decode_to_wav(const void * data, size_t size, int target_sr = 24000) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    char input_path_buf[256];
    char wav_path_buf[256];
    snprintf(input_path_buf, sizeof(input_path_buf), "/tmp/tts_audio_%ld_%ld", ts.tv_sec, ts.tv_nsec);
    snprintf(wav_path_buf, sizeof(wav_path_buf), "/tmp/tts_audio_%ld_%ld.wav", ts.tv_sec, ts.tv_nsec);
    std::string input_path = input_path_buf;
    std::string wav_path = wav_path_buf;
    
    // Write input data to temp file
    int fd_input = open(input_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd_input < 0) {
        fprintf(stderr, "[AudioDecode] Failed to create temp input file: %s\n", strerror(errno));
        return {};
    }
    
    if (write(fd_input, data, size) != (ssize_t)size) {
        close(fd_input);
        unlink(input_path.c_str());
        fprintf(stderr, "[AudioDecode] Failed to write temp input file\n");
        return {};
    }
    close(fd_input);
    
    // Convert to WAV using ffmpeg
    std::string cmd = "ffmpeg -y -loglevel error -i ";
    cmd += input_path;
    cmd += " -ar ";
    cmd += std::to_string(target_sr);
    cmd += " -ac 1 -f wav -acodec pcm_s16le ";
    cmd += wav_path;
    
    int ret = system(cmd.c_str());
    if (ret != 0) {
        unlink(input_path.c_str());
        unlink(wav_path.c_str());
        fprintf(stderr, "[AudioDecode] ffmpeg failed with code %d\n", ret);
        fprintf(stderr, "[AudioDecode] Command: %s\n", cmd.c_str());
        return {};
    }
    
    // Read WAV output
    FILE *fp = fopen(wav_path.c_str(), "rb");
    if (!fp) {
        unlink(input_path.c_str());
        unlink(wav_path.c_str());
        fprintf(stderr, "[AudioDecode] Failed to read WAV output\n");
        return {};
    }
    
    fseek(fp, 0, SEEK_END);
    long wav_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    std::string wav_data;
    wav_data.resize(wav_size);
    size_t read_size = fread(&wav_data[0], 1, wav_size, fp);
    fclose(fp);
    
    // Cleanup temp files
    unlink(input_path.c_str());
    unlink(wav_path.c_str());
    
    if (read_size != (size_t)wav_size) {
        fprintf(stderr, "[AudioDecode] Failed to read WAV data\n");
        return {};
    }
    
    return wav_data;
}

static std::string audio_encode_mp3(const float * audio, int T_audio, int sr) {
    std::string wav_data = audio_encode_wav(audio, T_audio, sr, WAV_S16);
    if (wav_data.empty()) {
        fprintf(stderr, "[MP3] Failed to encode WAV buffer\n");
        return {};
    }

    char temp_wav[] = "/tmp/tts_input_XXXXXX.wav";
    char temp_mp3[] = "/tmp/tts_output_XXXXXX.mp3";
    
    fprintf(stderr, "[MP3] Creating temp WAV file...\n");
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    char wav_path_buf[256];
    char mp3_path_buf[256];
    snprintf(wav_path_buf, sizeof(wav_path_buf), "/tmp/tts_%ld_%ld.wav", ts.tv_sec, ts.tv_nsec);
    snprintf(mp3_path_buf, sizeof(mp3_path_buf), "/tmp/tts_%ld_%ld.mp3", ts.tv_sec, ts.tv_nsec);
    std::string wav_path = wav_path_buf;
    std::string mp3_path = mp3_path_buf;
    
    int fd_wav = open(wav_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd_wav < 0) {
        fprintf(stderr, "[MP3] Failed to create WAV temp file: %s\n", strerror(errno));
        return {};
    }
    fprintf(stderr, "[MP3] WAV temp file: %s (fd=%d)\n", wav_path.c_str(), fd_wav);
    
    fprintf(stderr, "[MP3] Creating temp MP3 file...\n");
    int fd_mp3 = open(mp3_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd_mp3 < 0) {
        fprintf(stderr, "[MP3] Failed to create MP3 temp file: %s\n", strerror(errno));
        close(fd_wav);
        unlink(wav_path.c_str());
        return {};
    }
    fprintf(stderr, "[MP3] MP3 temp file: %s (fd=%d)\n", mp3_path.c_str(), fd_mp3);
    
    if (write(fd_wav, wav_data.data(), wav_data.size()) != (ssize_t)wav_data.size()) {
        close(fd_wav); close(fd_mp3);
        unlink(temp_wav); unlink(temp_mp3);
        fprintf(stderr, "[MP3] Failed to write temp WAV\n");
        return {};
    }
    close(fd_wav);
    close(fd_mp3);
    
    std::string cmd = "ffmpeg -y -loglevel error -i ";
    cmd += wav_path;
    cmd += " -ar 24000 -ac 1 -b:a 128k -vn ";
    cmd += mp3_path;
    
    int ret = system(cmd.c_str());
    if (ret != 0) {
        unlink(wav_path.c_str());
        unlink(mp3_path.c_str());
        fprintf(stderr, "[MP3] ffmpeg failed with code %d\n", ret);
        fprintf(stderr, "[MP3] Command: %s\n", cmd.c_str());
        fprintf(stderr, "[MP3] Input file size: %zu bytes\n", wav_data.size());
        return {};
    }
    
    FILE *fp = fopen(mp3_path.c_str(), "rb");
    if (!fp) {
        unlink(wav_path.c_str());
        unlink(mp3_path.c_str());
        fprintf(stderr, "[MP3] Failed to read MP3 output\n");
        return {};
    }
    
    fseek(fp, 0, SEEK_END);
    long mp3_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    std::string mp3_data;
    mp3_data.resize(mp3_size);
    size_t read_size = fread(&mp3_data[0], 1, mp3_size, fp);
    fclose(fp);
    
    unlink(wav_path.c_str());
    unlink(mp3_path.c_str());
    
    if (read_size != (size_t)mp3_size) {
        fprintf(stderr, "[MP3] Failed to read MP3 data\n");
        return {};
    }
    
    return mp3_data;
}

// Stored voice reference: audio samples and reference text.
struct tts_voice {
    std::string            id;        // voice identifier
    std::vector<float>     audio;     // mono f32 24 kHz reference audio
    std::string            text;      // reference text
    std::string            filepath;  // path to saved WAV file (optional)
};

// One synthesis request parsed from the OAI JSON body.
struct tts_request {
    std::string input;         // text to speak
    std::string voice;         // OAI voice, mapped to a speaker by the adapter or voice id
    std::string instructions;  // OAI instructions, mapped to the ABI instruct field
    std::string format;        // "pcm" (stream), "wav" (one-shot), or "mp3" (one-shot)
    float       speed;         // OAI speed, parsed then ignored (no time stretch in the ABI)
};

// Voice reference data passed to synthesis.
struct tts_voice_ref {
    const float * audio_24k;  // mono f32 24 kHz audio
    int           n_samples;  // length in samples
    const char *  text;       // reference text
};

// The adapter pushes mono f32 24 kHz audio here. Returns false to abort the
// synthesis (client gone or cancellation), which propagates into the ABI
// on_chunk and stops generation.
using tts_sink = std::function<bool(const float * samples, int n_samples)>;

// Voice storage and callbacks for voice cloning.
using tts_voice_map = std::unordered_map<std::string, tts_voice>;

// Adapter implemented by each project tool.
struct tts_backend {
    std::string              model_id;  // reported by GET /v1/models
    mutable std::vector<std::string> voices;    // reported by GET /v1/voices, may be empty
    mutable tts_voice_map    voice_data; // stored voice references by id
    std::string              voices_dir; // directory to save/load voice files

    // Load voices from local directory on startup.
    std::function<bool()> load_voices_from_disk;

    // Save a voice reference to disk.
    std::function<bool(const tts_voice & voice)> save_voice_to_disk;

    // Run synthesis. When the request streams, the adapter routes the ABI
    // on_chunk to sink ; otherwise it pushes the whole buffer once. Returns
    // the ABI status (0 on success), and fills err with the ABI message on
    // failure. The shared layer maps the status to an HTTP code.
    // The voice field can be a voice id; if found in voice_data, the
    // reference audio and text are used for cloning.
    std::function<int(const tts_request & req, const tts_sink & sink, std::string & err)> synthesize;
};

struct server_config {
    std::string host = "127.0.0.1";
    int         port = 8080;
};

// Single GPU context : synthesis is serialised FIFO across connections.
static std::mutex        g_synth_mutex;
static httplib::Server * g_svr = nullptr;

static void tts_on_signal(int) {
    if (g_svr) {
        g_svr->stop();
    }
}

// Clamp to [-1, 1] and scale to s16. lrintf rounds to nearest, ties to even.
static inline int16_t tts_f32_to_s16(float x) {
    float v = x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
    return (int16_t) lrintf(v * 32767.0f);
}

// Append a mono f32 block as s16le bytes onto out.
static void tts_append_s16le(std::string & out, const float * samples, int n_samples) {
    size_t base = out.size();
    out.resize(base + (size_t) n_samples * 2);
    char * p = &out[base];
    for (int i = 0; i < n_samples; i++) {
        int16_t s = tts_f32_to_s16(samples[i]);
        *p++      = (char) ((uint16_t) s & 0xff);
        *p++      = (char) (((uint16_t) s >> 8) & 0xff);
    }
}

// Write a JSON error body in the OAI error envelope and set the status.
static void tts_json_error(httplib::Response & res, int status, const char * type, const char * message) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_val * err = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_str(doc, err, "message", message);
    yyjson_mut_obj_add_str(doc, err, "type", type);
    yyjson_mut_obj_add_val(doc, root, "error", err);
    char * json = yyjson_mut_write(doc, 0, NULL);
    res.status  = status;
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
    yyjson_mut_doc_free(doc);
}

// Parse the OAI body into req. Returns false and fills err on bad input.
static bool tts_parse_request(const std::string & body, tts_request & req, std::string & err) {
    yyjson_doc * doc = yyjson_read(body.c_str(), body.size(), 0);
    if (!doc) {
        err = "request body is not valid JSON";
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) {
        err = "request body must be a JSON object";
        yyjson_doc_free(doc);
        return false;
    }

    yyjson_val * input = yyjson_obj_get(root, "input");
    if (!yyjson_is_str(input) || yyjson_get_len(input) == 0) {
        err = "'input' must be a non-empty string";
        yyjson_doc_free(doc);
        return false;
    }
    req.input = yyjson_get_str(input);

    yyjson_val * voice = yyjson_obj_get(root, "voice");
    req.voice          = yyjson_is_str(voice) ? yyjson_get_str(voice) : "";

    yyjson_val * instructions = yyjson_obj_get(root, "instructions");
    req.instructions          = yyjson_is_str(instructions) ? yyjson_get_str(instructions) : "";

    yyjson_val * fmt = yyjson_obj_get(root, "response_format");
    req.format       = yyjson_is_str(fmt) ? yyjson_get_str(fmt) : "mp3";

    yyjson_val * speed = yyjson_obj_get(root, "speed");
    req.speed          = yyjson_is_num(speed) ? (float) yyjson_get_num(speed) : 1.0f;

    yyjson_doc_free(doc);
    return true;
}

// Map an ABI status to an HTTP code. The two ABIs share numeric values:
// -1 invalid params, -2 mode/instruct invalid -> client error ; the rest
// are server side failures.
static int tts_status_to_http(int rc) {
    if (rc == 0) {
        return 200;
    }
    if (rc == -1 || rc == -2) {
        return 400;
    }
    return 502;
}

static void tts_handle_speech(const tts_backend & be, const httplib::Request & http_req, httplib::Response & res) {
    tts_request req;
    std::string err;
    if (!tts_parse_request(http_req.body, req, err)) {
        tts_json_error(res, 400, "invalid_request_error", err.c_str());
        return;
    }

    if (req.format == "wav" || req.format == "mp3") {
        // One-shot : collect the whole utterance, then emit a file.
        std::vector<float> buf;
        tts_sink           sink = [&buf](const float * s, int n) {
            buf.insert(buf.end(), s, s + n);
            return true;
        };
        std::string synth_err;
        int         rc;
        {
            std::lock_guard<std::mutex> lock(g_synth_mutex);
            rc = be.synthesize(req, sink, synth_err);
        }
        if (rc != 0) {
            tts_json_error(res, tts_status_to_http(rc), "server_error",
                           synth_err.empty() ? "synthesis failed" : synth_err.c_str());
            return;
        }
        
        if (req.format == "wav") {
            std::string wav = audio_encode_wav(buf.data(), (int) buf.size(), 24000, WAV_S16);
            res.set_content(std::move(wav), "audio/wav");
        } else {
            std::string mp3 = audio_encode_mp3(buf.data(), (int) buf.size(), 24000);
            if (mp3.empty()) {
                tts_json_error(res, 500, "server_error", "Failed to encode MP3");
                return;
            }
            res.set_content(std::move(mp3), "audio/mpeg");
        }
        return;
    }

    // Streaming : run synthesis inside the chunked provider on the connection
    // thread, pushing s16le frames as the codec produces them. A failed
    // sink.write means the client disconnected, which aborts generation and
    // frees the GPU instead of finishing a stream nobody reads.
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");
    res.set_chunked_content_provider("audio/pcm", [&be, req](size_t, httplib::DataSink & sink) mutable -> bool {
        tts_sink push = [&sink](const float * s, int n) {
            std::string bytes;
            tts_append_s16le(bytes, s, n);
            return sink.write(bytes.data(), bytes.size());
        };
        std::string synth_err;
        {
            std::lock_guard<std::mutex> lock(g_synth_mutex);
            be.synthesize(req, push, synth_err);
        }
        sink.done();
        return true;
    });
}

static void tts_handle_models(const tts_backend & be, const httplib::Request &, httplib::Response & res) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "object", "list");
    yyjson_mut_val * data = yyjson_mut_arr(doc);
    yyjson_mut_val * one  = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_str(doc, one, "id", be.model_id.c_str());
    yyjson_mut_obj_add_str(doc, one, "object", "model");
    yyjson_mut_obj_add_str(doc, one, "owned_by", "local");
    yyjson_mut_arr_add_val(data, one);
    yyjson_mut_obj_add_val(doc, root, "data", data);
    char * json = yyjson_mut_write(doc, 0, NULL);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
    yyjson_mut_doc_free(doc);
}

// Parse multipart form data.
static bool tts_parse_multipart_form(
    const std::string & body,
    const std::string & boundary,
    std::string & audio_wav,
    std::string & ref_text,
    std::string & voice_id) {

    audio_wav.clear();
    ref_text.clear();
    voice_id.clear();

    std::string delimiter = "--" + boundary;
    size_t pos = 0;

    while ((pos = body.find(delimiter, pos)) != std::string::npos) {
        pos += delimiter.length();

        // Skip CRLF or LF
        if (pos < body.length() && body[pos] == '\r') pos++;
        if (pos < body.length() && body[pos] == '\n') pos++;

        // Find headers
        size_t header_end = body.find("\r\n\r\n", pos);
        if (header_end == std::string::npos) break;

        std::string headers = body.substr(pos, header_end - pos);
        size_t body_start = header_end + 4;

        // Find part end
        size_t part_end = body.find(delimiter, body_start);
        if (part_end == std::string::npos) break;

        std::string part_body = body.substr(body_start, part_end - body_start);
        // Remove trailing CRLF if present
        if (part_body.length() >= 2 && part_body.substr(part_body.length() - 2) == "\r\n") {
            part_body = part_body.substr(0, part_body.length() - 2);
        }

        // Extract form field name (handles both "name=" and "name="; filename="...")
        size_t name_pos = headers.find("name=\"");
        if (name_pos != std::string::npos) {
            name_pos += 6;
            size_t name_end = headers.find("\"", name_pos);
            std::string name = headers.substr(name_pos, name_end - name_pos);

            if (name == "audio") {
                audio_wav = part_body;
            } else if (name == "text") {
                ref_text = part_body;
            } else if (name == "id") {
                voice_id = part_body;
            }
        }

        pos = part_end;
    }

    fprintf(stderr, "[Voices] Parsed: audio=%zu, text=%zu, id=%zu\n", audio_wav.size(), ref_text.size(), voice_id.size());
    return !audio_wav.empty() && !ref_text.empty() && !voice_id.empty();
}

static void tts_handle_voices_get(const tts_backend & be, const httplib::Request &, httplib::Response & res) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_val * arr = yyjson_mut_arr(doc);
    for (const std::string & v : be.voices) {
        yyjson_mut_val * one = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_str(doc, one, "id", v.c_str());
        yyjson_mut_obj_add_str(doc, one, "name", v.c_str());
        yyjson_mut_arr_add_val(arr, one);
    }
    yyjson_mut_obj_add_val(doc, root, "voices", arr);
    char * json = yyjson_mut_write(doc, 0, NULL);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
    yyjson_mut_doc_free(doc);
}

static void tts_handle_voices_post(
    const tts_backend & be,
    const httplib::Request & http_req,
    httplib::Response & res) {

    // httplib already parsed the multipart form data into http_req.form
    // (req.body is empty because httplib consumes it during its own parsing).
    if (!http_req.is_multipart_form_data()) {
        tts_json_error(res, 400, "invalid_request_error", "Content-Type must be multipart/form-data");
        return;
    }

    if (!http_req.form.has_field("id") || !http_req.form.has_field("text") || !http_req.form.has_file("audio")) {
        tts_json_error(res, 400, "invalid_request_error",
                       "Missing required fields: audio, text, or id");
        return;
    }

    std::string voice_id  = http_req.form.get_field("id");
    std::string ref_text  = http_req.form.get_field("text");
    httplib::FormData audio_file = http_req.form.get_file("audio");
    std::string audio_data = audio_file.content;
    std::string filename = audio_file.filename;
    std::string content_type = audio_file.content_type;

    // Determine audio format from filename extension or content type
    std::string audio_wav;
    int n_samples = 0;
    float * audio = nullptr;

    // Convert filename to lowercase for case-insensitive comparison
    std::string ext = filename;
    for (size_t i = 0; i < ext.size(); i++) {
        ext[i] = (char)std::tolower((unsigned char)ext[i]);
    }

    // Check if it's a WAV file (can use direct parsing)
    bool is_wav = (ext.size() >= 4 && ext.substr(ext.size() - 4) == ".wav") ||
                  content_type.find("audio/wav") != std::string::npos ||
                  content_type.find("audio/x-wav") != std::string::npos;

    if (is_wav) {
        // Use direct WAV parsing for WAV files
        audio = audio_read_mono_from_buf(audio_data.data(), audio_data.size(), 24000, &n_samples);
        if (!audio || n_samples <= 0) {
            tts_json_error(res, 400, "invalid_request_error", "Failed to parse audio WAV");
            return;
        }
    } else {
        // Use ffmpeg to decode other formats (mp3, ogg, m4a, flac, etc.)
        audio_wav = audio_decode_to_wav(audio_data.data(), audio_data.size(), 24000);
        if (audio_wav.empty()) {
            tts_json_error(res, 400, "invalid_request_error", 
                           "Failed to decode audio format. Supported formats: wav, mp3, ogg, m4a, flac");
            return;
        }
        audio = audio_read_mono_from_buf(audio_wav.data(), audio_wav.size(), 24000, &n_samples);
        if (!audio || n_samples <= 0) {
            tts_json_error(res, 400, "invalid_request_error", "Failed to parse decoded audio");
            return;
        }
    }

    // Create voice entry
    tts_voice voice;
    voice.id = voice_id;
    voice.audio.assign(audio, audio + n_samples);
    voice.text = ref_text;
    free(audio);

    // Save to disk if callback provided
    if (be.save_voice_to_disk) {
        if (!be.save_voice_to_disk(voice)) {
            tts_json_error(res, 500, "server_error", "Failed to save voice to disk");
            return;
        }
    }

    // Add to voice map and list
    be.voice_data[voice_id] = std::move(voice);
    be.voices.push_back(voice_id);

    // Return success
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "id", voice_id.c_str());
    yyjson_mut_obj_add_str(doc, root, "object", "voice");
    char * json = yyjson_mut_write(doc, 0, NULL);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
    yyjson_mut_doc_free(doc);
}

static void tts_handle_health(const httplib::Request &, httplib::Response & res) {
    res.set_content("{\"status\":\"ok\"}", "application/json");
}

static int tts_server_run(const tts_backend & be, const server_config & cfg) {
    httplib::Server svr;
    g_svr = &svr;

    // per-operation socket idle timeouts. read is small (text in), write is
    // generous to cover a long streamed utterance without tripping on a slow
    // client.
    svr.set_read_timeout(60);
    svr.set_write_timeout(120);

    // reject oversized bodies. text plus an optional reference clip stays
    // well under this.
    svr.set_payload_max_length(32 * 1024 * 1024);

    // Nagle coalescing holds small packets back for tens of ms ; streamed
    // PCM chunks must leave the socket the moment they are written.
    svr.set_tcp_nodelay(true);

    // SO_REUSEADDR lets us rebind a port still in TIME_WAIT after a restart.
    // SO_REUSEPORT is deliberately not set : a second instance on the same
    // port then fails with EADDRINUSE instead of silently sharing the socket
    // and splitting traffic between two daemons.
    svr.set_socket_options([](socket_t sock) {
        int one = 1;
#ifdef _WIN32
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char *) &one, sizeof(one));
#else
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
    });

    // permissive CORS so a browser client can call the API directly.
    svr.set_default_headers({
        { "Access-Control-Allow-Origin", "*" }
    });
    svr.Options("/.*", [](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
    });

    svr.Post("/v1/audio/speech",
             [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_speech(be, req, res); });
    svr.Get("/v1/models",
            [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_models(be, req, res); });
    svr.Post("/v1/voices",
             [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_voices_post(be, req, res); });
    svr.Get("/v1/voices",
            [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_voices_get(be, req, res); });
    svr.Get("/health", tts_handle_health);
    svr.Get("/", [](const httplib::Request & req, httplib::Response & res) {
        // Serve index.html from the project root
        std::ifstream file("index.html", std::ios::binary);
        if (file) {
            std::string content((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
            res.set_content(content, "text/html; charset=utf-8");
        } else {
            res.set_content("<html><body><h1>Index page not found</h1></body></html>", "text/html");
            res.status = 404;
        }
    });

    signal(SIGINT, tts_on_signal);
    signal(SIGTERM, tts_on_signal);

    fprintf(stderr, "[Server] model %s\n", be.model_id.c_str());
    fprintf(stderr, "[Server] listening on %s:%d\n", cfg.host.c_str(), cfg.port);
    if (!svr.listen(cfg.host.c_str(), cfg.port)) {
        fprintf(stderr, "[Server] FATAL: cannot bind %s:%d\n", cfg.host.c_str(), cfg.port);
        return 1;
    }
    return 0;
}
