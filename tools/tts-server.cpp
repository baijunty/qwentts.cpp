// tts-server.cpp: OpenAI-compatible HTTP server backed by the qwentts
// ABI. Loads a talker + codec once, GPU resident, and serves synthesis over
// POST /v1/audio/speech. The shared core lives in src/tts-server.h ; this
// file only wires the qt_* ABI into the adapter.
//
// OmniVoice has no named speaker table, so GET /v1/voices is empty unless
// voice cloning is enabled. The OAI voice field can be a voice id to
// trigger voice cloning with pre-sampled reference audio.

#include "tts-server.h"

#include "qwen.h"
#include "version.h"

#include "utf8.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

static void print_usage(const char * prog) {
    fprintf(stderr, "qwentts.cpp %s\n\n", QWEN_VERSION);
    fprintf(stderr,
            "Usage: %s --model <gguf> --codec <gguf> [options]\n\n"
            "Required:\n"
            "  --model <gguf>          Talker LM GGUF (qwen-talker-*.gguf)\n"
            "  --codec <gguf>          Codec GGUF (qwen-tokenizer-*.gguf)\n\n"
            "Optional:\n"
            "  --host <ip>             Listen address (default: 127.0.0.1)\n"
            "  --port <n>              Listen port (default: 8080)\n"
            "  --lang <name>           Language label (default: auto)\n"
            "  --no-fa                 Disable flash attention\n"
            "  --clamp-fp16            Clamp hidden states to FP16 range\n"
            "  --voices-dir <path>     Directory to save/load voice references\n\n"
            "API Endpoints:\n"
            "  POST /v1/audio/speech   OAI TTS (response_format: pcm, wav, mp3; default: mp3)\n"
            "  POST /v1/voices         Upload voice reference (audio + text + id)\n"
            "  GET  /v1/voices         List saved voices\n"
            "  GET  /v1/models         List loaded model\n"
            "  GET  /health            Liveness probe\n",
            prog);
}

// Trim a path down to its file name for the reported model id.
static std::string basename_of(const char * path) {
    std::string s = path;
    size_t      p = s.find_last_of("/\\");
    return p == std::string::npos ? s : s.substr(p + 1);
}

// Save a voice reference to disk as WAV + text file.
static bool save_voice_to_file(const tts_voice & voice, const std::string & voices_dir) {
    try {
        fs::create_directories(voices_dir);
    } catch (const std::exception & e) {
        fprintf(stderr, "[Voices] Failed to create voices directory %s: %s\n", voices_dir.c_str(), e.what());
        return false;
    }

    std::string wav_path = voices_dir + "/" + voice.id + ".wav";
    std::string txt_path = voices_dir + "/" + voice.id + ".txt";

    // Write WAV file
    if (!audio_write_wav(wav_path.c_str(), voice.audio.data(), (int) voice.audio.size(), 24000, WAV_S16)) {
        fprintf(stderr, "[Voices] Failed to write WAV %s\n", wav_path.c_str());
        return false;
    }

    // Write reference text
    std::ofstream txt_file(txt_path);
    if (!txt_file) {
        fprintf(stderr, "[Voices] Failed to open text file %s for writing\n", txt_path.c_str());
        return false;
    }
    txt_file << voice.text;
    txt_file.close();

    fprintf(stderr, "[Voices] Saved voice %s to %s\n", voice.id.c_str(), voices_dir.c_str());
    return true;
}

// Load a single voice from disk.
static bool load_voice_from_file(const std::string & id, const std::string & voices_dir, tts_voice & voice) {
    std::string wav_path = voices_dir + "/" + id + ".wav";
    std::string txt_path = voices_dir + "/" + id + ".txt";

    // Check if files exist
    if (!fs::exists(wav_path) || !fs::exists(txt_path)) {
        return false;
    }

    // Read WAV file
    int n_samples = 0;
    float * audio = audio_read_mono(wav_path.c_str(), 24000, &n_samples);
    if (!audio || n_samples <= 0) {
        fprintf(stderr, "[Voices] Failed to read WAV %s\n", wav_path.c_str());
        return false;
    }

    // Read reference text
    std::ifstream txt_file(txt_path);
    if (!txt_file) {
        free(audio);
        fprintf(stderr, "[Voices] Failed to open text file %s\n", txt_path.c_str());
        return false;
    }
    std::string text((std::istreambuf_iterator<char>(txt_file)),
                     std::istreambuf_iterator<char>());
    txt_file.close();

    voice.id = id;
    voice.audio.assign(audio, audio + n_samples);
    voice.text = text;
    voice.filepath = wav_path;

    free(audio);
    return true;
}

int main(int argc, char ** argv) {
    const char *  talker_path = NULL;
    const char *  codec_path  = NULL;
    std::string   lang        = "auto";
    std::string   voices_dir  = "voices";
    server_config cfg;
    bool          use_fa     = true;
    bool          clamp_fp16 = false;

    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (!std::strcmp(arg, "--model") && i + 1 < argc) {
            talker_path = argv[++i];
        } else if (!std::strcmp(arg, "--codec") && i + 1 < argc) {
            codec_path = argv[++i];
        } else if (!std::strcmp(arg, "--host") && i + 1 < argc) {
            cfg.host = argv[++i];
        } else if (!std::strcmp(arg, "--port") && i + 1 < argc) {
            cfg.port = std::atoi(argv[++i]);
        } else if (!std::strcmp(arg, "--lang") && i + 1 < argc) {
            lang = argv[++i];
        } else if (!std::strcmp(arg, "--voices-dir") && i + 1 < argc) {
            voices_dir = argv[++i];
        } else if (!std::strcmp(arg, "--no-fa")) {
            use_fa = false;
        } else if (!std::strcmp(arg, "--clamp-fp16")) {
            clamp_fp16 = true;
        } else if (!std::strcmp(arg, "--help") || !std::strcmp(arg, "-h")) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "[CLI] ERROR: unknown arg: %s\n", arg);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!talker_path || !codec_path) {
        print_usage(argv[0]);
        return 0;
    }

    struct qt_init_params iparams;
    qt_init_default_params(&iparams);
    iparams.talker_path = talker_path;
    iparams.codec_path  = codec_path;
    iparams.use_fa      = use_fa;
    iparams.clamp_fp16  = clamp_fp16;

    struct qt_context * q = qt_init(&iparams);
    if (!q) {
        fprintf(stderr, "[Server] FATAL: %s\n", qt_last_error());
        return 1;
    }

    tts_backend be;
    be.model_id = basename_of(talker_path);
    int n = qt_n_speakers(q);
    for (int i = 0; i < n; i++) {
        be.voices.push_back(qt_speaker_name(q, i));
    }
    be.voices_dir = voices_dir;

    // Set up voice loading callback
    if (!voices_dir.empty()) {
        be.load_voices_from_disk = [&be]() -> bool {
            if (be.voices_dir.empty()) return true;

            try {
                if (!fs::exists(be.voices_dir)) {
                    fs::create_directories(be.voices_dir);
                    return true;
                }

                int loaded = 0;
                for (const auto & entry : fs::directory_iterator(be.voices_dir)) {
                    if (entry.is_regular_file() && entry.path().extension() == ".wav") {
                        std::string id = entry.path().stem().string();
                        tts_voice voice;
                        if (load_voice_from_file(id, be.voices_dir, voice)) {
                            be.voice_data[id] = std::move(voice);
                            be.voices.push_back(id);
                            loaded++;
                        }
                    }
                }
                fprintf(stderr, "[Voices] Loaded %d voices from %s\n", loaded, be.voices_dir.c_str());
                return true;
            } catch (const std::exception & e) {
                fprintf(stderr, "[Voices] Error loading voices: %s\n", e.what());
                return false;
            }
        };

        be.save_voice_to_disk = [&be](const tts_voice & voice) -> bool {
            return save_voice_to_file(voice, be.voices_dir);
        };

        // Load voices from disk on startup
        if (!be.load_voices_from_disk()) {
            fprintf(stderr, "[Voices] Warning: failed to load voices from disk\n");
        }
    }

    // The adapter always drives the streaming pipeline : on_chunk routes to
    // the shared sink, which either streams to the socket (pcm) or fills a
    // one-shot buffer (wav). Either way the audio path is identical.
    //
    // Voice cloning: if req.voice matches a stored voice id, use its
    // reference audio and text for synthesis.
    be.synthesize = [q, &lang, &be](const tts_request & req, const tts_sink & sink, std::string & err) -> int {
        struct qt_tts_params p;
        qt_tts_default_params(&p);
        p.text = req.input.c_str();
        p.lang = lang.c_str();
        if (!req.voice.empty() && qt_n_speakers(q) > 0) {
            p.speaker = req.voice.c_str();
        }
        if (!req.instructions.empty()) {
            p.instruct = req.instructions.c_str();
        }

        // Check if voice field is a stored voice id
        tts_voice_ref voice_ref = {};
        if (!req.voice.empty()) {
            auto it = be.voice_data.find(req.voice);
            if (it != be.voice_data.end()) {
                // Use stored voice for cloning
                voice_ref.audio_24k = it->second.audio.data();
                voice_ref.n_samples = (int) it->second.audio.size();
                voice_ref.text = it->second.text.c_str();

                p.ref_audio_24k = voice_ref.audio_24k;
                p.ref_n_samples = voice_ref.n_samples;
                p.ref_text = voice_ref.text;

                fprintf(stderr, "[TTS] Voice cloning with voice id: %s (%d samples)\n",
                        req.voice.c_str(), voice_ref.n_samples);
            }
        }

        // Trampoline : the C ABI on_chunk forwards to the C++ sink.
        const tts_sink * sink_ptr = &sink;
        p.on_chunk                = [](const float * s, int ns, void * u) -> bool {
            return (*static_cast<const tts_sink *>(u))(s, ns);
        };
        p.on_chunk_user_data = (void *) sink_ptr;

        struct qt_audio out = {};
        enum qt_status  rc  = qt_synthesize(q, &p, &out);
        qt_audio_free(&out);
        if (rc != QT_STATUS_OK) {
            err = qt_last_error();
            return (int) rc;
        }
        return 0;
    };

    int rc = tts_server_run(be, cfg);
    qt_free(q);
    return rc;
}
