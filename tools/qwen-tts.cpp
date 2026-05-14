// qwen-tts.cpp: thin CLI wrapper around the qwentts.cpp public ABI.
// Parses arguments, reads the optional reference WAV plus transcript,
// hands off to qt_synthesize and writes the resulting waveform as a
// WAV file. All synthesis logic, mode validation and seed resolution
// live behind the qwen_* facade declared in qwen.h.
//
// Talker variants: 0.6B-Base / 0.6B-CustomVoice / 1.7B-Base /
// 1.7B-CustomVoice / 1.7B-VoiceDesign. The decoder path is selected
// from GGUF metadata at qt_init time. The CLI surface mirrors the
// omnivoice.cpp tooling: kebab-case flags, --format wav16/wav24/wav32,
// -o '-' streams to stdout, --seed -1 means non deterministic
// (resolved inside qt_synthesize), the utterance text comes from
// --text or stdin if --text is absent.

#include "audio-io.h"
#include "pipeline-codec.h"
#include "qwen.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

static void print_usage(const char * prog) {
    fprintf(stderr, "qwentts.cpp %s\n\n", qt_version());
    fprintf(stderr,
            "Usage: %s --model <gguf> --codec <gguf> [options] -o <out.wav>\n\n"
            "Required:\n"
            "  --model <gguf>          Talker LM GGUF (qwen-talker-*.gguf)\n"
            "  --codec <gguf>          Tokenizer GGUF (qwen-tokenizer-*.gguf)\n"
            "  -o <path>               Output WAV. '-' streams to stdout (pipe friendly).\n\n"
            "Input:\n"
            "  --text <s>              Utterance text. If absent, stdin is read fully.\n\n"
            "Synthesis options:\n"
            "  --lang <name>           Language (auto, english, chinese, ...) (default: english)\n"
            "  --instruct <s>          Style instruction. Required for VoiceDesign, optional\n"
            "                          for CustomVoice. Rejected for Base.\n"
            "  --speaker <name>        Speaker name. Only valid for CustomVoice.\n"
            "  --ref-wav <wav>       Reference WAV path for voice clone (Base only). Mutually\n"
            "                          exclusive with --speaker. Mode A (x_vector_only) extracts\n"
            "                          a speaker embedding via the ECAPA-TDNN encoder.\n"
            "  --ref-text <path>       Path to a UTF-8 text file containing the reference\n"
            "                          transcript for voice clone ICL mode (Base only, requires\n"
            "                          --ref-wav). Switches the prompt to ICL mode B where the\n"
            "                          talker conditions on the reference codec codes.\n"
            "  --max-new <n>           Max new audio frames (default: 2048)\n"
            "  --format <fmt>          WAV output format: wav16, wav24, wav32 (default: wav16)\n\n"
            "Sampling options:\n"
            "  --seed <n>              Sampling seed, -1 for random (default: -1)\n"
            "  --greedy                Disable stochastic sampling on both stacks\n"
            "  --temp <f>              Talker temperature (default: 0.9)\n"
            "  --top-k <n>             Talker top-k (default: 50, 0 = disabled)\n"
            "  --top-p <f>             Talker top-p (default: 1.0)\n"
            "  --rep-pen <f>           Talker repetition penalty (default: 1.05)\n"
            "  --sub-temp <f>          Sub-talker temperature (default: 0.9)\n"
            "  --sub-top-k <n>         Sub-talker top-k (default: 50)\n"
            "  --sub-top-p <f>         Sub-talker top-p (default: 1.0)\n\n"
            "Debug:\n"
            "  --dump <dir>            Dump intermediate tensors for cossim debug\n",
            prog);
}

struct Args {
    const char * model;
    const char * codec;
    const char * text;
    const char * lang;
    const char * instruct;
    const char * speaker;
    const char * ref_wav;
    const char * ref_text_path;
    const char * dump_dir;
    const char * out_wav;
    const char * format;
    int          max_new_tokens;
    int64_t      seed;
    bool         do_sample;
    float        temperature;
    int          top_k;
    float        top_p;
    float        repetition_penalty;
    int          subtalker_top_k;
    float        subtalker_top_p;
    float        subtalker_temperature;
    bool         subtalker_do_sample;
};

// Read all of stdin into a string. Trims trailing newlines so a piped
// text file behaves like a clean --text argument.
static std::string read_stdin_text() {
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    std::string s = ss.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
        s.pop_back();
    }
    return s;
}

// Read a small text file into a string. Trims trailing newlines.
static bool read_text_file(const char * path, std::string & out) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[CLI] FATAL: cannot open '%s'\n", path);
        return false;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) {
        fclose(f);
        fprintf(stderr, "[CLI] FATAL: ftell failed on '%s'\n", path);
        return false;
    }
    out.resize((size_t) sz);
    if (sz > 0 && fread(&out[0], 1, (size_t) sz, f) != (size_t) sz) {
        fclose(f);
        fprintf(stderr, "[CLI] FATAL: fread failed on '%s'\n", path);
        return false;
    }
    fclose(f);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
        out.pop_back();
    }
    return true;
}

static bool parse_args(int argc, char ** argv, Args & a) {
    a                       = {};
    a.lang                  = "english";
    a.format                = "wav16";
    a.max_new_tokens        = 2048;
    a.seed                  = -1;
    a.do_sample             = true;
    a.temperature           = 0.9f;
    a.top_k                 = 50;
    a.top_p                 = 1.0f;
    a.repetition_penalty    = 1.05f;
    a.subtalker_do_sample   = true;
    a.subtalker_top_k       = 50;
    a.subtalker_top_p       = 1.0f;
    a.subtalker_temperature = 0.9f;
    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
            return false;
        }
        if (std::strcmp(arg, "--model") == 0 && i + 1 < argc) {
            a.model = argv[++i];
        } else if (std::strcmp(arg, "--codec") == 0 && i + 1 < argc) {
            a.codec = argv[++i];
        } else if (std::strcmp(arg, "--text") == 0 && i + 1 < argc) {
            a.text = argv[++i];
        } else if (std::strcmp(arg, "--lang") == 0 && i + 1 < argc) {
            a.lang = argv[++i];
        } else if (std::strcmp(arg, "--instruct") == 0 && i + 1 < argc) {
            a.instruct = argv[++i];
        } else if (std::strcmp(arg, "--speaker") == 0 && i + 1 < argc) {
            a.speaker = argv[++i];
        } else if (std::strcmp(arg, "--ref-wav") == 0 && i + 1 < argc) {
            a.ref_wav = argv[++i];
        } else if (std::strcmp(arg, "--ref-text") == 0 && i + 1 < argc) {
            a.ref_text_path = argv[++i];
        } else if (std::strcmp(arg, "--format") == 0 && i + 1 < argc) {
            a.format = argv[++i];
        } else if (std::strcmp(arg, "--dump") == 0 && i + 1 < argc) {
            a.dump_dir = argv[++i];
        } else if (std::strcmp(arg, "--max-new") == 0 && i + 1 < argc) {
            a.max_new_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--seed") == 0 && i + 1 < argc) {
            a.seed = (int64_t) std::atoll(argv[++i]);
        } else if (std::strcmp(arg, "--greedy") == 0) {
            // Greedy mode: argmax sampling on both stacks. The sampling
            // fast path in sampling.h uses temperature <= 0 to short
            // circuit to argmax, bypassing rep penalty and top-k/p
            // truncation, which exactly mirrors the Python reference
            // greedy behaviour used by tests/debug-tts-cossim.py.
            a.do_sample           = false;
            a.subtalker_do_sample = false;
        } else if (std::strcmp(arg, "--temp") == 0 && i + 1 < argc) {
            a.temperature = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--top-k") == 0 && i + 1 < argc) {
            a.top_k = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--top-p") == 0 && i + 1 < argc) {
            a.top_p = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--rep-pen") == 0 && i + 1 < argc) {
            a.repetition_penalty = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--sub-temp") == 0 && i + 1 < argc) {
            a.subtalker_temperature = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--sub-top-k") == 0 && i + 1 < argc) {
            a.subtalker_top_k = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--sub-top-p") == 0 && i + 1 < argc) {
            a.subtalker_top_p = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "-o") == 0 && i + 1 < argc) {
            a.out_wav = argv[++i];
        } else {
            fprintf(stderr, "[CLI] ERROR: unknown or incomplete argument: %s\n", arg);
            return false;
        }
    }
    return a.model && a.codec;
}

static int run(const Args & a) {
    // Init the facade. The seven mode validations
    // (base / custom_voice / voice_design rules) and the BPE tokenizer
    // load live inside qt_init / qt_synthesize; the CLI just hands
    // off the two GGUF paths and reports qt_last_error on failure.
    qt_init_params iparams;
    qt_init_default_params(&iparams);
    iparams.talker_path = a.model;
    iparams.codec_path  = a.codec;

    qt_context * q = qt_init(&iparams);
    if (!q) {
        fprintf(stderr, "[CLI] ERROR: %s\n", qt_last_error());
        return 1;
    }

    // Load the reference transcript from the file passed via --ref-text.
    // Empty content is rejected since the ICL prompt needs it non empty.
    std::string  ref_text_buf;
    const char * ref_text = NULL;
    if (a.ref_text_path) {
        if (!read_text_file(a.ref_text_path, ref_text_buf)) {
            qt_free(q);
            return 1;
        }
        if (ref_text_buf.empty()) {
            fprintf(stderr, "[CLI] ERROR: --ref-text file '%s' is empty\n", a.ref_text_path);
            qt_free(q);
            return 1;
        }
        ref_text = ref_text_buf.c_str();
    }

    // Decode the reference WAV once, mono at the codec sample rate. The
    // facade consumes the buffer directly so the WAV is read exactly
    // once regardless of how many encoders need the audio (speaker
    // encoder embedding + codec encoder RVQ codes for ICL mode B).
    std::unique_ptr<float, void (*)(void *)> raw_holder(NULL, std::free);
    const float *                            ref_audio_24k = NULL;
    int                                      ref_n_samples = 0;
    if (a.ref_wav) {
        int     T_in = 0;
        float * raw  = audio_read_mono(a.ref_wav, TOKENIZER_SAMPLE_RATE, &T_in);
        if (!raw || T_in <= 0) {
            fprintf(stderr, "[CLI] ERROR: cannot read --ref-wav '%s'\n", a.ref_wav);
            if (raw) {
                std::free(raw);
            }
            qt_free(q);
            return 1;
        }
        raw_holder.reset(raw);
        ref_audio_24k = raw;
        ref_n_samples = T_in;
    }

    // Resolve output WAV format string: wav16 / wav24 / wav32. Default
    // wav16 mirrors the omnivoice.cpp default.
    WavFormat wav_fmt;
    if (!audio_parse_format(a.format, wav_fmt)) {
        fprintf(stderr, "[CLI] ERROR: invalid --format '%s' (expected wav16, wav24, wav32)\n", a.format);
        qt_free(q);
        return 1;
    }

    // Resolve utterance text: explicit --text wins, otherwise read stdin
    // fully. Empty stdin combined with no --text triggers a clean error.
    std::string  text_buf;
    const char * text = a.text;
    if (!text) {
        text_buf = read_stdin_text();
        if (text_buf.empty()) {
            fprintf(stderr, "[CLI] ERROR: no --text and stdin is empty\n");
            qt_free(q);
            return 1;
        }
        text = text_buf.c_str();
    }

    // Translate CLI args into the facade params. Seed -1 is forwarded
    // verbatim and resolved by qt_synthesize via std::random_device.
    qt_tts_params params;
    qt_tts_default_params(&params);
    params.text                  = text;
    params.lang                  = a.lang;
    params.instruct              = a.instruct;
    params.speaker               = a.speaker;
    params.ref_audio_24k         = ref_audio_24k;
    params.ref_n_samples         = ref_n_samples;
    params.ref_text              = ref_text;
    params.seed                  = a.seed;
    params.max_new_tokens        = a.max_new_tokens;
    params.do_sample             = a.do_sample;
    params.temperature           = a.temperature;
    params.top_k                 = a.top_k;
    params.top_p                 = a.top_p;
    params.repetition_penalty    = a.repetition_penalty;
    params.subtalker_do_sample   = a.subtalker_do_sample;
    params.subtalker_temperature = a.subtalker_temperature;
    params.subtalker_top_k       = a.subtalker_top_k;
    params.subtalker_top_p       = a.subtalker_top_p;
    params.dump_dir              = a.dump_dir;

    qt_audio  audio  = {};
    qt_status status = qt_synthesize(q, &params, &audio);
    if (status != QT_STATUS_OK) {
        fprintf(stderr, "[CLI] ERROR: %s\n", qt_last_error());
        qt_audio_free(&audio);
        qt_free(q);
        return 1;
    }

    if (audio.n_samples > 0) {
        const char * out_path = a.out_wav ? a.out_wav : "out.wav";
        if (!audio_write_wav(out_path, audio.samples, audio.n_samples, audio.sample_rate, wav_fmt)) {
            fprintf(stderr, "[Pipeline] FATAL: WAV write failed for %s\n", out_path);
            qt_audio_free(&audio);
            qt_free(q);
            return 1;
        }
        fprintf(stderr, "[Pipeline] Wrote %d samples (%.2f s) -> %s\n", audio.n_samples,
                (double) audio.n_samples / (double) audio.sample_rate, out_path);
    }

    qt_audio_free(&audio);
    qt_free(q);
    return 0;
}

int main(int argc, char ** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) {
        print_usage(argv[0]);
        return 1;
    }
    // The facade absorbs every std::exception thrown deep in the load
    // and synthesis chains, converting them into qt_status + a
    // qt_last_error message. No top-level try / catch needed here :
    // the CLI just reads the status returned by qt_init /
    // qt_synthesize and renders qt_last_error to stderr.
    return run(a);
}
