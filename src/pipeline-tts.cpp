// pipeline-tts.cpp: load and verify both GGUF files (talker + codec)
// onto the same shared backend, parse all metadata into typed structs,
// and provide a structured load-time summary for --load-only mode.

#include "pipeline-tts.h"

#include "audio-io.h"
#include "bpe.h"
#include "code-predictor-forward.h"
#include "debug.h"
#include "ggml.h"
#include "pipeline-codec.h"
#include "prompt-builder.h"
#include "qt-error.h"
#include "sampling.h"
#include "speaker-encoder-extract.h"
#include "talker-forward.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void parse_codec_specials(const GGUFModel & gf, CodecSpecials & cs) {
    cs.pad_id       = (int) gf_get_u32(gf, "qwen3-tts.codec.pad_id");
    cs.bos_id       = (int) gf_get_u32(gf, "qwen3-tts.codec.bos_id");
    cs.eos_id       = (int) gf_get_u32(gf, "qwen3-tts.codec.eos_id");
    cs.think_id     = (int) gf_get_u32(gf, "qwen3-tts.codec.think_id");
    cs.nothink_id   = (int) gf_get_u32(gf, "qwen3-tts.codec.nothink_id");
    cs.think_bos_id = (int) gf_get_u32(gf, "qwen3-tts.codec.think_bos_id");
    cs.think_eos_id = (int) gf_get_u32(gf, "qwen3-tts.codec.think_eos_id");
}

static void parse_text_specials(const GGUFModel & gf, TextSpecials & ts) {
    ts.im_start_id = (int) gf_get_u32(gf, "qwen3-tts.text.im_start_id");
    ts.im_end_id   = (int) gf_get_u32(gf, "qwen3-tts.text.im_end_id");
    ts.tts_pad_id  = (int) gf_get_u32(gf, "qwen3-tts.text.tts_pad_id");
    ts.tts_bos_id  = (int) gf_get_u32(gf, "qwen3-tts.text.tts_bos_id");
    ts.tts_eos_id  = (int) gf_get_u32(gf, "qwen3-tts.text.tts_eos_id");
}

static void parse_languages(const GGUFModel & gf, std::vector<LanguageEntry> & out) {
    int64_t name_idx = gguf_find_key(gf.gguf, "qwen3-tts.codec.language_names");
    int64_t id_idx   = gguf_find_key(gf.gguf, "qwen3-tts.codec.language_ids");
    if (name_idx < 0 || id_idx < 0) {
        return;
    }
    size_t n_names = gguf_get_arr_n(gf.gguf, name_idx);
    size_t n_ids   = gguf_get_arr_n(gf.gguf, id_idx);
    if (n_names != n_ids) {
        qt_log(QT_LOG_WARN, "[Pipeline] language arrays size mismatch (names=%zu, ids=%zu)", n_names, n_ids);
        return;
    }
    const uint32_t * ids = (const uint32_t *) gguf_get_arr_data(gf.gguf, id_idx);
    out.reserve(n_names);
    for (size_t i = 0; i < n_names; i++) {
        LanguageEntry e;
        e.name = gguf_get_arr_str(gf.gguf, name_idx, i);
        e.id   = (int) ids[i];
        out.push_back(e);
    }
}

// Parse the speaker table for CustomVoice variants. Three parallel arrays
// produced by convert.py: speaker_names, speaker_ids, speaker_dialects.
// Empty dialect string means the speaker keeps the user supplied language.
// Skipped silently when the GGUF carries no speaker table (Base / VoiceDesign).
static void parse_speakers(const GGUFModel & gf, std::vector<SpeakerEntry> & out) {
    int64_t name_idx    = gguf_find_key(gf.gguf, "qwen3-tts.codec.speaker_names");
    int64_t id_idx      = gguf_find_key(gf.gguf, "qwen3-tts.codec.speaker_ids");
    int64_t dialect_idx = gguf_find_key(gf.gguf, "qwen3-tts.codec.speaker_dialects");
    if (name_idx < 0 || id_idx < 0 || dialect_idx < 0) {
        return;
    }
    size_t n_names    = gguf_get_arr_n(gf.gguf, name_idx);
    size_t n_ids      = gguf_get_arr_n(gf.gguf, id_idx);
    size_t n_dialects = gguf_get_arr_n(gf.gguf, dialect_idx);
    if (n_names != n_ids || n_names != n_dialects) {
        qt_log(QT_LOG_WARN, "[Pipeline] speaker arrays size mismatch (names=%zu, ids=%zu, dialects=%zu)", n_names,
               n_ids, n_dialects);
        return;
    }
    const uint32_t * ids = (const uint32_t *) gguf_get_arr_data(gf.gguf, id_idx);
    out.reserve(n_names);
    for (size_t i = 0; i < n_names; i++) {
        SpeakerEntry e;
        e.name    = gguf_get_arr_str(gf.gguf, name_idx, i);
        e.id      = (int) ids[i];
        e.dialect = gguf_get_arr_str(gf.gguf, dialect_idx, i);
        out.push_back(e);
    }
}

static void parse_generation_defaults(const GGUFModel & gf, GenerationDefaults & g) {
    g.do_sample             = gf_get_bool(gf, "generation.do_sample");
    g.top_k                 = (int) gf_get_u32(gf, "generation.top_k");
    g.top_p                 = gf_get_f32(gf, "generation.top_p");
    g.temperature           = gf_get_f32(gf, "generation.temperature");
    g.repetition_penalty    = gf_get_f32(gf, "generation.repetition_penalty");
    g.subtalker_do_sample   = gf_get_bool(gf, "generation.subtalker_do_sample");
    g.subtalker_top_k       = (int) gf_get_u32(gf, "generation.subtalker_top_k");
    g.subtalker_top_p       = gf_get_f32(gf, "generation.subtalker_top_p");
    g.subtalker_temperature = gf_get_f32(gf, "generation.subtalker_temperature");
    g.max_new_tokens        = (int) gf_get_u32(gf, "generation.max_new_tokens");
}

bool pipeline_tts_load(PipelineTTS * pt, const char * talker_gguf_path, const char * codec_gguf_path, BackendPair bp) {
    pt->bp                  = bp;
    pt->backend             = bp.backend;
    pt->sched               = NULL;
    pt->has_speaker_encoder = false;

    if (!gf_load(&pt->gguf_talker, talker_gguf_path)) {
        qt_log(QT_LOG_ERROR, "[Pipeline] failed to load talker GGUF: %s", talker_gguf_path);
        return false;
    }

    const char * arch = gf_get_str(pt->gguf_talker, "general.architecture");
    if (!arch || std::strcmp(arch, "qwen3-tts") != 0) {
        qt_log(QT_LOG_ERROR, "[Pipeline] talker GGUF has wrong architecture '%s', expected 'qwen3-tts'",
               arch ? arch : "");
        gf_close(&pt->gguf_talker);
        return false;
    }

    pt->tokenizer_type  = gf_get_str(pt->gguf_talker, "qwen3-tts.tokenizer_type");
    pt->model_size      = gf_get_str(pt->gguf_talker, "qwen3-tts.model_size");
    pt->model_type      = gf_get_str(pt->gguf_talker, "qwen3-tts.model_type");
    pt->num_code_groups = (int) gf_get_u32(pt->gguf_talker, "qwen3-tts.num_code_groups");

    parse_codec_specials(pt->gguf_talker, pt->codec_specials);
    parse_text_specials(pt->gguf_talker, pt->text_specials);
    parse_languages(pt->gguf_talker, pt->languages);
    parse_speakers(pt->gguf_talker, pt->speakers);
    parse_generation_defaults(pt->gguf_talker, pt->gen_defaults);

    if (!talker_weights_load(&pt->talker, pt->gguf_talker, pt->backend)) {
        gf_close(&pt->gguf_talker);
        return false;
    }

    if (!code_predictor_weights_load(&pt->code_predictor, pt->gguf_talker, pt->backend)) {
        talker_weights_free(&pt->talker);
        gf_close(&pt->gguf_talker);
        return false;
    }

    // Speaker encoder is only present in Base checkpoints. Treat absence
    // as a soft condition: voice clone path stays disabled, base-direct
    // synthesis still works.
    if (pt->model_type == "base") {
        if (!speaker_encoder_weights_load(&pt->speaker_encoder, pt->gguf_talker, pt->backend)) {
            code_predictor_weights_free(&pt->code_predictor);
            talker_weights_free(&pt->talker);
            gf_close(&pt->gguf_talker);
            return false;
        }
        pt->has_speaker_encoder = (pt->speaker_encoder.weight_buf != NULL);
    }

    if (!pipeline_codec_load(&pt->codec, codec_gguf_path, bp)) {
        if (pt->has_speaker_encoder) {
            speaker_encoder_weights_free(&pt->speaker_encoder);
        }
        code_predictor_weights_free(&pt->code_predictor);
        talker_weights_free(&pt->talker);
        gf_close(&pt->gguf_talker);
        return false;
    }

    // Scheduler shared by talker_forward_* and code_predictor_step.
    // Routes ops the GPU backend cannot run (typical case: K-quant
    // get_rows on CUDA) to the CPU backend. 4096 nodes covers the 28L
    // Qwen3 talker graph (~48 ops per layer with KV cache writes) with
    // headroom; the 5L code predictor uses a fraction of that.
    pt->sched = backend_sched_new(bp, 4096);
    if (!pt->sched) {
        pipeline_codec_free(&pt->codec);
        if (pt->has_speaker_encoder) {
            speaker_encoder_weights_free(&pt->speaker_encoder);
        }
        code_predictor_weights_free(&pt->code_predictor);
        talker_weights_free(&pt->talker);
        gf_close(&pt->gguf_talker);
        return false;
    }

    // KV caches: talker holds the LM context up to 4096 positions (the
    // longest ICL prompt observed is ~250 + max_new_tokens ~ 1500, so
    // 4096 has 60% headroom). Predictor holds one frame of 16 sub-steps.
    if (!kv_cache_init(&pt->talker_kv, pt->talker.num_hidden_layers, pt->talker.num_key_value_heads,
                       pt->talker.head_dim, 4096, pt->backend)) {
        ggml_backend_sched_free(pt->sched);
        pt->sched = NULL;
        pipeline_codec_free(&pt->codec);
        if (pt->has_speaker_encoder) {
            speaker_encoder_weights_free(&pt->speaker_encoder);
        }
        code_predictor_weights_free(&pt->code_predictor);
        talker_weights_free(&pt->talker);
        gf_close(&pt->gguf_talker);
        return false;
    }
    if (!kv_cache_init(&pt->code_predictor_kv, pt->code_predictor.num_hidden_layers,
                       pt->code_predictor.num_key_value_heads, pt->code_predictor.head_dim, pt->num_code_groups,
                       pt->backend)) {
        kv_cache_free(&pt->talker_kv);
        ggml_backend_sched_free(pt->sched);
        pt->sched = NULL;
        pipeline_codec_free(&pt->codec);
        if (pt->has_speaker_encoder) {
            speaker_encoder_weights_free(&pt->speaker_encoder);
        }
        code_predictor_weights_free(&pt->code_predictor);
        talker_weights_free(&pt->talker);
        gf_close(&pt->gguf_talker);
        return false;
    }

    qt_log(QT_LOG_INFO,
           "[Pipeline] Loaded: arch=%s variant=%s tokenizer=%s codebooks=%d speaker_encoder=%s speakers=%zu",
           pt->model_size.c_str(), pt->model_type.c_str(), pt->tokenizer_type.c_str(), pt->num_code_groups,
           pt->has_speaker_encoder ? "loaded" : "absent", pt->speakers.size());
    return true;
}

void pipeline_tts_free(PipelineTTS * pt) {
    kv_cache_free(&pt->code_predictor_kv);
    kv_cache_free(&pt->talker_kv);
    if (pt->sched) {
        ggml_backend_sched_free(pt->sched);
        pt->sched = NULL;
    }
    pipeline_codec_free(&pt->codec);
    if (pt->has_speaker_encoder) {
        speaker_encoder_weights_free(&pt->speaker_encoder);
    }
    code_predictor_weights_free(&pt->code_predictor);
    talker_weights_free(&pt->talker);
    gf_close(&pt->gguf_talker);
    pt->backend             = NULL;
    pt->bp                  = {};
    pt->has_speaker_encoder = false;
}

// Pull one row of an embedding table directly from the GGUF mmap. Used
// in the generation loop to assemble the next-token embedding (sum of 16
// codebook embeddings) without paying for a backend round-trip per row.
static void embed_row_from_gguf(const GGUFModel & gf, const char * tensor_name, int row_id, int hidden, float * dst) {
    struct ggml_tensor * src = ggml_get_tensor(gf.meta, tensor_name);
    if (!src) {
        qt_throw("[Pipeline] tensor not found in GGUF: %s", tensor_name);
    }
    const uint8_t * base = (const uint8_t *) gf_get_data(gf, tensor_name);
    if (!base) {
        qt_throw("[Pipeline] tensor data missing in GGUF: %s", tensor_name);
    }
    const size_t row_bytes = ggml_row_size(src->type, hidden);
    const void * row       = base + (size_t) row_id * row_bytes;
    if (src->type == GGML_TYPE_F32) {
        std::memcpy(dst, row, (size_t) hidden * sizeof(float));
        return;
    }
    const struct ggml_type_traits * tt = ggml_get_type_traits(src->type);
    if (!tt || !tt->to_float) {
        qt_throw("[Pipeline] unsupported codec_embedding dtype %d for %s", (int) src->type, tensor_name);
    }
    tt->to_float(row, dst, hidden);
}

bool pipeline_tts_synthesize(PipelineTTS *                       pt,
                             BPETokenizer *                      tok,
                             const PipelineTTSSynthesizeParams & params,
                             PipelineTTSSynthesizeOutput *       out) {
    out->audio.clear();
    out->sample_rate = TOKENIZER_SAMPLE_RATE;

    PromptBuilderOutput prompt;
    const std::string   instruct = params.instruct ? params.instruct : "";
    const std::string   speaker  = params.speaker ? params.speaker : "";
    const std::string   ref_text = params.ref_text ? params.ref_text : "";

    // Voice clone mode A: if ref_audio_24k is given, run the speaker
    // encoder on the pre-decoded mono buffer and feed the resulting
    // embedding straight into the prompt builder. Mutually exclusive
    // with --speaker.
    const bool         has_ref_audio = (params.ref_audio_24k != NULL) && (params.ref_n_samples > 0);
    std::vector<float> ref_spk_emb;
    const float *      ref_spk_emb_ptr = NULL;
    if (has_ref_audio) {
        if (!pt->has_speaker_encoder) {
            qt_set_error(
                "pipeline_tts_synthesize: --ref-wav requires a model with a loaded speaker encoder (Base only)");
            qt_log(QT_LOG_ERROR, "[Pipeline] --ref-wav requires a model with a loaded speaker encoder (Base only)");
            return false;
        }
        if (!speaker_encoder_extract(&pt->speaker_encoder, pt->sched, params.ref_audio_24k, params.ref_n_samples,
                                     ref_spk_emb, params.dump_dir)) {
            return false;
        }
        if ((int) ref_spk_emb.size() != pt->talker.hidden_size) {
            qt_set_error("pipeline_tts_synthesize: speaker embedding size %zu mismatches talker hidden %d",
                         ref_spk_emb.size(), pt->talker.hidden_size);
            qt_log(QT_LOG_ERROR, "[Pipeline] speaker embedding size %zu mismatches talker hidden %d",
                   ref_spk_emb.size(), pt->talker.hidden_size);
            return false;
        }
        ref_spk_emb_ptr = ref_spk_emb.data();
    }

    // Voice clone mode B: if ref_text is also given, encode the
    // reference audio into 16 codebook indices via the codec encoder.
    // Layout returned by pipeline_codec_encode is [num_codebooks, T_codec]
    // row major, matching what the prompt builder expects for the ICL
    // sum loop.
    std::vector<int32_t> ref_codes;
    int                  ref_codes_T = 0;
    if (!ref_text.empty()) {
        if (!has_ref_audio) {
            qt_set_error("pipeline_tts_synthesize: --ref-text requires --ref-wav");
            qt_log(QT_LOG_ERROR, "[Pipeline] --ref-text requires --ref-wav");
            return false;
        }
        // The codec hop is 1920 samples at 24 kHz so n_samples must be
        // a multiple of 1920. Truncate to the nearest hop boundary.
        if (params.ref_n_samples < TOKENIZER_HOP_LENGTH) {
            qt_set_error("pipeline_tts_synthesize: ref_wav too short for ICL (%d samples)", params.ref_n_samples);
            qt_log(QT_LOG_ERROR, "[Pipeline] ref_wav too short for ICL (%d samples)", params.ref_n_samples);
            return false;
        }
        int aligned_T = (params.ref_n_samples / TOKENIZER_HOP_LENGTH) * TOKENIZER_HOP_LENGTH;
        ref_codes     = pipeline_codec_encode(&pt->codec, params.ref_audio_24k, aligned_T, params.dump_dir);
        if (ref_codes.empty()) {
            qt_set_error("pipeline_tts_synthesize: pipeline_codec_encode returned empty codes");
            qt_log(QT_LOG_ERROR, "[Pipeline] pipeline_codec_encode returned empty codes");
            return false;
        }
        ref_codes_T = (int) ref_codes.size() / pt->num_code_groups;
        qt_log(QT_LOG_INFO, "[Pipeline] ICL ref_codes: %d frames at 12.5 Hz (%d audio samples)", ref_codes_T,
               aligned_T);
    }

    if (!prompt_builder_build(pt, tok, params.text, params.lang, instruct, speaker, ref_spk_emb_ptr, ref_text,
                              ref_codes_T > 0 ? ref_codes.data() : NULL, ref_codes_T, &prompt)) {
        return false;
    }

    if (params.dump_dir) {
        DebugDumper d;
        debug_init(&d, params.dump_dir);
        std::vector<int32_t> ids32(prompt.prompt_ids.begin(), prompt.prompt_ids.end());
        int                  n_ids = (int) ids32.size();
        debug_dump_i32_as_f32(&d, "prompt-ids", ids32.data(), &n_ids, 1);
        debug_dump_2d(&d, "talker-input-embed", prompt.input_embed.data(), prompt.T_ctx, prompt.hidden);
        debug_dump_2d(&d, "trailing-text-hidden", prompt.trailing_text_hidden.data(), prompt.T_trailing, prompt.hidden);
        debug_dump_1d(&d, "tts-pad-embed", prompt.tts_pad_embed.data(), prompt.hidden);

        // Voice clone dumps: speaker-emb fires when ref_wav is set
        // (modes A and B), ref-codes fires only when ref_text is also set
        // (mode B ICL). Both are no-ops in base / tts / customvoice modes,
        // the dump files simply do not appear in those runs.
        if (ref_spk_emb_ptr != NULL) {
            debug_dump_1d(&d, "speaker-emb", ref_spk_emb_ptr, pt->talker.hidden_size);
        }
        if (ref_codes_T > 0) {
            const int shape[2] = { pt->num_code_groups, ref_codes_T };
            debug_dump_i32_as_f32(&d, "ref-codes", ref_codes.data(), shape, 2);
        }
    }

    // Generation loop: at each step we recompute the full Talker prefix
    // (no KV cache yet) over the prompt prefix concatenated with all the
    // next-token embeddings produced so far, sample c0, run the code
    // predictor for the 15 acoustic codes, build the next-token
    // embedding by summing the 16 codebook embeddings and the matching
    // trailing-text overlay, and append it to the running context. We
    // stop on codec_eos or when max_new_tokens is reached.
    const int hidden        = prompt.hidden;
    const int codec_eos_id  = pt->codec_specials.eos_id;
    const int num_codebooks = pt->num_code_groups;
    const int talker_vocab  = pt->talker.vocab_size;

    // Greedy collapses to temperature <= 0 in sample_top_k_p.
    float talker_T  = params.do_sample ? params.temperature : 0.0f;
    float subtk_T   = params.subtalker_do_sample ? params.subtalker_temperature : 0.0f;
    float talker_rp = params.repetition_penalty;

    // Generation loop: step 0 prefills the talker over the full prompt
    // and writes T_ctx positions into the KV cache. Subsequent steps
    // feed one next_emb at a time and append one position. The code
    // predictor maintains its own per-frame cache that gets reset at
    // every step.
    std::vector<std::vector<int32_t>> all_codes;
    all_codes.reserve((size_t) params.max_new_tokens);

    // c0 codes already emitted, fed to repetition penalty.
    std::vector<int32_t> talker_history;
    talker_history.reserve((size_t) params.max_new_tokens);

    // Global Philox subsequence counter advances once per primitive
    // sample (one for c0 of each step, then 15 for the predictor codes).
    int64_t subseq_counter = 0;

    std::vector<float> next_emb((size_t) hidden, 0.0f);

    for (int step = 0; step < params.max_new_tokens; step++) {
        TalkerForwardOutput fw;
        const char *        step_dump = (params.dump_dir && step == 0) ? params.dump_dir : NULL;
        bool                ok;
        if (step == 0) {
            ok = talker_forward_prefill(&pt->talker, &pt->talker_kv, pt->sched, prompt.input_embed.data(), prompt.T_ctx,
                                        step_dump, &fw);
        } else {
            ok = talker_forward_decode(&pt->talker, &pt->talker_kv, pt->sched, next_emb.data(), &fw);
        }
        if (!ok) {
            return false;
        }

        // Bisection dump: the talker hidden_last at step 1 is the input
        // the code predictor consumes after consuming the next-emb of
        // step 0. Pairing it byte for byte with the Python hook tells us
        // whether the next-emb composition + talker decode round trip
        // is bit exact end to end.
        if (params.dump_dir && step == 1) {
            DebugDumper d;
            debug_init(&d, params.dump_dir);
            debug_dump_1d(&d, "talker-hidden-step1", fw.hidden_last.data(), hidden);
        }

        // Apply codec suppression: forbid [vocab - 1024, vocab) except
        // codec_eos. Then run the upstream sampling chain.
        apply_suppress(fw.logits_last.data(), talker_vocab, talker_vocab - 1024, talker_vocab, codec_eos_id);
        float u_c0 = 0.0f;
        int   c0 = sample_top_k_p(fw.logits_last.data(), talker_vocab, talker_T, params.top_k, params.top_p, talker_rp,
                                  talker_history.data(), (int) talker_history.size(), params.seed, subseq_counter, &u_c0);
        subseq_counter++;
        if (c0 < 0) {
            qt_log(QT_LOG_ERROR, "[Pipeline] c0 sample returned no candidate");
            return false;
        }

        // Trace the first 32 samples unconditionally so [Sample] lines
        // up with [Sample-PY] / [Sample-CP] across the 16 codes of step
        // 0 and step 1 the Python harness emits.
        if ((subseq_counter - 1) < 32) {
            qt_log(QT_LOG_DEBUG, "[Sample] step=%d c0=%d u=%.10f subseq=%lld", step, c0, (double) u_c0,
                   (long long) (subseq_counter - 1));
        }

        if (c0 == codec_eos_id) {
            qt_log(QT_LOG_INFO, "[Pipeline] EOS at step %d, stopping", step);
            break;
        }

        CodePredictorOutput cp;
        const char *        cp_dump = (params.dump_dir && step == 0) ? params.dump_dir : NULL;
        if (!code_predictor_step(&pt->talker, &pt->code_predictor, &pt->code_predictor_kv, pt->sched,
                                 fw.hidden_last.data(), c0, subtk_T, params.subtalker_top_k, params.subtalker_top_p,
                                 params.seed, subseq_counter - 1, cp_dump, &cp)) {
            return false;
        }
        // Predictor consumed (num_codebooks - 1) subsequences after the
        // c0 one (subseq_base + 1 .. subseq_base + 15).
        subseq_counter += (num_codebooks - 1);

        all_codes.push_back(cp.codes);
        talker_history.push_back(c0);

        // Build next-token embedding: sum of 16 codebook embeddings.
        // codebook 0 uses talker.codec_embedding, the 15 acoustic
        // codebooks use the predictor's private embedding tables.
        std::fill(next_emb.begin(), next_emb.end(), 0.0f);
        std::vector<float> tmp((size_t) hidden);

        embed_row_from_gguf(pt->gguf_talker, "talker.codec_embd.weight", c0, hidden, tmp.data());
        for (int i = 0; i < hidden; i++) {
            next_emb[(size_t) i] += tmp[(size_t) i];
        }
        for (int g = 0; g < num_codebooks - 1; g++) {
            int  cg = cp.codes[(size_t) (g + 1)];
            char name[64];
            snprintf(name, sizeof(name), "code_pred.codec_embd.%d.weight", g);
            embed_row_from_gguf(pt->gguf_talker, name, cg, hidden, tmp.data());
            for (int i = 0; i < hidden; i++) {
                next_emb[(size_t) i] += tmp[(size_t) i];
            }
        }

        // Trailing text overlay: while we still have utterance text
        // hiddens to consume, add the next one; otherwise add the
        // tts_pad embedding.
        const float * overlay = (step < prompt.T_trailing) ?
                                    prompt.trailing_text_hidden.data() + (size_t) step * (size_t) hidden :
                                    prompt.tts_pad_embed.data();
        for (int i = 0; i < hidden; i++) {
            next_emb[(size_t) i] += overlay[(size_t) i];
        }

        // Bisection dump: the next-token embedding produced at step 0
        // is the only thing controlling the talker forward at step 1, so
        // matching it bit-exact against Python pinpoints any drift in
        // the codebook embedding sums or the trailing text overlay.
        if (params.dump_dir && step == 0) {
            DebugDumper d;
            debug_init(&d, params.dump_dir);
            debug_dump_1d(&d, "next-emb-step0", next_emb.data(), hidden);
        }

        if (((step + 1) % 8) == 0) {
            qt_log(QT_LOG_INFO, "[Pipeline] Generated %d frames", step + 1);
        }
    }

    qt_log(QT_LOG_INFO, "[Pipeline] Generation done : %zu frames", all_codes.size());

    if (params.dump_dir && !all_codes.empty()) {
        DebugDumper d;
        debug_init(&d, params.dump_dir);
        int                  T_frames = (int) all_codes.size();
        std::vector<int32_t> flat((size_t) T_frames * (size_t) num_codebooks);
        for (int t = 0; t < T_frames; t++) {
            for (int k = 0; k < num_codebooks; k++) {
                flat[(size_t) t * (size_t) num_codebooks + (size_t) k] = all_codes[(size_t) t][(size_t) k];
            }
        }
        int shape[2] = { T_frames, num_codebooks };
        debug_dump_i32_as_f32(&d, "codes-full", flat.data(), shape, 2);
    }

    // Codec decode: transpose codes from [T_frames, K] to [K, T_frames]
    // because pipeline_codec_decode expects K-major layout (codebooks
    // first, frames second), then return the 24 kHz mono audio.
    if (all_codes.empty()) {
        return true;
    }

    int                  T_frames = (int) all_codes.size();
    std::vector<int32_t> codes_kt((size_t) num_codebooks * (size_t) T_frames);
    for (int t = 0; t < T_frames; t++) {
        for (int k = 0; k < num_codebooks; k++) {
            codes_kt[(size_t) k * (size_t) T_frames + (size_t) t] = all_codes[(size_t) t][(size_t) k];
        }
    }
    out->audio = pipeline_codec_decode(&pt->codec, codes_kt.data(), num_codebooks, T_frames);
    if (out->audio.empty()) {
        qt_log(QT_LOG_ERROR, "[Pipeline] codec decode returned no audio");
        return false;
    }

    if (params.dump_dir) {
        DebugDumper d;
        debug_init(&d, params.dump_dir);
        debug_dump_1d(&d, "output-audio", out->audio.data(), (int) out->audio.size());
    }

    return true;
}
